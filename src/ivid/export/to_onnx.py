"""FR-07 — Export best.pt sang ONNX voi opset co dinh, roi kiem chung.

Hai buoc kiem chung, khong bo buoc nao:
  1. onnx.checker  — do thi hop le ve mat cau truc
  2. so sanh so hoc — chay PyTorch va ONNX Runtime tren cung N anh THAT
     (khong phai nhieu ngau nhien) va do do lech lon nhat cua tensor dau ra

Buoc 2 moi la buoc quan trong. Mot file ONNX co the pass checker ma van sai so
hoc (sai thu tu kenh, sai chuan hoa, layer bi thay the sai). Neu bo qua, sai so
do se lang le di vao bang benchmark duoi dang "TensorRT lam giam mAP".

    PYTHONPATH=src python -m ivid.export.to_onnx --name yolov8n
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from ..data.common import IMG_EXT, rel_to_root, repo_root, write_json
from ..preprocess import preprocess, read_image


def sample_images(data_yaml: Path, n: int) -> list[Path]:
    """Lay N anh dau tien cua tap test, theo thu tu ten -> lap lai duoc."""
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    img_dir = Path(d["path"]) / d.get("test", "test/images")
    imgs = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT), key=lambda p: p.name)
    return imgs[:n]


def compare_outputs(pt_path: Path, onnx_path: Path, images: list[Path], imgsz: int) -> dict:
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    # ep ve CPU: so sanh so hoc chi can dung ket qua, va CPU tranh duoc
    # khac biet do TF32/FP16 tu dong tren GPU lam nhieu phep do lech
    torch_model = model.model.float().cpu().eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    diffs = []
    for p in images:
        x, _, _ = preprocess(read_image(p), imgsz)
        with torch.no_grad():
            y_pt = torch_model(torch.from_numpy(x))
        y_pt = y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt
        y_pt = y_pt.cpu().numpy()
        y_ort = sess.run(None, {in_name: x})[0]

        if y_pt.shape != y_ort.shape:
            return {"ok": False, "reason": f"shape lech: torch {y_pt.shape} vs onnx {y_ort.shape}"}
        diffs.append(float(np.abs(y_pt - y_ort).max()))

    return {
        "ok": True,
        "n_images": len(images),
        "images": [p.name for p in images],
        "max_abs_diff": max(diffs),
        "mean_abs_diff_per_image": round(float(np.mean(diffs)), 8),
        "per_image": [round(d, 8) for d in diffs],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="ten model, vd yolov8n")
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    oc = cfg["onnx"]

    pt = root / "models" / a.name / "best.pt"
    if not pt.exists():
        print(f"[loi] khong thay {pt} — train truoc da", file=sys.stderr)
        return 1
    onnx_path = pt.with_suffix(".onnx")

    print(f"[onnx] nguon : {pt}")
    print(f"[onnx] opset={oc['opset']} imgsz={oc['imgsz']} batch={oc['batch']} "
          f"dynamic={oc['dynamic']} simplify={oc['simplify']}")

    from ultralytics import YOLO

    produced = YOLO(str(pt)).export(
        format="onnx", opset=int(oc["opset"]), imgsz=int(oc["imgsz"]),
        batch=int(oc["batch"]), dynamic=bool(oc["dynamic"]),
        simplify=bool(oc["simplify"]), half=bool(oc.get("half", False)),
    )
    produced = Path(produced)
    if produced.resolve() != onnx_path.resolve():
        shutil.move(str(produced), onnx_path)

    import onnx

    onnx.checker.check_model(str(onnx_path))
    m = onnx.load(str(onnx_path))
    ir = {
        "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import],
        "inputs": [{"name": i.name,
                    "shape": [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}
                   for i in m.graph.input],
        "outputs": [{"name": o.name,
                     "shape": [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]}
                    for o in m.graph.output],
    }
    print(f"[onnx] checker PASS  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"[onnx] input  {ir['inputs']}")
    print(f"[onnx] output {ir['outputs']}")

    report = {"model": a.name, "onnx": rel_to_root(onnx_path),
              "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
              "checker": "pass", "graph": ir, "config": oc}

    if not a.skip_parity:
        tol = float(cfg["parity"]["tolerance"])
        n = int(cfg["parity"]["n_images_onnx"])
        imgs = sample_images(root / a.data, n)
        print(f"[onnx] so sanh so hoc PyTorch vs ONNX tren {len(imgs)} anh that...")
        par = compare_outputs(pt, onnx_path, imgs, int(oc["imgsz"]))
        report["parity"] = par
        if not par["ok"]:
            print(f"[loi] {par['reason']}", file=sys.stderr)
            write_json(root / "results" / f"export_{a.name}_onnx.json", report)
            return 1
        passed = par["max_abs_diff"] <= tol
        report["parity"]["tolerance"] = tol
        report["parity"]["passed"] = passed
        print(f"[onnx] do lech lon nhat = {par['max_abs_diff']:.3e}  (nguong {tol:.0e})  "
              f"-> {'DAT' if passed else 'KHONG DAT'}")
        if not passed:
            print("[canh bao] Vuot nguong FR-07. Nguyen nhan hay gap: simplify lam doi "
                  "phep tinh, hoac opset qua thap khien mot layer bi thay the.", file=sys.stderr)

    write_json(root / "results" / f"export_{a.name}_onnx.json", report)
    print(f"[onnx] -> {onnx_path}")
    print(f"[onnx] bao cao -> results/export_{a.name}_onnx.json")
    print(f"[onnx] buoc tiep (TREN JETSON): "
          f"PYTHONPATH=src python -m ivid.export.to_tensorrt --name {a.name}")
    return 0 if a.skip_parity or report.get("parity", {}).get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
