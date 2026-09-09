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
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    img_dir = Path(d["path"]) / d.get("test", "test/images")
    imgs = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT), key=lambda p: p.name)
    return imgs[:n]


def compare_outputs(pt_path: Path, onnx_path: Path, images: list[Path], imgsz: int,
                    nc: int, conf: float, iou: float) -> dict:
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    from ..postprocess import decode, match_detections

    model = YOLO(str(pt_path))
    torch_model = model.model.float().cpu().eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    raw_max, box_px, box_rel, score_abs = [], [], [], []
    det_match = {"matched": 0, "only_pt": 0, "only_onnx": 0}
    n_pt = n_ort = 0
    per_image = []

    for p in images:
        x, _, _ = preprocess(read_image(p), imgsz)
        with torch.no_grad():
            y_pt = torch_model(torch.from_numpy(x))
        y_pt = y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt
        y_pt = y_pt.cpu().numpy()
        y_ort = sess.run(None, {in_name: x})[0]

        if y_pt.shape != y_ort.shape:
            return {"ok": False, "reason": f"shape lech: torch {y_pt.shape} vs onnx {y_ort.shape}"}

        d = np.abs(y_pt - y_ort)
        raw_max.append(float(d.max()))
        box_px.append(float(d[:, :4, :].max()))
        denom = np.maximum(np.abs(y_pt[:, :4, :]), 1e-6)
        box_rel.append(float((d[:, :4, :] / denom).max()))
        score_abs.append(float(d[:, 4:, :].max()))

        a = decode(y_pt, conf, iou, nc=nc)
        b = decode(y_ort, conf, iou, nc=nc)
        m = match_detections(a, b, iou_thres=0.5)
        det_match["matched"] += m["matched"]
        det_match["only_pt"] += m["only_a"]
        det_match["only_onnx"] += m["only_b"]
        n_pt += len(a)
        n_ort += len(b)
        per_image.append({"image": p.name, "raw_max": round(float(d.max()), 8),
                          "box_px": round(box_px[-1], 8), "score_abs": round(score_abs[-1], 8),
                          "n_det_pt": len(a), "n_det_onnx": len(b)})

    total = det_match["matched"] + det_match["only_pt"] + det_match["only_onnx"]
    return {
        "ok": True,
        "n_images": len(images),
        "images": [p.name for p in images],
        "max_abs_diff": max(raw_max),
        "mean_abs_diff_per_image": round(float(np.mean(raw_max)), 8),
        "box_max_diff_px": max(box_px),
        "box_max_relative": max(box_rel),
        "score_max_abs_diff": max(score_abs),
        "detections": {
            "conf": conf, "iou": iou,
            "n_pytorch": n_pt, "n_onnx": n_ort,
            "matched": det_match["matched"],
            "only_pytorch": det_match["only_pt"],
            "only_onnx": det_match["only_onnx"],
            "match_rate": round(det_match["matched"] / total, 6) if total else 1.0,
        },
        "per_image": per_image,
    }


def judge(par: dict, tol_raw: float, tol_box_px: float, tol_score: float) -> dict:
    checks = {
        "toa_do_hop": {
            "gia_tri": par["box_max_diff_px"], "nguong": tol_box_px, "don_vi": "pixel",
            "dat": par["box_max_diff_px"] <= tol_box_px,
            "y_nghia": "lech duoi mot phan tram pixel thi khong the doi ket qua detection",
        },
        "diem_so_lop": {
            "gia_tri": par["score_max_abs_diff"], "nguong": tol_score, "don_vi": "tuyet doi (0..1)",
            "dat": par["score_max_abs_diff"] <= tol_score,
            "y_nghia": "diem so quyet dinh lop va viec vuot nguong conf",
        },
        "detection_cuoi_cung": {
            "gia_tri": par["detections"]["match_rate"], "nguong": 1.0, "don_vi": "ti le khop",
            "dat": par["detections"]["match_rate"] >= 1.0,
            "y_nghia": "sau giai ma va NMS, hai dinh dang co cho ra cung cac hop khong",
        },
    }
    return {
        "checks": checks,
        "dat_tat_ca": all(c["dat"] for c in checks.values()),
        "srs_raw_metric": {
            "gia_tri": par["max_abs_diff"], "nguong": tol_raw,
            "dat": par["max_abs_diff"] <= tol_raw,
            "ghi_chu": ("Chi so tho theo dung chu SRS FR-07. No gop toa do (pixel, gia tri "
                        "toi 640) voi diem so (0..1) vao mot nguong tuyet doi duy nhat, nen "
                        "bi chi phoi boi toa do va khong phan anh dung anh huong thuc te. "
                        "Giu lai de doi chieu, khong dung de ket luan."),
        },
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

    passed = True
    if not a.skip_parity:
        pc = cfg["parity"]
        n = int(pc["n_images_onnx"])
        imgs = sample_images(root / a.data, n)
        import yaml as _yaml

        nc = len(_yaml.safe_load((root / a.data).read_text(encoding="utf-8"))["names"])
        print(f"[onnx] so sanh so hoc PyTorch vs ONNX tren {len(imgs)} anh that...")
        par = compare_outputs(pt, onnx_path, imgs, int(oc["imgsz"]), nc,
                              float(pc.get("conf", 0.25)), float(pc.get("iou", 0.7)))
        if not par["ok"]:
            print(f"[loi] {par['reason']}", file=sys.stderr)
            report["parity"] = par
            write_json(root / "results" / f"export_{a.name}_onnx.json", report)
            return 1

        verdict = judge(par, float(pc["tolerance"]),
                        float(pc.get("tolerance_box_px", 0.1)),
                        float(pc.get("tolerance_score", 1.0e-3)))
        par["verdict"] = verdict
        report["parity"] = par
        passed = verdict["dat_tat_ca"]

        print(f"[onnx] do lech tho toan tensor = {par['max_abs_diff']:.3e}   "
              f"(chi so tho cua SRS, gop lan toa do va diem so)")
        print(f"[onnx] {'muc kiem tra':<22}{'gia tri':>12}{'nguong':>12}   ket qua")
        for name, c in verdict["checks"].items():
            print(f"       {name:<22}{c['gia_tri']:>12.3e}{c['nguong']:>12.3e}   "
                  f"{'DAT' if c['dat'] else 'KHONG DAT'}")
        det = par["detections"]
        print(f"[onnx] detection: PyTorch {det['n_pytorch']}, ONNX {det['n_onnx']}, "
              f"khop {det['matched']} ({det['match_rate']*100:.1f}%)")
        if not passed:
            print("[canh bao] Co muc kiem tra KHONG DAT — xem "
                  f"results/export_{a.name}_onnx.json truoc khi build engine.", file=sys.stderr)

    write_json(root / "results" / f"export_{a.name}_onnx.json", report)
    print(f"[onnx] -> {onnx_path}")
    print(f"[onnx] bao cao -> results/export_{a.name}_onnx.json")
    print(f"[onnx] buoc tiep (TREN JETSON): "
          f"PYTHONPATH=src python -m ivid.export.to_tensorrt --name {a.name}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
