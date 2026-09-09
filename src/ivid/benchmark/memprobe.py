"""FR-12 — Do bo nho MOI runtime thuc su can, moi cau hinh mot TIEN TRINH RIENG.

VI SAO PHAI TACH TIEN TRINH

Ban dau cot bo nho trong bao cao lay `system_used_delta_mb`: muc tang cua
MemTotal - MemAvailable tren toan he thong. Do la con so vo nghia — no dem ca
moi tien trinh khac, ca bo dem trang cua chinh nhung file anh vua doc. Hai lan
chay cung cau hinh, cach nhau 40 phut:

    cau hinh            lan day du   lan nhanh
    yolov8n tensorrt      31.2 MB      0.0 MB
    yolov8s pytorch       32.3 MB      0.4 MB

Chuyen sang `process_rss_peak_mb` thi on dinh (lech 1.2% giua hai lan) nhung
VAN khong doc duoc: ivid.benchmark.latency chay ca sau cau hinh trong CUNG MOT
tien trinh, nen RSS cong don. ONNX hien 2683 MB roi TensorRT hien 2272 MB khong
co nghia TensorRT ton it hon — chi la no duoc nap sau, khi bo nho cua backend
truoc chua duoc tra het.

Bo nho, khac voi thoi gian, khong tro ve trang thai cu sau moi phep do. Cach
duy nhat de co con so so sanh duoc la moi runtime mot tien trinh sach.

Con so bao cao la RSS dinh diem TRU RSS luc tien trinh vua khoi dong, nen no
BAO GOM chi phi nap thu vien (torch, onnxruntime, tensorrt). Do la dung y: cau
hoi trien khai that su la "chay runtime nay tren Jetson 8GB ton bao nhieu RAM",
chu khong phai "engine chiem bao nhieu byte".

    PYTHONPATH=src python -m ivid.benchmark.memprobe            # do tat ca
    PYTHONPATH=src python -m ivid.benchmark.memprobe --con --model yolov8n --backend onnx
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ..data.common import repo_root, write_json
from .resources import _rss_mb

BACKEND_LABEL = {"pytorch": "PyTorch", "onnx": "ONNX Runtime", "tensorrt": "TensorRT FP16"}


def do_mot_cau_hinh(model: str, backend: str, imgsz: int, conf: float, iou: float,
                    n_anh: int, root: Path, resize_to: int | None = None) -> dict:
    """Chay trong tien trinh CON. Tra ve muc bo nho tang them cua rieng no."""
    nen = _rss_mb()                      # truoc khi nap bat cu thu vien nang nao

    from ..preprocess import read_image
    from .runners.base import build_runner, weights_for

    w = weights_for(root / "models" / model, backend)
    if not w.exists():
        return {"skipped": True, "reason": f"khong thay {w.name}"}

    t0 = time.perf_counter()
    runner = build_runner(backend, w, imgsz=imgsz, conf=conf, iou=iou, resize_to=resize_to)
    giay_nap = time.perf_counter() - t0
    sau_nap = _rss_mb()

    anh = root / "data/processed/test/images"
    ds = sorted(p for p in anh.iterdir() if p.suffix.lower() in {".jpg", ".png", ".bmp"})[:n_anh]
    img = read_image(ds[0])
    runner.warmup(10, img)

    dinh = sau_nap
    for p in ds:
        runner.run_array(read_image(p))
        dinh = max(dinh, _rss_mb())
    runner.close()

    return {
        "skipped": False, "model": model, "backend": backend,
        "backend_label": BACKEND_LABEL.get(backend, backend),
        "rss_khoi_dong_mb": round(nen, 1),
        "rss_sau_nap_mb": round(sau_nap, 1),
        "rss_dinh_mb": round(dinh, 1),
        "rss_delta_mb": round(dinh - nen, 1),
        "nap_giay": round(giay_nap, 2),
        "n_anh": len(ds),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--con", action="store_true", help="che do tien trinh con (noi bo)")
    ap.add_argument("--model")
    ap.add_argument("--backend")
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--n-anh", type=int, default=30)
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    imgsz, conf, iou = int(cfg["imgsz"]), float(cfg["conf"]), float(cfg["iou"])
    resize_to = int(cfg["resize_to"]) if cfg.get("resize_to") else None

    if a.con:
        r = do_mot_cau_hinh(a.model, a.backend, imgsz, conf, iou, a.n_anh, root, resize_to)
        print(json.dumps(r))
        return 0

    print(f"[mem] moi cau hinh mot tien trinh rieng, {a.n_anh} anh moi lan")
    ket_qua: dict[str, dict] = {}
    for model in cfg["models"]:
        for backend in cfg["backends"]:
            key = f"{model}|{backend}"
            lenh = [sys.executable, "-m", "ivid.benchmark.memprobe", "--con",
                    "--model", model, "--backend", backend,
                    "--config", a.config, "--n-anh", str(a.n_anh)]
            p = subprocess.run(lenh, capture_output=True, text=True, cwd=root,  # noqa: S603
                               env={**os.environ,
                                    "PYTHONPATH": str(root / "src"), "PYTHONUTF8": "1"})
            dong = [x for x in p.stdout.splitlines() if x.startswith("{")]
            if p.returncode != 0 or not dong:
                ket_qua[key] = {"skipped": True,
                                "reason": (p.stderr.strip().splitlines() or ["loi khong ro"])[-1]}
                print(f"    {key:<26} BO QUA: {ket_qua[key]['reason']}")
                continue
            r = json.loads(dong[-1])
            ket_qua[key] = r
            if r.get("skipped"):
                print(f"    {key:<26} BO QUA: {r['reason']}")
            else:
                print(f"    {key:<26} +{r['rss_delta_mb']:>7.1f} MB  "
                      f"(nap {r['nap_giay']}s, nen {r['rss_khoi_dong_mb']} MB)")

    out = root / a.out
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    payload["memory"] = ket_qua
    payload["memory_method"] = (
        "Moi cau hinh chay trong mot tien trinh rieng; con so la RSS dinh diem tru "
        "RSS luc tien trinh vua khoi dong, nen bao gom ca chi phi nap thu vien. "
        "Do chung mot tien trinh cho ca sau cau hinh se cho so cong don, khong so "
        "sanh duoc — xem docstring ivid/benchmark/memprobe.py.")
    write_json(out, payload)
    print(f"[mem] ghi -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
