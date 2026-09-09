"""FR-13 — Do mAP rieng cho tung dinh dang tren cung tap test.

Cot loi cua ca du an nam o day: neu chi do latency, khong ai biet TensorRT FP16
danh doi bao nhieu do chinh xac. SRS §3.1 noi ro — "phai dinh luong, khong gia
dinh la bang nhau".

Dung lai ivid.train.evaluate cho ca ba dinh dang de dam bao CUNG mot ham cham
diem. Neu moi dinh dang duoc cham bang mot ham khac, chenh lech mAP do duoc co
the den tu cach cham chu khong phai tu runtime.

    PYTHONPATH=src python -m ivid.benchmark.accuracy --models yolov8n yolov8s
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from ..data.common import repo_root, write_json
from ..train.evaluate import evaluate
from .runners.base import weights_for

BACKEND_LABEL = {"pytorch": "PyTorch", "onnx": "ONNX Runtime", "tensorrt": "TensorRT FP16"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="thap de duong PR day du; KHAC voi conf luc do latency")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    models = a.models or cfg["models"]
    backends = a.backends or cfg["backends"]
    imgsz = int(cfg["imgsz"])

    data = root / a.data
    if not data.exists():
        print(f"[loi] khong thay {data}", file=sys.stderr)
        return 1

    out_path = root / a.out
    payload = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.setdefault("accuracy", {})

    for model in models:
        for backend in backends:
            key = f"{model}|{backend}"
            w = weights_for(root / "models" / model, backend)
            print(f"\n[acc] === {model} / {BACKEND_LABEL.get(backend, backend)} ===")
            if not w.exists():
                print(f"    BO QUA: khong thay {w.relative_to(root)}")
                payload["accuracy"][key] = {"skipped": True,
                                            "reason": f"khong thay {w.name}"}
                write_json(out_path, payload)
                continue
            try:
                res = evaluate(w, data, cfg["split"], imgsz, batch=1,
                               device=a.device, conf=a.conf, iou=a.iou)
            except Exception as e:
                print(f"    LOI: {type(e).__name__}: {e}")
                payload["accuracy"][key] = {"skipped": True,
                                            "reason": f"{type(e).__name__}: {e}"}
                write_json(out_path, payload)
                continue

            res["skipped"] = False
            payload["accuracy"][key] = res
            o = res["overall"]
            print(f"    mAP@0.5 {o['mAP50']:.4f}   mAP@0.5:0.95 {o['mAP50_95']:.4f}   "
                  f"P {o['precision']:.4f}  R {o['recall']:.4f}")
            write_json(out_path, payload)

    # --- so sanh suy giam so voi PyTorch ---
    print("\n[acc] Suy giam do chinh xac so voi PyTorch (moc goc):")
    for model in models:
        base = payload["accuracy"].get(f"{model}|pytorch")
        if not base or base.get("skipped"):
            continue
        b50 = base["overall"]["mAP50"]
        for backend in backends:
            r = payload["accuracy"].get(f"{model}|{backend}")
            if not r or r.get("skipped"):
                continue
            d = r["overall"]["mAP50"] - b50
            print(f"    {model:<9} {BACKEND_LABEL.get(backend, backend):<15} "
                  f"mAP@0.5 {r['overall']['mAP50']:.4f}  ({d:+.4f})")

    print(f"\n[acc] ghi -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
