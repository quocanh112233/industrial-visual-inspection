"""FR-05 — Danh gia model tren tap test: mAP tong the va theo tung lop.

Chay duoc voi bat ky dinh dang trong so nao (.pt / .onnx / .engine), nen buoc
do mAP cho ba runtime (FR-13) dung lai chinh script nay thay vi viet lai —
va quan trong hon: dam bao ba dinh dang duoc cham diem bang CUNG mot ham,
neu khong thi chenh lech mAP do duoc co the la do cach cham chu khong phai runtime.

    PYTHONPATH=src python -m ivid.train.evaluate --name yolov8n
    PYTHONPATH=src python -m ivid.train.evaluate --weights models/yolov8n/best.engine
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..data.common import repo_root, write_json

MAP50_TARGET = 0.65  # nguong FR-05 cho YOLOv8n tren NEU-DET


def _to_float(x) -> float:
    try:
        return round(float(x), 4)
    except Exception:
        return float("nan")


def evaluate(weights: Path, data: Path, split: str, imgsz: int, batch: int,
             device: str | None, conf: float, iou: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kw = dict(data=str(data), split=split, imgsz=imgsz, batch=batch,
              conf=conf, iou=iou, plots=False, verbose=False)
    if device is not None:
        kw["device"] = device
    m = model.val(**kw)

    box = m.box
    names = getattr(m, "names", None) or getattr(model, "names", {})
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    # Cac mang p/r/ap50 chi chua cac lop CO MAT trong ket qua, danh so qua
    # ap_class_index. Con maps thi danh so thang theo class id. Tron hai kieu
    # nay la nguon sai lech im lang kinh dien -> anh xa tuong minh.
    idx = list(getattr(box, "ap_class_index", range(len(names))))
    per_class: dict[str, dict] = {}
    for pos, cid in enumerate(idx):
        cname = names[cid] if cid < len(names) else str(cid)
        entry = {}
        for key, attr in (("precision", "p"), ("recall", "r"), ("mAP50", "ap50")):
            arr = getattr(box, attr, None)
            if arr is not None and pos < len(arr):
                entry[key] = _to_float(arr[pos])
        maps = getattr(box, "maps", None)
        if maps is not None and cid < len(maps):
            entry["mAP50_95"] = _to_float(maps[cid])
        per_class[cname] = entry

    return {
        "weights": str(weights),
        "format": weights.suffix.lstrip("."),
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "overall": {
            "mAP50": _to_float(box.map50),
            "mAP50_95": _to_float(box.map),
            "mAP75": _to_float(getattr(box, "map75", float("nan"))),
            "precision": _to_float(box.mp),
            "recall": _to_float(box.mr),
        },
        "per_class": per_class,
        # ms/anh do chinh ultralytics bao — chi de tham khao, KHONG dung cho
        # bang benchmark (FR-10 do rieng, co warm-up va lap 3 phien)
        "speed_ms_ultralytics": {k: _to_float(v) for k, v in getattr(m, "speed", {}).items()},
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="ten run, se dung models/<name>/best.pt")
    g.add_argument("--weights", help="duong dan trong so bat ky (.pt/.onnx/.engine)")
    ap.add_argument("--config", default=None, help="config train, de lay imgsz cho khop")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="thap de mAP khong bi cat ngon")
    ap.add_argument("--iou", type=float, default=0.7, help="nguong IoU cua NMS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = repo_root()
    weights = Path(a.weights) if a.weights else root / "models" / a.name / "best.pt"
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        print(f"[loi] khong thay trong so: {weights}", file=sys.stderr)
        return 1

    data = Path(a.data)
    if not data.is_absolute():
        data = root / data
    if not data.exists():
        print(f"[loi] khong thay {data} — chay 'make data' truoc", file=sys.stderr)
        return 1

    imgsz = a.imgsz
    if imgsz is None and a.config:
        cfg_p = Path(a.config) if Path(a.config).is_absolute() else root / a.config
        imgsz = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))["train"]["imgsz"]
    if imgsz is None and a.name:
        mf = root / "results" / f"train_{a.name}_manifest.json"
        if mf.exists():
            imgsz = json.loads(mf.read_text(encoding="utf-8"))["config"]["train"]["imgsz"]
    imgsz = imgsz or 640

    print(f"[eval] trong so : {weights}")
    print(f"[eval] tap      : {a.split}  imgsz={imgsz}  conf={a.conf}  iou={a.iou}")

    res = evaluate(weights, data, a.split, imgsz, a.batch, a.device, a.conf, a.iou)

    tag = a.name or weights.stem
    out = Path(a.out) if a.out else root / "results" / (
        "train_eval.json" if a.name and weights.suffix == ".pt" else f"eval_{tag}_{res['format']}.json"
    )
    if not out.is_absolute():
        out = root / out

    # gom nhieu lan danh gia vao cung mot file thay vi ghi de
    payload = {}
    if out.exists():
        try:
            payload = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload[f"{tag}_{res['format']}"] = res
    write_json(out, payload)

    o = res["overall"]
    print(f"\n[eval] mAP@0.5      {o['mAP50']:.4f}")
    print(f"[eval] mAP@0.5:0.95 {o['mAP50_95']:.4f}")
    print(f"[eval] precision    {o['precision']:.4f}")
    print(f"[eval] recall       {o['recall']:.4f}")
    print(f"\n[eval] {'lop':<18}{'mAP50':>9}{'mAP50-95':>11}{'P':>9}{'R':>9}")
    for c, v in res["per_class"].items():
        print(f"       {c:<18}{v.get('mAP50', 0):>9.4f}{v.get('mAP50_95', 0):>11.4f}"
              f"{v.get('precision', 0):>9.4f}{v.get('recall', 0):>9.4f}")

    print(f"\n[eval] ghi -> {out.relative_to(root) if out.is_relative_to(root) else out}")

    if res["format"] == "pt" and o["mAP50"] < MAP50_TARGET:
        print(f"\n[canh bao] mAP@0.5 = {o['mAP50']:.4f} < nguong FR-05 la {MAP50_TARGET}.")
        print("           Xem mAP theo lop o tren: neu chi mot lop keo xuong thi do la")
        print("           dac tinh dataset (rui ro R4), khong phai loi pipeline.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
