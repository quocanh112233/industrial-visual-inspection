from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..data.common import rel_to_root, repo_root, write_json

MAP50_TARGET = 0.65


def _to_float(x) -> float:
    try:
        return round(float(x), 4)
    except Exception:
        return float("nan")


def evaluate(weights: Path, data: Path, split: str, imgsz: int, batch: int,
             device: str | None, conf: float, iou: float,
             rect: bool | None = None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kw = dict(data=str(data), split=split, imgsz=imgsz, batch=batch,
              conf=conf, iou=iou, plots=False, verbose=False)
    if rect is not None:
        kw["rect"] = rect
    if device is not None:
        kw["device"] = device
    m = model.val(**kw)

    box = m.box
    names = getattr(m, "names", None) or getattr(model, "names", {})
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

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
        "weights": rel_to_root(weights),
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
        "speed_ms_ultralytics": {k: _to_float(v) for k, v in getattr(m, "speed", {}).items()},
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="tên run, sẽ dùng models/<name>/best.pt")
    g.add_argument("--weights", help="đường dẫn trọng số bất kỳ (.pt/.onnx/.engine)")
    ap.add_argument("--config", default=None, help="config train, để lấy imgsz cho khớp")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="thấp để mAP không bị cắt ngọn")
    ap.add_argument("--iou", type=float, default=0.7, help="ngưỡng IoU của NMS")
    ap.add_argument("--rect", default=None, choices=["true", "false"],
                    help="ep chế độ rect. Bỏ trống = để ultralytics tự quyết "
                         "(True cho .pt, False cho định dạng xuất) — xem docs/input-framing.md")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = repo_root()
    weights = Path(a.weights) if a.weights else root / "models" / a.name / "best.pt"
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        print(f"[lỗi] không thấy trọng số: {weights}", file=sys.stderr)
        return 1

    data = Path(a.data)
    if not data.is_absolute():
        data = root / data
    if not data.exists():
        print(f"[lỗi] không thấy {data} — chạy 'make data' trước", file=sys.stderr)
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

    print(f"[eval] trọng số : {weights}")
    print(f"[eval] tập      : {a.split}  imgsz={imgsz}  conf={a.conf}  iou={a.iou}")

    rect = None if a.rect is None else (a.rect == "true")
    res = evaluate(weights, data, a.split, imgsz, a.batch, a.device, a.conf, a.iou, rect)

    tag = a.name or weights.stem
    out = Path(a.out) if a.out else root / "results" / (
        "train_eval.json" if a.name and weights.suffix == ".pt" else f"eval_{tag}_{res['format']}.json"
    )
    if not out.is_absolute():
        out = root / out

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
    print(f"\n[eval] {'lớp':<18}{'mAP50':>9}{'mAP50-95':>11}{'P':>9}{'R':>9}")
    for c, v in res["per_class"].items():
        print(f"       {c:<18}{v.get('mAP50', 0):>9.4f}{v.get('mAP50_95', 0):>11.4f}"
              f"{v.get('precision', 0):>9.4f}{v.get('recall', 0):>9.4f}")

    print(f"\n[eval] ghi -> {out.relative_to(root) if out.is_relative_to(root) else out}")

    if res["format"] == "pt" and o["mAP50"] < MAP50_TARGET:
        print(f"\n[cảnh báo] mAP@0.5 = {o['mAP50']:.4f} < ngưỡng FR-05 là {MAP50_TARGET}.")
        print("           Xem mAP theo lớp o trên: nếu chỉ một lớp kéo xuống thì đó là")
        print("           đặc tính dataset (rủi ro R4), không phải lỗi pipeline.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
