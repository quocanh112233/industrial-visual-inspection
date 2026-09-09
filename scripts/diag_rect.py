"""Kiem chung gia thuyet: ultralytics cham ban .pt o 672x672, con moi thu khac o 640x640.

BOI CANH. scripts/diag_map.py da chung minh hai dieu:
  * phep tinh mAP cua du an dung (lech 0.0004 so voi ma ultralytics tren cung
    mot tap detection),
  * va che do tien xu ly hien tai (phong anh 200 -> 640) la dung, vi khong
    phong to thi mAP sap tu 0.73 xuong 0.51.

Con lai chenh lech 0.030 giua ban .pt (0.7621) va moi thu khac (~0.7315).
Nghi van: ultralytics dat rect=True cho .pt nhung ep rect=False cho cac dinh
dang xuat (engine/validator.py, "if not (pt or dynamic): self.args.rect = False").
O che do rect voi pad=0.5, khung anh khong phai 640 ma la

    ceil(640/32 + 0.5) * 32 = 21 * 32 = 672

nghia la ban .pt duoc cham voi anh 640 nam giua mot khung 672 co vien xam 16px,
trong khi ONNX/TensorRT — von co dau vao co dinh 640x640 — khong the lam vay.

Script chay ultralytics.val() ba lan tren CUNG ban .pt, chi doi mot tham so,
va in kich thuoc tensor that su di vao model (bat bang forward hook).

    PYTHONPATH=src PYTHONUTF8=1 python3 scripts/diag_rect.py
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.data.common import repo_root  # noqa: E402


def chay(weights: Path, data: Path, imgsz: int, rect: bool, conf: float,
         iou: float, batch: int) -> tuple[float, float, str]:
    """Mot lan val(). Tra ve (mAP50, mAP50_95, kich thuoc dau vao that)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    thay: Counter[str] = Counter()

    def hook(_mod, inp):
        # KHONG lay lan forward dau tien: truoc vong val, ultralytics con chay
        # mot luot dung stride bang tensor gia rat nho (32x32). Dem tan suat roi
        # lay hinh dang pho bien nhat moi ra dung anh that.
        if inp and hasattr(inp[0], "shape") and len(inp[0].shape) == 4:
            thay["x".join(str(int(v)) for v in inp[0].shape)] += 1

    h = None
    with contextlib.suppress(Exception):
        h = model.model.register_forward_pre_hook(hook)

    m = model.val(data=str(data), split="test", imgsz=imgsz, batch=batch,
                  rect=rect, conf=conf, iou=iou, plots=False, verbose=False)
    if h is not None:
        h.remove()
    pho_bien = thay.most_common(1)[0][0] if thay else "?"
    return float(m.box.map50), float(m.box.map), pho_bien


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--data", default="data/processed/data.yaml")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()

    root = repo_root()
    w = root / "models" / a.model / "best.pt"
    if not w.exists():
        print(f"[loi] khong thay {w}", file=sys.stderr)
        return 1
    data = root / a.data

    print(f"[rect] {a.model}/best.pt  conf={a.conf} iou={a.iou} batch={a.batch}")
    print("[rect] moc: duong ong cua du an (640x640) = 0.7317\n")

    thu_nghiem = [
        ("rect=True  imgsz=640  (mac dinh cua ultralytics cho .pt)", 640, True),
        ("rect=False imgsz=640  (ultralytics ep cho ONNX/TensorRT)", 640, False),
        ("rect=False imgsz=672  (do phan giai cao hon, khong vien)", 672, False),
    ]
    for nhan, imgsz, rect in thu_nghiem:
        m50, m95, shape = chay(w, data, imgsz, rect, a.conf, a.iou, a.batch)
        print(f"    {nhan}")
        print(f"        -> mAP@0.5 {m50:.4f}   mAP@0.5:0.95 {m95:.4f}   "
              f"tensor vao model: {shape}\n")

    print("[rect] Doc ket qua:")
    print("    Neu dong 2 ra ~0.731 thi toan bo chenh lech la do vien xam cua che do")
    print("    rect, va con so 0.7621 trong log huan luyen KHONG phai con so ma")
    print("    engine 640x640 tren day chuyen dat duoc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
