from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

COLORS = [
    (60, 76, 231),
    (80, 175, 76),
    (243, 150, 33),
    (39, 174, 245),
    (156, 39, 176),
    (22, 190, 207),
]
WHITE = (255, 255, 255)


def draw_detections(img_bgr: np.ndarray, det: np.ndarray, names: list[str],
                    gt: np.ndarray | None = None, conf_thres: float = 0.25,
                    scale: float = 1.0) -> np.ndarray:
    import cv2

    out = img_bgr.copy()
    thick = max(1, int(round(scale)))

    if gt is not None:
        for x1, y1, x2, y2, _c in gt:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), WHITE, thick)

    for x1, y1, x2, y2, conf, c in det:
        if conf < conf_thres:
            continue
        ci = int(c)
        color = COLORS[ci % len(COLORS)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, thick + 1)

        nhan = f"{names[ci] if ci < len(names) else ci} {conf:.2f}"
        fs = 0.35 * scale
        (tw, th), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        ty = p1[1] - 4 if p1[1] - th - 6 >= 0 else p1[1] + th + 6
        tx = min(p1[0], out.shape[1] - tw - 6)
        tx = max(tx, 0)
        cv2.rectangle(out, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
        cv2.putText(out, nhan, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def nhan_o(lop: str, n_hop: int, n_that: int) -> str:
    return f"{lop}  |  {n_hop} du doan / {n_that} that"


def co_chu_vua(text: str, rong: int, lon_nhat: float = 0.45,
               nho_nhat: float = 0.28, le: int = 16) -> float:
    import cv2

    co = lon_nhat
    while co > nho_nhat:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, co, 1)
        if tw <= rong - le:
            return round(co, 3)
        co -= 0.01
    return round(nho_nhat, 3)


def add_caption(tile: np.ndarray, text: str, height: int = 30) -> np.ndarray:
    import cv2

    w = tile.shape[1]
    strip = np.zeros((height, w, 3), dtype=np.uint8)
    cv2.putText(strip, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                co_chu_vua(text, w), WHITE, 1, cv2.LINE_AA)
    return np.vstack([strip, tile])


def build_grid(tiles: list[np.ndarray], cols: int = 3, gap: int = 6) -> np.ndarray:
    if not tiles:
        raise ValueError("không có o nao để ghep")
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * h + (rows - 1) * gap,
                      cols * w + (cols - 1) * gap, 3), 40, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (h + gap), c * (w + gap)
        canvas[y:y + h, x:x + w] = t
    return canvas


def chon_anh_moi_lop(images: list[Path], gts: list[np.ndarray], nc: int) -> list[int]:
    chon: list[int] = []
    for c in range(nc):
        du_phong = None
        for i, g in enumerate(gts):
            if len(g) == 0 or i in chon:
                continue
            lop = set(g[:, 4].astype(int).tolist())
            if lop == {c}:
                chon.append(i)
                break
            if c in lop and du_phong is None:
                du_phong = i
        else:
            if du_phong is not None:
                chon.append(du_phong)
    return chon


def main() -> int:
    import cv2

    from .benchmark.accuracy import test_set
    from .benchmark.metrics import load_ground_truth
    from .benchmark.runners.base import build_runner, weights_for
    from .data.common import class_names, load_config, rel_to_root, repo_root
    from .preprocess import read_image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--backend", default="tensorrt", choices=["pytorch", "onnx", "tensorrt"])
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--config", default="configs/benchmark.yaml",
                    help="lấy imgsz và resize_to từ đây, để hình minh hoạ đúng chế độ "
                         "khung ảnh với số liệu trong báo cáo")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--resize-to", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--tile", type=int, default=320, help="canh mọi o ảnh, pixel")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="docs/images/sample_detections.png")
    a = ap.parse_args()

    import yaml

    root = repo_root()
    names = class_names(load_config(root / "configs/data.yaml"))
    nc = len(names)

    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    imgsz = int(a.imgsz or cfg["imgsz"])
    resize_to = a.resize_to if a.resize_to is not None else (
        int(cfg["resize_to"]) if cfg.get("resize_to") else None)

    images, label_dir = test_set(root / a.data, "test")
    gts = load_ground_truth(images, label_dir)
    idx = chon_anh_moi_lop(images, gts, nc)
    if not idx:
        print("[lỗi] không chon được ảnh nao", file=sys.stderr)
        return 1

    w = weights_for(root / "models" / a.model, a.backend)
    if not w.exists():
        print(f"[lỗi] không thấy {w}", file=sys.stderr)
        return 1
    runner = build_runner(a.backend, w, imgsz=imgsz, conf=a.conf, iou=a.iou, nc=nc,
                          resize_to=resize_to)

    tiles = []
    for i in idx:
        img = read_image(images[i])
        det = runner.run_array(img).detections
        r = a.tile / max(img.shape[:2])
        big = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                         interpolation=cv2.INTER_CUBIC)
        det_s = det.copy()
        det_s[:, :4] *= r
        gt_s = gts[i].copy()
        if len(gt_s):
            gt_s[:, :4] *= r
        ve = draw_detections(big, det_s, names, gt_s, conf_thres=a.conf, scale=r)
        n_hop = int((det[:, 4] >= a.conf).sum()) if len(det) else 0
        lop_that = names[int(gts[i][0, 4])] if len(gts[i]) else "?"
        tiles.append(add_caption(ve, nhan_o(lop_that, n_hop, len(gts[i]))))
    runner.close()

    grid = build_grid(tiles, cols=a.cols)
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    khung = f"{imgsz}" + (f"/{resize_to}" if resize_to else "")
    print(f"[demo] {len(tiles)} ảnh, backend {a.backend}, conf {a.conf}, khung {khung}")
    print("[demo] hộp trang = ground truth, hộp mau = du doan")
    print(f"[demo] ghi -> {rel_to_root(out)}  ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
