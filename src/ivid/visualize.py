"""Ve detection len anh test -> docs/images/sample_detections.png.

Bang so lieu tra loi duoc "nhanh bao nhieu, chinh xac bao nhieu", nhung khong
cho thay model thuc su nhin thay gi. Hinh nay lam viec do: moi lop mot anh, hop
du doan ve theo mau lop kem diem tin cay, hop that ve bang net trang mong de
doi chieu.

Anh duoc chon TAT DINH — voi moi lop, lay anh test dau tien (sap theo ten) chi
chua dung lop do. Nho vay chay lai tren may khac cho ra dung hinh ay, khong sinh
diff giay trong git.

    PYTHONPATH=src python -m ivid.visualize --backend tensorrt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Mau BGR cho 6 lop NEU-DET, chon de phan biet duoc ca khi in den trang
COLORS = [
    (60, 76, 231),    # do
    (80, 175, 76),    # xanh la
    (243, 150, 33),   # xanh duong
    (39, 174, 245),   # cam
    (156, 39, 176),   # tim
    (22, 190, 207),   # vang
]
WHITE = (255, 255, 255)


def draw_detections(img_bgr: np.ndarray, det: np.ndarray, names: list[str],
                    gt: np.ndarray | None = None, conf_thres: float = 0.25,
                    scale: float = 1.0) -> np.ndarray:
    """Ve hop du doan (mau theo lop) va hop that (net trang mong) len mot ban sao."""
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
        # nhan nam tren hop, nhung neu hop sat mep tren thi lat xuong duoi
        ty = p1[1] - 4 if p1[1] - th - 6 >= 0 else p1[1] + th + 6
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty + 2), color, -1)
        cv2.putText(out, nhan, (p1[0] + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def add_caption(tile: np.ndarray, text: str, height: int = 30) -> np.ndarray:
    """Dan mot dai chu den phia tren o anh."""
    import cv2

    w = tile.shape[1]
    strip = np.zeros((height, w, 3), dtype=np.uint8)
    cv2.putText(strip, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, WHITE, 1, cv2.LINE_AA)
    return np.vstack([strip, tile])


def build_grid(tiles: list[np.ndarray], cols: int = 3, gap: int = 6) -> np.ndarray:
    """Ghep cac o cung kich thuoc thanh luoi, chen khe mau xam."""
    if not tiles:
        raise ValueError("khong co o nao de ghep")
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
    """Voi moi lop, lay anh dau tien CHI chua lop do; khong co thi lay anh dau
    tien co chua lop do. Duyet theo thu tu ten file nen ket qua tat dinh."""
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
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--tile", type=int, default=320, help="canh moi o anh, pixel")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="docs/images/sample_detections.png")
    a = ap.parse_args()

    root = repo_root()
    names = class_names(load_config(root / "configs/data.yaml"))
    nc = len(names)

    images, label_dir = test_set(root / a.data, "test")
    gts = load_ground_truth(images, label_dir)
    idx = chon_anh_moi_lop(images, gts, nc)
    if not idx:
        print("[loi] khong chon duoc anh nao", file=sys.stderr)
        return 1

    w = weights_for(root / "models" / a.model, a.backend)
    if not w.exists():
        print(f"[loi] khong thay {w}", file=sys.stderr)
        return 1
    runner = build_runner(a.backend, w, imgsz=a.imgsz, conf=a.conf, iou=a.iou, nc=nc)

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
        tiles.append(add_caption(ve, f"{lop_that}  —  {n_hop} hop / {len(gts[i])} that"))
    runner.close()

    grid = build_grid(tiles, cols=a.cols)
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"[demo] {len(tiles)} anh, backend {a.backend}, conf {a.conf}")
    print("[demo] hop trang = ground truth, hop mau = du doan")
    print(f"[demo] ghi -> {rel_to_root(out)}  ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
