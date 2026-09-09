"""Tinh mAP theo kieu COCO — thuan NumPy, dung chung cho ca ba dinh dang.

VI SAO PHAI TU VIET thay vi goi ultralytics.val():

Ban dau FR-13 duoc do bang `ultralytics.val()` cho ca ba dinh dang, voi ly do
"cung mot ham cham diem thi khong the lech vi cach cham". Do la sai lam: goi
CUNG MOT HAM khong co nghia la dung CUNG MOT PHEP DO — ultralytics doi pipeline
danh gia ben duoi tuy theo dinh dang model.

Bang chung: tren cung 50 anh o conf=0.001, PyTorch sinh 8624 detection va ONNX
sinh 8629, khop nhau 98.9% voi IoU trung binh 0.993 (results/parity_conf001.json).
Hai model gan nhu dong nhat. Nhung ultralytics.val() bao ONNX mat 0.031 mAP@0.5
va 0.083 mAP@0.5:0.95. Co che gay ra chenh lech ay sau do da truy ra duoc
(scripts/diag_rect.py): ultralytics dat rect=True cho .pt nhung ep rect=False
cho dinh dang xuat, va o che do rect thi khung anh la 672x672 (anh 640 co vien
xam 16 px) chu khong phai 640x640. Cung mot ban .pt cham o rect=False cho
0.7310 — dung bang con so ma module nay tinh ra. Tuc khong phai model kem di,
ma la hai che do danh gia khac nhau bi dem ra so voi nhau.

Module nay nhan detection tu chinh runner cua du an (dung chung ivid.preprocess
va ivid.postprocess, da duoc kiem chung la tuong duong) roi tinh mAP. Nho vay
chenh lech giua ba dinh dang chi con den tu ban than runtime, va ca ba deu duoc
cham o dung che do ma engine trien khai thuc su chay: 640x640, khong vien.

Phep tinh trong file nay da duoc kiem chung doc lap bang scripts/diag_map.py:
cham CUNG MOT tap detection hai lan — mot lan bang ham o day, mot lan bang chinh
ma cua ultralytics (BaseValidator.match_predictions + utils.metrics.ap_per_class)
— cho ket qua lech 0.0004 mAP@0.5.

Thuat toan theo COCO: ghep detection voi ground truth theo tung nguong IoU
0.50:0.05:0.95, tinh duong precision-recall, lay AP bang noi suy 101 diem.
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giua moi cap (a_i, b_j). a: (N,4), b: (M,4), dang xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP bang noi suy 101 diem (chuan COCO).

    Truoc khi noi suy, precision duoc lam don dieu giam tu phai sang trai: tai
    moi muc recall ta lay precision TOT NHAT dat duoc o muc recall do tro len.
    Khong lam buoc nay thi duong PR rang cua se keo AP xuong mot cach nhan tao.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x)) if hasattr(np, "trapezoid") \
        else float(np.trapz(np.interp(x, mrec, mpre), x))


def compute_map(detections: list[np.ndarray], ground_truth: list[np.ndarray],
                nc: int, iou_thresholds: np.ndarray = IOU_THRESHOLDS) -> dict:
    """mAP cho mot tap anh.

    detections[i]:   (Ni, 6) = x1 y1 x2 y2 conf class   — toa do anh GOC
    ground_truth[i]: (Mi, 5) = x1 y1 x2 y2 class        — toa do anh GOC
    """
    assert len(detections) == len(ground_truth), "so anh khong khop"
    n_thr = len(iou_thresholds)

    # gom theo lop: moi phan tu la (conf, tp_theo_tung_nguong)
    stats: dict[int, list] = {c: [] for c in range(nc)}
    n_gt: dict[int, int] = {c: 0 for c in range(nc)}

    for det, gt in zip(detections, ground_truth, strict=True):
        for c in range(nc):
            gt_c = gt[gt[:, 4] == c][:, :4] if len(gt) else np.zeros((0, 4), np.float32)
            n_gt[c] += len(gt_c)
            det_c = det[det[:, 5] == c] if len(det) else np.zeros((0, 6), np.float32)
            if len(det_c) == 0:
                continue
            order = np.argsort(-det_c[:, 4])          # diem cao truoc
            det_c = det_c[order]
            tp = np.zeros((len(det_c), n_thr), dtype=bool)
            if len(gt_c):
                ious = box_iou_matrix(det_c[:, :4], gt_c)
                for ti, thr in enumerate(iou_thresholds):
                    matched = np.zeros(len(gt_c), dtype=bool)
                    for di in range(len(det_c)):
                        # ghep tham lam: detection diem cao chon GT tot nhat con trong
                        cand = np.where((ious[di] >= thr) & (~matched))[0]
                        if len(cand):
                            best = cand[np.argmax(ious[di, cand])]
                            matched[best] = True
                            tp[di, ti] = True
            for di in range(len(det_c)):
                stats[c].append((float(det_c[di, 4]), tp[di].copy()))

    ap = np.zeros((nc, n_thr), dtype=np.float64)
    p_at_maxf1 = np.zeros(nc)
    r_at_maxf1 = np.zeros(nc)
    present = []

    for c in range(nc):
        if n_gt[c] == 0:
            continue
        present.append(c)
        if not stats[c]:
            continue
        confs = np.array([s[0] for s in stats[c]])
        tps = np.stack([s[1] for s in stats[c]])       # (N, n_thr)
        order = np.argsort(-confs)
        tps = tps[order]

        for ti in range(n_thr):
            tp_cum = np.cumsum(tps[:, ti])
            fp_cum = np.cumsum(~tps[:, ti])
            recall = tp_cum / n_gt[c]
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap[c, ti] = average_precision(recall, precision)
            if ti == 0:                                 # tai IoU 0.5, lay P/R o diem F1 lon nhat
                f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
                k = int(np.argmax(f1))
                p_at_maxf1[c], r_at_maxf1[c] = precision[k], recall[k]

    idx = np.array(present, dtype=int)
    return {
        "mAP50": round(float(ap[idx, 0].mean()), 4) if len(idx) else 0.0,
        "mAP50_95": round(float(ap[idx].mean()), 4) if len(idx) else 0.0,
        "precision": round(float(p_at_maxf1[idx].mean()), 4) if len(idx) else 0.0,
        "recall": round(float(r_at_maxf1[idx].mean()), 4) if len(idx) else 0.0,
        "per_class": {int(c): {"mAP50": round(float(ap[c, 0]), 4),
                               "mAP50_95": round(float(ap[c].mean()), 4),
                               "precision": round(float(p_at_maxf1[c]), 4),
                               "recall": round(float(r_at_maxf1[c]), 4),
                               "n_gt": int(n_gt[c])} for c in present},
        "n_images": len(detections),
        "n_detections": int(sum(len(d) for d in detections)),
        "n_ground_truth": int(sum(n_gt.values())),
        "iou_thresholds": [round(float(t), 2) for t in iou_thresholds],
    }


def load_ground_truth(image_paths: list, label_dir) -> list[np.ndarray]:
    """Doc nhan YOLO (cx cy w h chuan hoa) -> (M,5) xyxy pixel anh goc + class."""
    from pathlib import Path

    import cv2

    out = []
    label_dir = Path(label_dir)
    for p in image_paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        lp = label_dir / f"{Path(p).stem}.txt"
        rows = []
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                f = line.split()
                if len(f) != 5:
                    continue
                c = int(f[0])
                cx, cy, bw, bh = (float(v) for v in f[1:])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h, c])
        out.append(np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32))
    return out
