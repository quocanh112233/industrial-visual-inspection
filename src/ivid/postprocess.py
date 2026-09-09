from __future__ import annotations

import numpy as np


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    half_w, half_h = x[:, 2] / 2, x[:, 3] / 2
    y[:, 0] = x[:, 0] - half_w
    y[:, 1] = x[:, 1] - half_h
    y[:, 2] = x[:, 0] + half_w
    y[:, 3] = x[:, 1] + half_h
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thres]
    return keep


MAX_NMS = 30000


def decode(raw: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.7,
           max_det: int = 300, nc: int | None = None,
           multi_label: bool = False) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[0]

    if nc is not None:
        if raw.shape[0] == 4 + nc:
            raw = raw.T
        elif raw.shape[1] != 4 + nc:
            raise ValueError(f"tensor {raw.shape} không khớp nc={nc} (mong đợi một trục = {4 + nc})")
    elif raw.shape[0] < raw.shape[1]:
        raw = raw.T

    boxes_xywh, scores_all = raw[:, :4], raw[:, 4:]

    if multi_label:
        rows, cols = np.nonzero(scores_all >= conf_thres)
        if len(rows) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = xywh2xyxy(boxes_xywh[rows].astype(np.float32))
        confs, class_ids = scores_all[rows, cols], cols
    else:
        class_ids = scores_all.argmax(axis=1)
        confs = scores_all[np.arange(len(scores_all)), class_ids]
        m = confs >= conf_thres
        if not m.any():
            return np.zeros((0, 6), dtype=np.float32)
        boxes = xywh2xyxy(boxes_xywh[m].astype(np.float32))
        confs, class_ids = confs[m], class_ids[m]

    if len(confs) > MAX_NMS:
        top = np.argsort(-confs)[:MAX_NMS]
        boxes, confs, class_ids = boxes[top], confs[top], class_ids[top]

    keep_all: list[int] = []
    for c in np.unique(class_ids):
        idx = np.nonzero(class_ids == c)[0]
        kept = nms(boxes[idx], confs[idx], iou_thres)
        keep_all.extend(idx[k] for k in kept)

    if not keep_all:
        return np.zeros((0, 6), dtype=np.float32)
    keep_all = np.array(keep_all)
    keep_all = keep_all[confs[keep_all].argsort()[::-1]][:max_det]

    out = np.empty((len(keep_all), 6), dtype=np.float32)
    out[:, :4] = boxes[keep_all]
    out[:, 4] = confs[keep_all]
    out[:, 5] = class_ids[keep_all]
    return out


def scale_boxes(det: np.ndarray, ratio: float, pad: tuple[int, int],
                orig_hw: tuple[int, int]) -> np.ndarray:
    if len(det) == 0:
        return det
    out = det.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / ratio
    h, w = orig_hw
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def match_detections(a: np.ndarray, b: np.ndarray, iou_thres: float = 0.5) -> dict:
    if len(a) == 0 and len(b) == 0:
        return {"matched": 0, "only_a": 0, "only_b": 0, "mean_iou": 1.0, "max_conf_diff": 0.0}
    if len(a) == 0 or len(b) == 0:
        return {"matched": 0, "only_a": len(a), "only_b": len(b),
                "mean_iou": 0.0, "max_conf_diff": 0.0}

    used_b: set[int] = set()
    ious, conf_diffs, matched = [], [], 0
    for i in range(len(a)):
        best_j, best_iou = -1, 0.0
        for j in range(len(b)):
            if j in used_b or a[i, 5] != b[j, 5]:
                continue
            xx1 = max(a[i, 0], b[j, 0])
            yy1 = max(a[i, 1], b[j, 1])
            xx2 = min(a[i, 2], b[j, 2])
            yy2 = min(a[i, 3], b[j, 3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            ua = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
            ub = (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1])
            iou = inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thres:
            used_b.add(best_j)
            matched += 1
            ious.append(best_iou)
            conf_diffs.append(abs(float(a[i, 4] - b[best_j, 4])))

    return {
        "matched": matched,
        "only_a": len(a) - matched,
        "only_b": len(b) - matched,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "max_conf_diff": round(max(conf_diffs), 4) if conf_diffs else 0.0,
    }
