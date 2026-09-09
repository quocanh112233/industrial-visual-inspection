"""Tien xu ly anh dung CHUNG cho moi runtime.

Ba runtime phai nhan dung mot mang so giong het nhau. Neu moi runtime tu resize
theo cach rieng, chenh lech mAP do duoc se lan lon giua "khac biet runtime" va
"khac biet tien xu ly" — va bang benchmark mat y nghia. Vi vay tat ca deu goi
ham o day.
"""
from __future__ import annotations

import numpy as np


def letterbox(img: np.ndarray, new_shape: int = 640,
              color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize giu ti le roi dem vien. Tra ve (anh, ti_le, (pad_x, pad_y)).

    NEU-DET la anh vuong 200x200 nen thuc te khong phai dem gi, nhung ham van
    xu ly truong hop tong quat de con dung lai cho anh camera that sau nay.
    """
    import cv2

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_y, pad_x = (new_shape - nh) / 2, (new_shape - nw) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img_bgr: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR uint8 HWC  ->  RGB float32 NCHW da chuan hoa 0..1, batch 1."""
    lb, r, pad = letterbox(img_bgr, imgsz)
    x = lb[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], r, pad


def read_image(path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"khong doc duoc anh: {path}")
    return img
