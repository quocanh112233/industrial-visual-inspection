from __future__ import annotations

import numpy as np


def letterbox(img: np.ndarray, new_shape: int = 640,
              color: tuple[int, int, int] = (114, 114, 114),
              scaleup: bool = True,
              resize_to: int | None = None) -> tuple[np.ndarray, float, tuple[int, int]]:
    import cv2

    h, w = img.shape[:2]
    dich = resize_to or new_shape
    if dich > new_shape:
        raise ValueError(f"resize_to={dich} lớn hơn khung new_shape={new_shape}")
    r = min(dich / h, dich / w)
    if not scaleup:
        r = min(r, 1.0)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_y, pad_x = (new_shape - nh) / 2, (new_shape - nw) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img_bgr: np.ndarray, imgsz: int = 640, scaleup: bool = True,
               resize_to: int | None = None) -> tuple[np.ndarray, float, tuple[int, int]]:
    lb, r, pad = letterbox(img_bgr, imgsz, scaleup=scaleup, resize_to=resize_to)
    x = lb[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], r, pad


def read_image(path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"không doc được ảnh: {path}")
    return img
