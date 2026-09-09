"""Tien xu ly anh dung CHUNG cho moi runtime.

Ba runtime phai nhan dung mot mang so giong het nhau. Neu moi runtime tu resize
theo cach rieng, chenh lech mAP do duoc se lan lon giua "khac biet runtime" va
"khac biet tien xu ly" — va bang benchmark mat y nghia. Vi vay tat ca deu goi
ham o day.
"""
from __future__ import annotations

import numpy as np


def letterbox(img: np.ndarray, new_shape: int = 640,
              color: tuple[int, int, int] = (114, 114, 114),
              scaleup: bool = True,
              resize_to: int | None = None) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize giu ti le roi dem vien. Tra ve (anh, ti_le, (pad_x, pad_y)).

    `scaleup=False` thi anh NHO HON khung KHONG duoc phong to, chi duoc dem xam
    cho du 640x640. Nghe nhu chi tiet vun, nhung voi NEU-DET (anh 200x200) day
    la khac biet lon: scaleup=True phong anh len 3.2 lan, tuc dua cho model mot
    thang do no chua tung thay luc huan luyen. Ultralytics dat scaleup=False cho
    duong val (ultralytics/data/dataset.py, build_transforms) — nen so mAP no
    bao la do o che do khong phong to.

    `resize_to` tach KICH THUOC THU NHO khoi KICH THUOC KHUNG. Mac dinh hai cai
    bang nhau (anh phu kin khung). Dat resize_to=640 voi new_shape=672 thi anh
    duoc thu ve 640 roi dem xam ra 672 — tuc anh 640 nam giua mot vien xam 16 px.
    Do dung la che do rect ma ultralytics dung khi val ban .pt, va no cho mAP cao
    hon han (xem scripts/diag_rect.py).
    """
    import cv2

    h, w = img.shape[:2]
    dich = resize_to or new_shape
    if dich > new_shape:
        raise ValueError(f"resize_to={dich} lon hon khung new_shape={new_shape}")
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
    """BGR uint8 HWC  ->  RGB float32 NCHW da chuan hoa 0..1, batch 1.

    `imgsz` la kich thuoc DAU VAO MODEL (khung), `resize_to` la kich thuoc anh
    duoc thu ve ben trong khung do. Xem letterbox().
    """
    lb, r, pad = letterbox(img_bgr, imgsz, scaleup=scaleup, resize_to=resize_to)
    x = lb[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], r, pad


def read_image(path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"khong doc duoc anh: {path}")
    return img
