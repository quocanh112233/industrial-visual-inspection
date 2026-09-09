"""Test tien xu ly dung chung (ivid.preprocess).

Trong tam la co `scaleup`. NEU-DET la anh 200x200 con imgsz la 640, nen day
khong phai chi tiet vun: scaleup=True phong anh len 3.2 lan truoc khi dua vao
model, con scaleup=False giu nguyen 200x200 roi dem xam. Ultralytics dung
scaleup=False o duong val — chinh khac biet nay lam mAP do duoc lech.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.postprocess import scale_boxes  # noqa: E402
from ivid.preprocess import letterbox, preprocess  # noqa: E402


def anh(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_scaleup_true_phong_anh_nho_len_vua_khung():
    lb, r, pad = letterbox(anh(200, 200), 640, scaleup=True)
    assert lb.shape[:2] == (640, 640)
    assert r == 640 / 200
    assert pad == (0, 0)


def test_scaleup_false_giu_nguyen_kich_thuoc_va_dem_xam():
    lb, r, pad = letterbox(anh(200, 200), 640, scaleup=False)
    assert lb.shape[:2] == (640, 640)          # van du kich thuoc model yeu cau
    assert r == 1.0                            # nhung khong phong to
    assert pad == (220, 220)                   # (640 - 200) / 2 moi ben
    assert lb[0, 0].tolist() == [114, 114, 114]        # goc la vien xam
    assert lb[320, 320].tolist() == [128, 128, 128]    # giua la anh that


def test_scaleup_khong_anh_huong_anh_lon_hon_khung():
    """Anh lon hon 640 van bi thu nho trong ca hai che do — scaleup chi chan
    viec PHONG TO, khong chan viec thu nho."""
    for su in (True, False):
        _, r, _ = letterbox(anh(1000, 1000), 640, scaleup=su)
        assert r == 0.64


def test_scale_boxes_dua_dung_ve_toa_do_goc_o_che_do_khong_phong_to():
    """Vong kin: box o he letterbox -> he anh goc, khi r=1.0 va pad=220."""
    _, r, pad = letterbox(anh(200, 200), 640, scaleup=False)
    det = np.array([[230.0, 240.0, 270.0, 280.0, 0.9, 0.0]], dtype=np.float32)
    out = scale_boxes(det, r, pad, (200, 200))
    assert out[0, :4].tolist() == [10.0, 20.0, 50.0, 60.0]


def test_preprocess_tra_ve_tensor_dung_dang_trong_ca_hai_che_do():
    for su in (True, False):
        x, _, _ = preprocess(anh(200, 200), 640, scaleup=su)
        assert x.shape == (1, 3, 640, 640)
        assert x.dtype == np.float32
        assert x.min() >= 0.0 and x.max() <= 1.0
