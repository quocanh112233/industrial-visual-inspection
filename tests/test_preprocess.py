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
import pytest

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


# --------------------------------- resize_to: tach kich thuoc anh khoi kich thuoc khung
def test_resize_to_tao_vien_xam_quanh_anh_da_thu_nho():
    """imgsz 672 + resize_to 640: anh 200x200 duoc phong len 640 roi dem 16 px
    moi ben. Day la che do rect cua ultralytics, cho mAP cao hon han."""
    lb, r, pad = letterbox(anh(200, 200), 672, resize_to=640)
    assert lb.shape[:2] == (672, 672)
    assert r == 640 / 200
    assert pad == (16, 16)
    assert lb[0, 0].tolist() == [114, 114, 114]        # goc la vien
    assert lb[336, 336].tolist() == [128, 128, 128]    # giua la anh


def test_khong_truyen_resize_to_thi_anh_phu_kin_khung():
    a = letterbox(anh(200, 200), 640)
    b = letterbox(anh(200, 200), 640, resize_to=640)
    assert a[1] == b[1] and a[2] == b[2]


def test_resize_to_lon_hon_khung_thi_bao_loi():
    """Chan cau hinh vo nghia ngay tai cho, thay vi de anh bi cat am tham."""
    with pytest.raises(ValueError, match="lon hon khung"):
        letterbox(anh(200, 200), 640, resize_to=672)


def test_scale_boxes_van_dung_khi_co_ca_resize_to_va_vien():
    _, r, pad = letterbox(anh(200, 200), 672, resize_to=640)
    det = np.array([[16.0 + 32, 16.0 + 64, 16.0 + 160, 16.0 + 192, 0.9, 0.0]], dtype=np.float32)
    out = scale_boxes(det, r, pad, (200, 200))
    assert out[0, :4].tolist() == [10.0, 20.0, 50.0, 60.0]


def test_preprocess_tra_ve_khung_672_khi_yeu_cau():
    x, _, _ = preprocess(anh(200, 200), 672, resize_to=640)
    assert x.shape == (1, 3, 672, 672)
