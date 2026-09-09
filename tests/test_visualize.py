"""Test phan ve hinh minh hoa (ivid.visualize).

Khong can model: cac ham o day thuan tuy nhan mang detection va tra ve anh.
Diem can canh la tinh TAT DINH cua viec chon anh — neu no doi theo may thi moi
lan chay lai se sinh mot file PNG khac va git bao thay doi vo co.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.visualize import add_caption, build_grid, chon_anh_moi_lop, draw_detections  # noqa: E402

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def anh(h=200, w=200):
    return np.full((h, w, 3), 90, dtype=np.uint8)


def test_ve_khong_sua_anh_goc():
    goc = anh()
    truoc = goc.copy()
    draw_detections(goc, np.array([[10, 10, 60, 60, 0.9, 1]], np.float32), NAMES)
    assert np.array_equal(goc, truoc), "ham phai ve len ban sao"


def test_ve_thay_doi_diem_anh_o_vien_hop():
    out = draw_detections(anh(), np.array([[10, 10, 60, 60, 0.9, 1]], np.float32), NAMES)
    assert not np.array_equal(out, anh())


def test_bo_qua_detection_duoi_nguong():
    det = np.array([[10, 10, 60, 60, 0.10, 1]], np.float32)
    assert np.array_equal(draw_detections(anh(), det, NAMES, conf_thres=0.25), anh())


def test_nhan_lat_xuong_duoi_khi_hop_sat_mep_tren():
    """Hop o y=0 thi nhan phai nam trong anh, khong bi cat mat."""
    det = np.array([[10, 0, 60, 40, 0.9, 1]], np.float32)
    out = draw_detections(anh(), det, NAMES)
    assert out.shape == (200, 200, 3)          # khong tran ra ngoai
    assert not np.array_equal(out, anh())


def test_caption_lam_anh_cao_them_dung_bang_dai_chu():
    assert add_caption(anh(200, 200), "crazing", height=30).shape == (230, 200, 3)


def test_luoi_ghep_dung_so_hang_cot():
    tiles = [anh(100, 100) for _ in range(6)]
    g = build_grid(tiles, cols=3, gap=6)
    assert g.shape == (2 * 100 + 6, 3 * 100 + 2 * 6, 3)


def test_luoi_hang_cuoi_thieu_o_van_ghep_duoc():
    g = build_grid([anh(50, 50) for _ in range(4)], cols=3, gap=0)
    assert g.shape == (2 * 50, 3 * 50, 3)


def test_chon_anh_uu_tien_anh_chi_co_mot_lop():
    gts = [
        np.array([[0, 0, 10, 10, 0], [0, 0, 10, 10, 1]], np.float32),  # tron 2 lop
        np.array([[0, 0, 10, 10, 0]], np.float32),                     # chi lop 0
        np.array([[0, 0, 10, 10, 1]], np.float32),                     # chi lop 1
    ]
    imgs = [Path(f"{i}.jpg") for i in range(3)]
    assert chon_anh_moi_lop(imgs, gts, nc=2) == [1, 2]


def test_chon_anh_lay_du_phong_khi_khong_co_anh_thuan_mot_lop():
    gts = [np.array([[0, 0, 10, 10, 0], [0, 0, 10, 10, 1]], np.float32)]
    assert chon_anh_moi_lop([Path("a.jpg")], gts, nc=2) == [0]


def test_chon_anh_khong_lay_trung_mot_anh_hai_lan():
    gts = [np.array([[0, 0, 10, 10, 0], [0, 0, 10, 10, 1]], np.float32)]
    chon = chon_anh_moi_lop([Path("a.jpg")], gts, nc=2)
    assert len(chon) == len(set(chon))


def test_chon_anh_bo_qua_anh_khong_co_nhan():
    gts = [np.zeros((0, 5), np.float32), np.array([[0, 0, 10, 10, 0]], np.float32)]
    assert chon_anh_moi_lop([Path("a.jpg"), Path("b.jpg")], gts, nc=1) == [1]
