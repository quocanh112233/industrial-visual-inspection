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
    assert np.array_equal(goc, truoc), "hàm phải vẽ lên bản sao"


def test_ve_thay_doi_diem_anh_o_vien_hop():
    out = draw_detections(anh(), np.array([[10, 10, 60, 60, 0.9, 1]], np.float32), NAMES)
    assert not np.array_equal(out, anh())


def test_bo_qua_detection_duoi_nguong():
    det = np.array([[10, 10, 60, 60, 0.10, 1]], np.float32)
    assert np.array_equal(draw_detections(anh(), det, NAMES, conf_thres=0.25), anh())


def test_nhan_lat_xuong_duoi_khi_hop_sat_mep_tren():
    det = np.array([[10, 0, 60, 40, 0.9, 1]], np.float32)
    out = draw_detections(anh(), det, NAMES)
    assert out.shape == (200, 200, 3)
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
        np.array([[0, 0, 10, 10, 0], [0, 0, 10, 10, 1]], np.float32),
        np.array([[0, 0, 10, 10, 0]], np.float32),
        np.array([[0, 0, 10, 10, 1]], np.float32),
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


def test_nhan_o_chi_dung_ascii():
    from ivid.visualize import nhan_o

    t = nhan_o("rolled-in_scale", 2, 3)
    assert t.isascii(), f"tiêu đề có ký tự ngoài ASCII: {t!r}"
    assert "2" in t and "3" in t


def test_nhan_khong_tran_qua_mep_phai():
    img = anh(200, 200)
    det = np.array([[190, 100, 199, 150, 0.87, 3]], np.float32)
    out = draw_detections(img, det, NAMES)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_nhan_o_ghi_ro_dau_la_du_doan_dau_la_that():
    from ivid.visualize import nhan_o

    t = nhan_o("crazing", 2, 3)
    assert "du doan" in t and "that" in t


def test_co_chu_tu_thu_nho_de_chu_dai_van_vua():
    import cv2

    from ivid.visualize import co_chu_vua, nhan_o

    for lop in ("crazing", "rolled-in_scale", "pitted_surface"):
        t = nhan_o(lop, 5, 4)
        co = co_chu_vua(t, 320)
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, co, 1)
        assert tw <= 320 - 16, f"{lop}: chu rong {tw}px, không vừa ô 320px"


def test_chu_ngan_van_duoc_co_lon_nhat():
    from ivid.visualize import co_chu_vua

    assert co_chu_vua("a", 320) == 0.45


def test_chu_qua_dai_thi_dung_o_co_nho_nhat_chu_khong_lap_vo_han():
    from ivid.visualize import co_chu_vua

    assert co_chu_vua("x" * 500, 320) == 0.28
