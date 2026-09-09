"""Test giai ma dau ra YOLOv8 va NMS (ivid.postprocess).

Day la doan code duoc dung boi CA BA runtime. Neu no sai, ca ba dinh dang cung
sai giong nhau — FR-09 se bao "khop 100%" trong khi thuc te tat ca deu sai.
Vi vay no phai duoc test bang du lieu tu dung, doc lap voi model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.postprocess import decode, match_detections, nms, scale_boxes, xywh2xyxy  # noqa: E402

NC = 6  # 6 lop NEU-DET


def raw_with(boxes: list[tuple[float, float, float, float, int, float]], n_slots: int = 20):
    """Dung tensor tho (1, 4+NC, N) tu danh sach (cx, cy, w, h, class, score)."""
    out = np.zeros((1, 4 + NC, n_slots), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, 0, i], out[0, 1, i], out[0, 2, i], out[0, 3, i] = cx, cy, w, h
        out[0, 4 + c, i] = s
    return out


# ------------------------------------------------------------------- co ban
def test_xywh2xyxy():
    got = xywh2xyxy(np.array([[100.0, 100.0, 40.0, 20.0]]))
    assert got.tolist() == [[80.0, 90.0, 120.0, 110.0]]


def test_decode_lay_dung_box_va_lop():
    raw = raw_with([(100, 100, 40, 20, 3, 0.9)])
    det = decode(raw, conf_thres=0.25)
    assert det.shape == (1, 6)
    assert det[0, :4].tolist() == [80.0, 90.0, 120.0, 110.0]
    assert det[0, 4] == pytest.approx(0.9)
    assert int(det[0, 5]) == 3


def test_decode_loc_theo_nguong_tin_cay():
    raw = raw_with([(100, 100, 40, 20, 0, 0.9), (300, 300, 40, 20, 1, 0.1)])
    assert len(decode(raw, conf_thres=0.25)) == 1
    assert len(decode(raw, conf_thres=0.05)) == 2


def test_decode_khong_co_gi_vuot_nguong_thi_tra_ve_rong():
    raw = raw_with([(100, 100, 40, 20, 0, 0.05)])
    det = decode(raw, conf_thres=0.25)
    assert det.shape == (0, 6)


def test_decode_chon_lop_co_diem_cao_nhat():
    raw = np.zeros((1, 4 + NC, 20), dtype=np.float32)
    raw[0, :4, 0] = [100, 100, 40, 20]
    raw[0, 4 + 1, 0] = 0.30
    raw[0, 4 + 4, 0] = 0.80          # lop 4 cao hon
    det = decode(raw, conf_thres=0.25)
    assert int(det[0, 5]) == 4
    assert det[0, 4] == pytest.approx(0.80)


def test_decode_can_biet_nc_khi_so_anchor_qua_nho():
    """Suy doan truc tu dong dua tren "so anchor > so kenh". Voi tensor do choi
    co 5 anchor va 10 kenh, gia thiet do sai — luc nay phai truyen nc."""
    raw = np.zeros((1, 4 + NC, 5), dtype=np.float32)
    raw[0, :4, 0] = [100, 100, 40, 20]
    raw[0, 4 + 4, 0] = 0.80

    assert len(decode(raw, conf_thres=0.25)) == 0            # doan sai, khong bat duoc gi
    det = decode(raw, conf_thres=0.25, nc=NC)                # noi ro thi dung
    assert len(det) == 1 and int(det[0, 5]) == 4


def test_decode_bao_loi_khi_nc_khong_khop():
    raw = np.zeros((1, 4 + NC, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="khong khop nc"):
        decode(raw, nc=99)


def test_decode_nhan_ca_dang_chuyen_vi():
    """Mot so engine tra ve (N, 4+nc) thay vi (4+nc, N)."""
    raw = raw_with([(100, 100, 40, 20, 2, 0.9)])
    a = decode(raw, conf_thres=0.25)
    b = decode(raw[0].T[None], conf_thres=0.25)
    assert np.allclose(a, b)


# --------------------------------------------------------------------- NMS
def test_nms_gop_hai_box_chong_nhau():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0]          # IoU ~0.82 > 0.5 -> bo cai thu hai


def test_nms_giu_hai_box_roi_nhau():
    boxes = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1]


def test_nms_giu_box_diem_cao_hon():
    boxes = np.array([[0, 0, 100, 100], [2, 2, 102, 102]], dtype=np.float32)
    scores = np.array([0.4, 0.95], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [1]


def test_nms_lam_theo_tung_lop_khong_trie_tieu_lop_khac():
    """Tren be mat thep, hai loai loi khac nhau CO THE chong len nhau.
    NMS phai giu ca hai thay vi bo mot."""
    raw = raw_with([(100, 100, 60, 60, 0, 0.9),
                    (102, 102, 60, 60, 3, 0.85)])   # gan nhu trung, khac lop
    det = decode(raw, conf_thres=0.25, iou_thres=0.5)
    assert len(det) == 2
    assert sorted(int(d) for d in det[:, 5]) == [0, 3]


def test_nms_gop_khi_cung_lop():
    raw = raw_with([(100, 100, 60, 60, 0, 0.9),
                    (102, 102, 60, 60, 0, 0.85)])   # cung lop -> gop
    assert len(decode(raw, conf_thres=0.25, iou_thres=0.5)) == 1


def test_ket_qua_sap_theo_diem_giam_dan():
    raw = raw_with([(50, 50, 20, 20, 0, 0.4), (200, 200, 20, 20, 1, 0.9),
                    (350, 350, 20, 20, 2, 0.6)])
    det = decode(raw, conf_thres=0.25)
    assert list(det[:, 4]) == sorted(det[:, 4], reverse=True)


# ------------------------------------------------------------- scale_boxes
def test_scale_boxes_dua_ve_toa_do_anh_goc():
    """Anh 200x200 -> letterbox 640 (ti le 3.2, khong dem vien)."""
    det = np.array([[64.0, 64.0, 320.0, 320.0, 0.9, 0]], dtype=np.float32)
    out = scale_boxes(det, ratio=3.2, pad=(0, 0), orig_hw=(200, 200))
    assert out[0, :4].tolist() == [20.0, 20.0, 100.0, 100.0]


def test_scale_boxes_tru_phan_dem_vien():
    det = np.array([[110.0, 60.0, 210.0, 160.0, 0.9, 0]], dtype=np.float32)
    out = scale_boxes(det, ratio=2.0, pad=(10, 20), orig_hw=(500, 500))
    assert out[0, :4].tolist() == [50.0, 20.0, 100.0, 70.0]


def test_scale_boxes_kep_vao_trong_bien_anh():
    det = np.array([[-50.0, -50.0, 5000.0, 5000.0, 0.9, 0]], dtype=np.float32)
    out = scale_boxes(det, ratio=1.0, pad=(0, 0), orig_hw=(200, 300))
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert out[0, 2] == 300.0 and out[0, 3] == 200.0


def test_scale_boxes_voi_mang_rong():
    assert len(scale_boxes(np.zeros((0, 6), np.float32), 1.0, (0, 0), (10, 10))) == 0


# -------------------------------------------------- match_detections (FR-09)
def test_match_hai_tap_giong_het_nhau():
    a = np.array([[0, 0, 50, 50, 0.9, 1], [100, 100, 150, 150, 0.8, 3]], dtype=np.float32)
    r = match_detections(a, a.copy())
    assert r["matched"] == 2 and r["only_a"] == 0 and r["only_b"] == 0
    assert r["mean_iou"] == pytest.approx(1.0)
    assert r["max_conf_diff"] == pytest.approx(0.0)


def test_match_phat_hien_detection_thua():
    a = np.array([[0, 0, 50, 50, 0.9, 1]], dtype=np.float32)
    b = np.array([[0, 0, 50, 50, 0.9, 1], [200, 200, 250, 250, 0.7, 2]], dtype=np.float32)
    r = match_detections(a, b)
    assert r["matched"] == 1 and r["only_a"] == 0 and r["only_b"] == 1


def test_match_khong_ghep_box_khac_lop():
    a = np.array([[0, 0, 50, 50, 0.9, 1]], dtype=np.float32)
    b = np.array([[0, 0, 50, 50, 0.9, 4]], dtype=np.float32)   # trung vi tri, khac lop
    r = match_detections(a, b)
    assert r["matched"] == 0 and r["only_a"] == 1 and r["only_b"] == 1


def test_match_ghi_nhan_lech_diem_tin_cay():
    a = np.array([[0, 0, 50, 50, 0.90, 1]], dtype=np.float32)
    b = np.array([[0, 0, 50, 50, 0.83, 1]], dtype=np.float32)
    r = match_detections(a, b)
    assert r["matched"] == 1
    assert r["max_conf_diff"] == pytest.approx(0.07, abs=1e-4)


def test_match_hai_tap_deu_rong_la_khop_hoan_toan():
    z = np.zeros((0, 6), np.float32)
    r = match_detections(z, z)
    assert r["matched"] == 0 and r["mean_iou"] == 1.0
