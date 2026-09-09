"""Test cho phep tinh mAP tu viet (ivid.benchmark.metrics).

Day la doan code quyet dinh cot mAP cua ca bang benchmark. Neu no sai, ca ba
dinh dang cung sai giong nhau va khong co gi lo ra. Vi vay no duoc kiem bang
cac truong hop TU DUNG co dap an biet truoc, khong phu thuoc model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.benchmark.metrics import box_iou_matrix, compute_map  # noqa: E402


def det(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def gt(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


# ------------------------------------------------------------------- IoU
def test_iou_hai_hop_trung_nhau():
    a = np.array([[0, 0, 10, 10]], np.float32)
    assert box_iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_hai_hop_roi_nhau():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[20, 20, 30, 30]], np.float32)
    assert box_iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_chong_mot_nua():
    a = np.array([[0, 0, 10, 10]], np.float32)     # dien tich 100
    b = np.array([[0, 0, 10, 5]], np.float32)      # dien tich 50, nam tron trong a
    assert box_iou_matrix(a, b)[0, 0] == pytest.approx(0.5)


def test_iou_mang_rong():
    assert box_iou_matrix(np.zeros((0, 4), np.float32),
                          np.array([[0, 0, 1, 1]], np.float32)).shape == (0, 1)


# ------------------------------------------------------------------ mAP
# AP TOI DA la 0.995 chu khong phai 1.0 — day khong phai loi.
# Cong thuc noi suy 101 diem cua ultralytics them mot diem moc (recall=1,
# precision=0) o cuoi duong PR, nen doan cuoi cung mat 0.005 dien tich. Ta co y
# khop voi ultralytics de doi chieu duoc con so cua ban .pt (xem
# test_khop_voi_so_cua_ultralytics ben duoi), thay vi dung cong thuc cua
# pycocotools (lay trung binh 101 diem, cho ra dung 1.0).
AP_MAX = 0.995


def test_du_doan_hoan_hao_cho_AP_toi_da():
    d = [det([10, 10, 50, 50, 0.9, 0])]
    g = [gt([10, 10, 50, 50, 0])]
    m = compute_map(d, g, nc=1)
    assert m["mAP50"] == pytest.approx(AP_MAX, abs=1e-3)
    assert m["mAP50_95"] == pytest.approx(AP_MAX, abs=1e-3)
    assert m["recall"] == pytest.approx(1.0)


def test_khong_du_doan_gi_thi_mAP_bang_0():
    m = compute_map([np.zeros((0, 6), np.float32)], [gt([0, 0, 10, 10, 0])], nc=1)
    assert m["mAP50"] == 0.0
    assert m["n_ground_truth"] == 1


def test_du_doan_sai_lop_thi_khong_tinh_la_dung():
    d = [det([10, 10, 50, 50, 0.9, 1])]   # dung vi tri, sai lop
    g = [gt([10, 10, 50, 50, 0])]
    assert compute_map(d, g, nc=2)["mAP50"] == pytest.approx(0.0, abs=1e-3)


def test_bat_duoc_mot_trong_hai_thi_recall_mot_nua():
    d = [det([0, 0, 10, 10, 0.9, 0])]
    g = [gt([0, 0, 10, 10, 0], [100, 100, 110, 110, 0])]
    m = compute_map(d, g, nc=1)
    assert m["recall"] == pytest.approx(0.5)
    # Cong thuc 'interp' cua ultralytics noi suy tuyen tinh phan duoi cua duong
    # PR, nen recall 0.5 voi precision 1.0 cho AP = 0.75 (khong phai ~0.5 nhu
    # pycocotools). Khac biet nay nhat quan cho moi dinh dang nen khong anh
    # huong toi viec SO SANH ba runtime — thu ma bang benchmark can.
    assert m["mAP50"] == pytest.approx(0.75, abs=0.01)


def test_du_doan_trung_lap_bi_tinh_la_FP():
    """Hai hop cung mot GT: cai thu hai phai la false positive."""
    d = [det([0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0])]
    g = [gt([0, 0, 10, 10, 0])]
    m = compute_map(d, g, nc=1)
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)   # tai diem F1 lon nhat, chi lay hop dau
    assert m["mAP50"] == pytest.approx(AP_MAX, abs=1e-3)


def test_hop_lech_nhe_dat_o_IoU_thap_nhung_truot_o_IoU_cao():
    """Day la ly do mAP50-95 nhay hon mAP50 nhieu lan."""
    d = [det([0, 0, 10, 12, 0.9, 0])]     # IoU voi GT 10x10 = 100/120 = 0.833
    g = [gt([0, 0, 10, 10, 0])]
    m = compute_map(d, g, nc=1)
    assert m["mAP50"] == pytest.approx(AP_MAX, abs=1e-3)   # IoU 0.833 >= 0.5
    # dat 7/10 nguong (0.50..0.80), truot 0.85/0.90/0.95
    assert m["mAP50_95"] == pytest.approx(AP_MAX * 0.7, abs=0.01)


def test_diem_tin_cay_cao_hon_duoc_ghep_truoc():
    """Detection diem cao phai duoc uu tien ghep — neu khong, duong PR sai."""
    d = [det([0, 0, 10, 10, 0.5, 0], [0, 0, 10, 11, 0.95, 0])]
    g = [gt([0, 0, 10, 10, 0])]
    m = compute_map(d, g, nc=1)
    assert m["n_detections"] == 2
    assert m["recall"] == pytest.approx(1.0)


def test_trung_binh_tren_cac_lop_co_mat_khong_tinh_lop_vang_mat():
    d = [det([0, 0, 10, 10, 0.9, 0])]
    g = [gt([0, 0, 10, 10, 0])]
    m5 = compute_map(d, g, nc=5)          # 4 lop khong co GT nao
    assert m5["mAP50"] == pytest.approx(AP_MAX, abs=1e-3), "lop vang mat khong duoc keo mAP xuong"
    assert set(m5["per_class"]) == {0}


def test_nhieu_anh_nhieu_lop():
    d = [det([0, 0, 10, 10, 0.9, 0]),
         det([5, 5, 25, 25, 0.8, 1], [50, 50, 60, 60, 0.3, 0])]
    g = [gt([0, 0, 10, 10, 0]),
         gt([5, 5, 25, 25, 1])]
    m = compute_map(d, g, nc=2)
    assert m["n_images"] == 2 and m["n_detections"] == 3 and m["n_ground_truth"] == 2
    assert m["per_class"][0]["n_gt"] == 1 and m["per_class"][1]["n_gt"] == 1
    assert m["mAP50"] > 0.7      # ca hai deu bat duoc, co mot FP diem thap


def test_ket_qua_khong_phu_thuoc_thu_tu_detection_dau_vao():
    a = det([0, 0, 10, 10, 0.9, 0], [50, 50, 60, 60, 0.4, 0])
    b = det([50, 50, 60, 60, 0.4, 0], [0, 0, 10, 10, 0.9, 0])
    g = [gt([0, 0, 10, 10, 0], [50, 50, 60, 60, 0])]
    assert compute_map([a], g, nc=1) == compute_map([b], g, nc=1)


def test_cong_thuc_khop_voi_ultralytics_khong_phai_pycocotools():
    """Ghi ro su khac biet de nguoi doc sau nay khong nham.

    Cung mot du lieu (recall 0.5, precision 1.0):
      - ultralytics compute_ap(method='interp') -> 0.75
      - pycocotools (trung binh 101 diem)       -> ~0.505
    Ta chon ultralytics vi con so mAP cua ban .pt trong bao cao den tu ho, va
    ta muon doi chieu duoc. Dieu QUAN TRONG la cung mot cong thuc ap cho ca ba
    dinh dang — do moi la thu quyet dinh tinh dung dan cua bang so sanh.
    """
    d = [det([0, 0, 10, 10, 0.9, 0])]
    g = [gt([0, 0, 10, 10, 0], [100, 100, 110, 110, 0])]
    ap = compute_map(d, g, nc=1)["mAP50"]
    assert ap == pytest.approx(0.75, abs=0.01)
    assert ap > 0.6, "neu ra ~0.5 nghia la da doi sang cong thuc pycocotools"
