from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.export.to_onnx import judge  # noqa: E402

TOL_RAW, TOL_BOX, TOL_SCORE = 1e-3, 0.1, 1e-3


def par(raw: float, box_px: float, score: float, match: float = 1.0) -> dict:
    return {"max_abs_diff": raw, "box_max_diff_px": box_px,
            "score_max_abs_diff": score, "box_max_relative": 1e-6,
            "detections": {"match_rate": match}}


def test_lech_o_toa_do_duoi_mot_phan_pixel_thi_khong_phai_that_bai():
    v = judge(par(raw=2.197e-3, box_px=2.197e-3, score=1e-7), TOL_RAW, TOL_BOX, TOL_SCORE)
    assert v["dat_tat_ca"], "lech 0.002 pixel tren anh 640 khong the doi detection"
    assert not v["srs_raw_metric"]["dat"], "chi so tho van phai bao la vuot, de doi chieu"


def test_lech_o_diem_so_lop_thi_PHAI_that_bai():
    v = judge(par(raw=5e-3, box_px=1e-6, score=5e-3), TOL_RAW, TOL_BOX, TOL_SCORE)
    assert not v["dat_tat_ca"]
    assert not v["checks"]["diem_so_lop"]["dat"]
    assert v["checks"]["toa_do_hop"]["dat"]


def test_toa_do_lech_nua_pixel_thi_that_bai():
    v = judge(par(raw=0.5, box_px=0.5, score=1e-7), TOL_RAW, TOL_BOX, TOL_SCORE)
    assert not v["dat_tat_ca"]
    assert not v["checks"]["toa_do_hop"]["dat"]


def test_detection_lech_thi_that_bai_du_moi_so_khac_deu_nho():
    v = judge(par(raw=1e-6, box_px=1e-6, score=1e-6, match=0.98), TOL_RAW, TOL_BOX, TOL_SCORE)
    assert not v["dat_tat_ca"]
    assert not v["checks"]["detection_cuoi_cung"]["dat"]


def test_moi_thu_hoan_hao_thi_dat_het():
    v = judge(par(raw=1e-7, box_px=1e-7, score=1e-8), TOL_RAW, TOL_BOX, TOL_SCORE)
    assert v["dat_tat_ca"]
    assert v["srs_raw_metric"]["dat"]
    assert all(c["dat"] for c in v["checks"].values())


def test_bao_cao_giu_lai_chi_so_tho_cua_SRS():
    v = judge(par(raw=2.2e-3, box_px=2.2e-3, score=1e-7), TOL_RAW, TOL_BOX, TOL_SCORE)
    assert v["srs_raw_metric"]["gia_tri"] == 2.2e-3
    assert v["srs_raw_metric"]["nguong"] == TOL_RAW
    assert "SRS" in v["srs_raw_metric"]["ghi_chu"]
