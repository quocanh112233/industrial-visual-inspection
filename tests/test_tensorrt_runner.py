from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.benchmark.runners.tensorrt_runner import kiem_tra  # noqa: E402


class MaLoi(int):
    def __new__(cls, gia_tri, ten):
        o = super().__new__(cls, gia_tri)
        o.name = ten
        return o


OK = MaLoi(0, "cudaSuccess")
HET_BO_NHO = MaLoi(2, "cudaErrorMemoryAllocation")


def test_ma_thanh_cong_tra_ve_gia_tri():
    assert kiem_tra((OK, 140234)) == 140234


def test_ma_thanh_cong_khong_kem_gia_tri_thi_tra_ve_none():
    assert kiem_tra((OK,)) is None


def test_ma_thanh_cong_nhieu_gia_tri_tra_ve_tuple():
    assert kiem_tra((OK, 1, 2)) == (1, 2)


def test_ma_loi_thi_nem_ngoai_le():
    with pytest.raises(RuntimeError, match="CUDA lỗi 2"):
        kiem_tra((HET_BO_NHO, 0))


def test_thong_bao_loi_co_ten_ma_loi():
    with pytest.raises(RuntimeError, match="cudaErrorMemoryAllocation"):
        kiem_tra((HET_BO_NHO,))


def test_nhan_ca_gia_tri_khong_phai_tuple():
    assert kiem_tra(OK) is None
    with pytest.raises(RuntimeError):
        kiem_tra(HET_BO_NHO)
