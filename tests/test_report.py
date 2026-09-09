"""Test sinh bao cao benchmark (FR-15).

Dung du lieu benchmark gia: test kiem tra LOGIC rut ket luan, khong kiem tra
so lieu that. Muc tieu la bat cac loi suy luan — vi du de xuat mot cau hinh
cham hon chi vi no hon 0.001 mAP.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.benchmark.report import conclusions, main_table, rows_from  # noqa: E402


def make_run(p50: float, size: float = 6.2, gpu: float = 400) -> dict:
    st = {k: {"p50": p50, "p95": round(p50 * 1.08, 3)} for k in
          ("preprocess", "inference", "postprocess", "total")}
    st["preprocess"]["p50"] = 1.9
    st["inference"]["p50"] = round(p50 - 5.0, 3)
    st["postprocess"]["p50"] = 3.1
    return {"skipped": False, "median_stages": st, "latency_p50_ms": p50,
            "latency_p95_ms": round(p50 * 1.08, 3), "fps": round(1000 / p50, 1),
            "model_size_mb": size, "session_p50_spread_percent": 2.5,
            "resources": {"torch_gpu_peak_mb": gpu, "process_rss_peak_mb": 900,
                          "system_used_delta_mb": 700},
            "backend_info": {"model_size_mb": size, "running_on_gpu": True}}


def make_acc(m50: float) -> dict:
    return {"skipped": False,
            "overall": {"mAP50": m50, "mAP50_95": round(m50 * 0.54, 4),
                        "precision": m50 - 0.05, "recall": m50 - 0.09}}


BASE = {
    "runs": {"yolov8n|pytorch": make_run(27.4), "yolov8n|tensorrt": make_run(11.3, 9.3, 180)},
    "accuracy": {"yolov8n|pytorch": make_acc(0.7412), "yolov8n|tensorrt": make_acc(0.7385)},
}


def test_rows_giu_dung_thu_tu_runtime():
    rows = rows_from(BASE)
    assert [r["backend"] for r in rows] == ["pytorch", "tensorrt"]
    assert rows[0]["p50"] == 27.4


def test_bang_chinh_co_du_cot():
    lines = main_table(rows_from(BASE))
    header = lines[0]
    for col in ("mAP@0.5", "p50 (ms)", "p95 (ms)", "FPS", "Model size", "GPU mem"):
        assert col in header
    assert len(lines) == 2 + 2          # header + separator + 2 dong


def test_ket_luan_tinh_dung_ty_le_tang_toc():
    text = "\n".join(conclusions(rows_from(BASE), 200.0))
    assert "2.42×" in text              # 27.4 / 11.3
    assert "-0.0027" in text            # 0.7385 - 0.7412


def test_khuyen_nghi_khong_chon_cau_hinh_cham_chi_vi_hon_ti_mAP():
    """Loi suy luan de mac: PyTorch hon TensorRT 0.0027 mAP nhung cham 2.4 lan.

    Khuyen nghi phai la TensorRT.
    """
    text = "\n".join(conclusions(rows_from(BASE), 200.0))
    reco = [line for line in text.splitlines() if "Khuyến nghị" in line]
    assert reco, "khong sinh ra dong khuyen nghi"
    assert "TensorRT" in reco[0]
    assert "PyTorch" not in reco[0]


def test_khuyen_nghi_van_chon_mAP_cao_khi_chenh_lech_that_su():
    """Neu chenh lech mAP lon (> 0.005) thi phai chon cau hinh chinh xac hon."""
    data = {
        "runs": {"yolov8n|tensorrt": make_run(11.3), "yolov8s|tensorrt": make_run(21.0)},
        "accuracy": {"yolov8n|tensorrt": make_acc(0.7385), "yolov8s|tensorrt": make_acc(0.7900)},
    }
    reco = [line for line in "\n".join(conclusions(rows_from(data), 200.0)).splitlines()
            if "Khuyến nghị" in line]
    assert "yolov8s" in reco[0]


def test_bao_cao_noi_ro_khi_khong_cau_hinh_nao_dap_ung_nhip():
    text = "\n".join(conclusions(rows_from(BASE), 5.0))   # nhip 5 ms, khong ai dat
    assert "Không cấu hình nào đáp ứng" in text


def test_thieu_du_lieu_thi_bao_thieu_chu_khong_no():
    partial = {"runs": {"yolov8n|onnx": {"skipped": True, "reason": "khong thay best.onnx"}},
               "accuracy": {}}
    rows = rows_from(partial)
    assert rows[0]["latency_missing"] == "khong thay best.onnx"
    text = "\n".join(conclusions(rows, 200.0))
    assert "chưa đủ dữ liệu" in text or "chưa có dữ liệu" in text
