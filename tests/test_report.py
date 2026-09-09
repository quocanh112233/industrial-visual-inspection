"""Test sinh bao cao benchmark (FR-15).

Dung du lieu benchmark gia: test kiem tra LOGIC rut ket luan, khong kiem tra
so lieu that. Muc tieu la bat cac loi suy luan — vi du de xuat mot cau hinh
cham hon chi vi no hon 0.001 mAP.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.benchmark.report import (  # noqa: E402
    NGUONG_NHIEU_MAP,
    conclusions,
    main_table,
    rows_from,
)


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
                          "system_used_delta_mb": 700 + gpu},
            "backend_info": {"model_size_mb": size, "running_on_gpu": True}}


def make_acc(m50: float) -> dict:
    """Dang tra ve cua ivid.benchmark.accuracy sau khi bo ultralytics.val():
    mAP nam o cap cao nhat, per_class danh so theo CHI SO lop."""
    return {"skipped": False, "mAP50": m50, "mAP50_95": round(m50 * 0.54, 4),
            "precision": round(m50 - 0.05, 4), "recall": round(m50 - 0.09, 4),
            "class_names": ["crazing", "inclusion", "patches",
                            "pitted_surface", "rolled-in_scale", "scratches"],
            "per_class": {i: {"mAP50": round(m50 + (i - 3) * 0.05, 4), "n_gt": 100}
                          for i in range(6)}}


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
    for col in ("mAP@0.5", "p50 (ms)", "p95 (ms)", "FPS", "Model size",
                "RAM tiến trình", "Nạp (s)"):
        assert col in header
    assert "GPU mem" not in header, (
        "Cot bo nho chinh khong duoc lay so cua torch: no bao 0 MB cho ONNX "
        "Runtime va bo qua bo nho engine cua TensorRT")
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


# ------------------------------------------------- bao cao phai tat dinh
def test_sinh_bao_cao_hai_lan_cho_ra_noi_dung_giong_het():
    """Bao cao duoc commit vao repo. Neu no dong dau thoi gian CHAY, thi may dev,
    Jetson va Colab moi may lai tao mot diff khac nhau du noi dung y het —
    va 'git pull' tren Jetson bao xung dot. Da xay ra that.
    """
    from ivid.benchmark.report import build_report

    payload = {
        "device": {"board_model": "Jetson", "l4t_release": "R36", "jetpack": "6.2",
                   "power_mode": {"current_name": "15W", "current_id": 0,
                                  "available": {"0": "15W"}},
                   "libraries": {"tensorrt": "10.3.0"}, "temperature_mean_c": 48.0,
                   "collected_utc": "2026-09-09T10:00:00+00:00"},
        "config": {"warmup": 20, "sessions": 3},
        "images": {"count": 270, "split": "test"},
        "benchmarked_utc": "2026-09-09T10:05:00+00:00",
        **BASE,
    }
    rows = rows_from(payload)
    a = build_report(payload, rows, [], 200.0)
    b = build_report(payload, rows, [], 200.0)
    assert a == b, "sinh hai lan tu cung mot du lieu ra hai noi dung khac nhau"
    assert "2026-09-09T10:05:00" in a, "bao cao phai ghi thoi diem DO"


def test_ma_sinh_bao_cao_khong_dong_dau_thoi_gian_chay():
    """Chan viec vo tinh dua datetime.now() tro lai phan tieu de bao cao."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("src/ivid/data/validate.py", "src/ivid/data/stats.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "datetime.now" not in text, (
            f"{rel} dong dau thoi gian chay vao bao cao duoc commit -> gay diff gia")


# ------------------------------------------------------- cot RAM (FR-12)
def test_cot_ram_lay_tu_memprobe_chu_khong_lay_so_do_kem_benchmark():
    """Con so RAM phai den tu khoi "memory" (memprobe, moi cau hinh mot tien
    trinh), khong duoc lay `system_used_delta_mb` — cai do dem ca may nen hai
    lan chay cung cau hinh ra 31.2 MB va 0.0 MB."""
    payload = dict(BASE)
    payload["memory"] = {
        "yolov8n|pytorch": {"skipped": False, "rss_delta_mb": 1450.0},
        "yolov8n|tensorrt": {"skipped": False, "rss_delta_mb": 620.0},
    }
    bang = main_table(rows_from(payload))
    dong_pt = next(x for x in bang if "PyTorch" in x)
    dong_trt = next(x for x in bang if "TensorRT" in x)
    assert "1450 MB" in dong_pt
    assert "620 MB" in dong_trt
    # 700 + gpu = 1100 / 880 la system_used_delta_mb cua du lieu gia — khong duoc dung
    assert "1100 MB" not in dong_pt and "880 MB" not in dong_trt


def test_chua_do_ram_thi_bao_ro_chu_khong_bia_so():
    """Khong co khoi "memory" thi o do phai noi ro la chua do. Dien 0 hoac bo
    trong se bi doc nham thanh "runtime nay khong ton RAM"."""
    bang = main_table(rows_from(BASE))
    for dong in bang[2:]:
        assert "make mem" in dong


def test_cau_hinh_bi_bo_qua_khi_do_ram_khong_lam_hong_bang():
    payload = dict(BASE)
    payload["memory"] = {"yolov8n|pytorch": {"skipped": True, "reason": "khong thay best.pt"}}
    bang = main_table(rows_from(payload))
    assert "make mem" in next(x for x in bang if "PyTorch" in x)


# ------------------------------------------- chi phi khoi dong (do thay tren Jetson)
def _payload_co_thoi_gian_nap(nap_onnx: float, nap_trt: float = 0.3) -> dict:
    d = {"runs": dict(BASE["runs"]), "accuracy": dict(BASE["accuracy"])}
    d["runs"]["yolov8n|onnx"] = make_run(18.5, 12.3, 200)
    d["accuracy"]["yolov8n|onnx"] = make_acc(0.7356)
    d["memory"] = {
        "yolov8n|pytorch": {"skipped": False, "rss_delta_mb": 909.0, "nap_giay": 3.0},
        "yolov8n|onnx": {"skipped": False, "rss_delta_mb": 1856.0, "nap_giay": nap_onnx},
        "yolov8n|tensorrt": {"skipped": False, "rss_delta_mb": 301.0, "nap_giay": nap_trt},
    }
    return d


def test_bao_cao_neu_bat_duoc_chi_phi_khoi_dong_lon():
    """ONNX Runtime chay TensorrtExecutionProvider tu dung engine luc nap: do
    duoc 100s so voi 0.3s cua engine dung san. Cot p50 khong he thay dieu do."""
    L = conclusions(rows_from(_payload_co_thoi_gian_nap(100.5)), cycle_ms=1000.0)
    van_ban = "\n".join(L)
    assert "Chi phí khởi động" in van_ban
    assert "100 s" in van_ban and "335×" in van_ban
    assert "trt_engine_cache_enable" in van_ban


def test_khong_bia_ra_canh_bao_khi_thoi_gian_nap_tuong_duong():
    L = conclusions(rows_from(_payload_co_thoi_gian_nap(0.9)), cycle_ms=1000.0)
    assert "Chi phí khởi động" not in "\n".join(L)


def test_thieu_so_lieu_nap_thi_khong_ket_luan():
    d = _payload_co_thoi_gian_nap(100.5)
    d["memory"]["yolov8n|onnx"].pop("nap_giay")
    assert "Chi phí khởi động" not in "\n".join(conclusions(rows_from(d), cycle_ms=1000.0))


# ------------------------------------ nguong nhieu khi so hai model (SRS 7.3 cau 3)
def _hai_model(map_n: float, map_s: float, p50_s: float = 18.4) -> dict:
    return {
        "runs": {"yolov8n|tensorrt": make_run(14.2, 9.0, 180),
                 "yolov8s|tensorrt": make_run(p50_s, 25.6, 200)},
        "accuracy": {"yolov8n|tensorrt": make_acc(map_n),
                     "yolov8s|tensorrt": make_acc(map_s)},
    }


def test_chenh_lech_duoi_nguong_nhieu_thi_ket_luan_la_khong_phan_biet_duoc():
    """Khong duoc noi 'YOLOv8s kem hon' khi chenh lech nho hon muc ma chinh
    quy trinh train tai lap lai duoc."""
    van_ban = "\n".join(conclusions(rows_from(_hai_model(0.7357, 0.7270)), cycle_ms=200.0))
    assert "không phân biệt được" in van_ban
    assert "reproducibility.md" in van_ban


def test_chenh_lech_tren_nguong_nhieu_thi_moi_ket_luan_la_dang():
    van_ban = "\n".join(
        conclusions(rows_from(_hai_model(0.70, 0.75)), cycle_ms=200.0))
    assert "**Đáng.**" in van_ban
    assert "không phân biệt được" not in van_ban


def test_vuot_nhip_day_chuyen_thi_khong_dang_du_mAP_cao_hon():
    van_ban = "\n".join(
        conclusions(rows_from(_hai_model(0.70, 0.75, p50_s=400.0)), cycle_ms=200.0))
    assert "vượt ngân sách thời gian" in van_ban


def test_nguong_nhieu_khop_voi_so_do_duoc_trong_tai_lieu():
    """0.0115 = |0.7749 - 0.7634|, hai lan train cung seed. Neu ai do sua hang so
    nay ma khong sua tai lieu thi test do."""
    assert round(abs(0.7749 - 0.7634), 4) == NGUONG_NHIEU_MAP
    tl = (Path(__file__).resolve().parents[1] / "docs/reproducibility.md")
    if tl.exists():
        assert "0.7749" in tl.read_text(encoding="utf-8")
