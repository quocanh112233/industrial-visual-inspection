"""Test 4 endpoint cua dich vu (FR-16, FR-18, FR-20).

Chay voi IVID_BACKEND=mock nen khong can GPU, TensorRT hay file trong so —
nho vay CI kiem tra duoc hop dong cua API. Phan inference that duoc kiem o
tests khac va o benchmark tren Jetson.
"""
from __future__ import annotations

import importlib
import io
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("cv2")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    os.environ["IVID_BACKEND"] = "mock"
    os.environ["IVID_MAX_UPLOAD_MB"] = "10"
    import ivid.serve.app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


def png_bytes(w: int = 200, h: int = 200) -> bytes:
    import cv2
    import numpy as np

    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------- /health
def test_health_bao_dung_backend_dang_dung(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["backend"] == "mock"          # FR-17: doi bien moi truong -> doi bao cao
    assert d["loaded"] is True
    assert d["model_version"]
    assert d["uptime_seconds"] >= 0


# ------------------------------------------------------------------ /predict
def test_predict_tra_ve_json_hop_le(client):
    r = client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["count"] == len(d["detections"])
    assert d["image"]["width"] == 200 and d["image"]["height"] == 200
    for det in d["detections"]:
        assert set(det) >= {"class", "class_id", "confidence", "bbox"}
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox"]) == 4
        x1, y1, x2, y2 = det["bbox"]
        assert x1 < x2 and y1 < y2
    t = d["timing"]
    assert t["total_ms"] > 0
    assert t["total_ms"] >= t["inference_ms"]


def test_predict_bbox_nam_trong_anh(client):
    r = client.post("/predict", files={"file": ("a.png", png_bytes(320, 240), "image/png")})
    d = r.json()
    for det in d["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 <= 320 and 0 <= x2 <= 320
        assert 0 <= y1 <= 240 and 0 <= y2 <= 240


# --------------------------------------------------- FR-20: dau vao khong hop le
def test_tu_choi_file_khong_phai_anh(client):
    r = client.post("/predict", files={"file": ("note.txt", b"khong phai anh", "text/plain")})
    assert r.status_code == 400
    assert "không được hỗ trợ" in r.json()["detail"].lower() or "hỗ trợ" in r.json()["detail"]


def test_tu_choi_anh_hong(client):
    """Duoi .png nhung noi dung la rac -> phai 400, khong duoc 500."""
    r = client.post("/predict", files={"file": ("hong.png", b"\x89PNG\r\n" + b"rac" * 50,
                                                "image/png")})
    assert r.status_code == 400
    assert "giải mã" in r.json()["detail"]


def test_tu_choi_file_qua_lon(client):
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)
    r = client.post("/predict", files={"file": ("big.png", io.BytesIO(big), "image/png")})
    assert r.status_code == 400
    assert "quá lớn" in r.json()["detail"]


def test_tu_choi_file_rong(client):
    r = client.post("/predict", files={"file": ("rong.png", b"", "image/png")})
    assert r.status_code == 400


def test_ba_loi_tren_khong_lam_sap_service(client):
    """FR-20: sau ca ba truong hop loi, service van phuc vu binh thuong."""
    client.post("/predict", files={"file": ("a.txt", b"x", "text/plain")})
    client.post("/predict", files={"file": ("b.png", b"rac", "image/png")})
    client.post("/predict", files={"file": ("c.png", b"\x00" * (11 * 1024 * 1024), "image/png")})
    r = client.post("/predict", files={"file": ("ok.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    assert client.get("/health").json()["status"] == "ok"


# ------------------------------------------------------------------ /metrics
def test_metrics_dem_dung_request(client):
    before = client.get("/metrics").json()
    for _ in range(3):
        client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    client.post("/predict", files={"file": ("a.txt", b"x", "text/plain")})
    after = client.get("/metrics").json()

    assert after["requests_total"] >= before["requests_total"] + 4
    assert after["requests_failed"] >= before["requests_failed"] + 1
    assert after["latency_ms"]["mean_ms"] > 0
    assert after["latency_ms"]["p95_ms"] >= after["latency_ms"]["p50_ms"]


def test_root_liet_ke_endpoint(client):
    d = client.get("/").json()
    assert set(d["endpoints"]) >= {"/predict", "/health", "/metrics"}
