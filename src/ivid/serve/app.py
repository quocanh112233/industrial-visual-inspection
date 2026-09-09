"""FR-16 / FR-18 / FR-20 — Dich vu REST phat hien loi be mat.

    IVID_BACKEND=tensorrt uvicorn ivid.serve.app:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /predict   nhan file anh -> danh sach {class, confidence, bbox} + thoi gian
    GET  /health    trang thai, backend, phien ban model
    GET  /metrics   so request va do tre trung binh
    GET  /          thong tin ngan + link toi /docs
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .backends import BackendHolder, Settings
from .schemas import (
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    Timing,
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff",
                 "application/octet-stream"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class Metrics:
    """Bo dem don gian, giu 1000 do tre gan nhat de tinh phan vi."""

    def __init__(self, keep: int = 1000):
        self.lock = threading.Lock()
        self.requests_total = 0
        self.requests_failed = 0
        self.predictions_total = 0
        self.latencies: deque[float] = deque(maxlen=keep)

    def record(self, ms: float, n_det: int) -> None:
        with self.lock:
            self.requests_total += 1
            self.predictions_total += n_det
            self.latencies.append(ms)

    def fail(self) -> None:
        with self.lock:
            self.requests_total += 1
            self.requests_failed += 1

    def snapshot(self) -> dict:
        with self.lock:
            lat = list(self.latencies)
        if not lat:
            return {"requests_total": self.requests_total,
                    "requests_failed": self.requests_failed,
                    "predictions_total": self.predictions_total,
                    "latency": {}}
        s = sorted(lat)
        return {
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "predictions_total": self.predictions_total,
            "latency": {
                "mean_ms": round(statistics.fmean(s), 2),
                "p50_ms": round(s[len(s) // 2], 2),
                "p95_ms": round(s[max(0, int(0.95 * len(s)) - 1)], 2),
                "min_ms": round(s[0], 2),
                "max_ms": round(s[-1], 2),
                "window": len(s),
            },
        }


settings = Settings()
holder = BackendHolder(settings)
metrics = Metrics()
STARTED = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nap model luc khoi dong nhung KHONG lam sap tien trinh neu that bai:
    # /health con phai tra loi duoc de biet vi sao.
    holder.load()
    yield
    holder.close()


app = FastAPI(
    title="IVID — Industrial Visual Inspection",
    description="Phát hiện lỗi bề mặt thép cán nóng. Chọn runtime bằng biến môi trường "
                "`IVID_BACKEND` (pytorch | onnx | tensorrt).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "IVID",
        "backend": settings.backend,
        "model": settings.model,
        "endpoints": ["/predict", "/health", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    info = holder.info()
    return HealthResponse(
        status="ok" if info["loaded"] else "degraded",
        backend=info["backend"],
        model=info["model"],
        model_version=info["model_version"],
        imgsz=info["imgsz"],
        device=info["device"],
        loaded=info["loaded"],
        error=info["error"],
        uptime_seconds=round(time.time() - STARTED, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    snap = metrics.snapshot()
    return MetricsResponse(
        requests_total=snap["requests_total"],
        requests_failed=snap["requests_failed"],
        predictions_total=snap["predictions_total"],
        latency_ms=snap["latency"],
        uptime_seconds=round(time.time() - STARTED, 1),
        backend=settings.backend,
    )


def _decode(raw: bytes) -> np.ndarray:
    """bytes -> anh BGR. Nem HTTPException 400 neu khong phai anh doc duoc (FR-20)."""
    import cv2

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Không giải mã được ảnh. File hỏng hoặc không phải định dạng ảnh hỗ trợ "
                   f"({', '.join(sorted(ALLOWED_EXT))}).",
        )
    return img


@app.post("/predict", response_model=PredictResponse,
          responses={400: {"description": "File không hợp lệ"},
                     503: {"description": "Model chưa nạp được"}})
async def predict(file: UploadFile = File(..., description="Ảnh cần kiểm tra")):
    t_start = time.perf_counter()

    # --- FR-20: kiem tra dau vao truoc khi lam bat cu viec gi ---
    name = (file.filename or "").lower()
    ext_ok = any(name.endswith(e) for e in ALLOWED_EXT)
    type_ok = (file.content_type or "") in ALLOWED_TYPES
    if not (ext_ok or type_ok):
        metrics.fail()
        raise HTTPException(
            status_code=400,
            detail=f"Loại file không được hỗ trợ (content-type={file.content_type!r}, "
                   f"tên={file.filename!r}). Chấp nhận: {', '.join(sorted(ALLOWED_EXT))}.",
        )

    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) == 0:
        metrics.fail()
        raise HTTPException(status_code=400, detail="File rỗng.")
    if len(raw) > settings.max_upload_bytes:
        metrics.fail()
        raise HTTPException(
            status_code=400,
            detail=f"File quá lớn (> {settings.max_upload_mb:.0f} MB).",
        )

    if not holder.loaded:
        holder.load()                      # thu nap lai: trong so co the vua duoc copy vao
    if not holder.loaded:
        metrics.fail()
        raise HTTPException(
            status_code=503,
            detail=f"Model chưa nạp được: {holder.error}. Xem GET /health.",
        )

    try:
        img = _decode(raw)
    except HTTPException:
        metrics.fail()
        raise

    try:
        out = holder.runner.run_array(img)
    except Exception as e:
        metrics.fail()
        raise HTTPException(status_code=500,
                            detail=f"Inference thất bại: {type(e).__name__}: {e}") from e

    dets = [
        Detection(
            **{"class": holder.class_name(int(d[5]))},
            class_id=int(d[5]),
            confidence=round(float(d[4]), 4),
            bbox=[round(float(v), 2) for v in d[:4]],
        )
        for d in out.detections
    ]

    total_ms = (time.perf_counter() - t_start) * 1000
    metrics.record(total_ms, len(dets))

    h, w = img.shape[:2]
    return PredictResponse(
        detections=dets,
        count=len(dets),
        image={"width": w, "height": h, "bytes": len(raw), "filename": file.filename},
        timing=Timing(
            preprocess_ms=round(out.timing.preprocess_ms, 3),
            inference_ms=round(out.timing.inference_ms, 3),
            postprocess_ms=round(out.timing.postprocess_ms, 3),
            # total do o tang HTTP nen lon hon tong ba phan: gom ca doc file va dung JSON
            total_ms=round(total_ms, 3),
        ),
        backend=settings.backend,
        model=settings.model,
    )


@app.exception_handler(413)
async def too_large(request, exc):
    return JSONResponse(status_code=400,
                        content={"detail": f"File quá lớn (> {settings.max_upload_mb:.0f} MB)."})
