from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    cls: str = Field(..., alias="class", description="Ten loai loi be mat")
    class_id: int = Field(..., description="Chi so lop, khop thu tu trong configs/data.yaml")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[float] = Field(..., min_length=4, max_length=4,
                              description="[x1, y1, x2, y2] theo pixel cua ANH GOC")

    model_config = {"populate_by_name": True}


class Timing(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class PredictResponse(BaseModel):
    detections: list[Detection]
    count: int
    image: dict
    timing: Timing
    backend: str
    model: str


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    backend: str
    model: str
    model_version: str
    imgsz: int
    device: str
    loaded: bool
    error: str | None = None
    uptime_seconds: float


class MetricsResponse(BaseModel):
    requests_total: int
    requests_failed: int
    predictions_total: int
    latency_ms: dict
    uptime_seconds: float
    backend: str


class ErrorResponse(BaseModel):
    detail: str
