"""Giao dien chung cho ba runtime.

Moi runner chi chiu trach nhiem MOT viec: nhan mang NCHW float32 va tra ve
tensor tho. Tien xu ly (ivid.preprocess) va hau xu ly (ivid.postprocess) nam
ngoai, dung chung — nho vay chenh lech do duoc giua ba dinh dang chi den tu
buoc inference, dung nhu muc dich cua bang benchmark.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...postprocess import decode, scale_boxes
from ...preprocess import preprocess, read_image


def _num_classes() -> int | None:
    """Doc so lop tu configs/data.yaml. None neu khong doc duoc — luc do
    decode() quay ve cach doan truc, van dung voi model that."""
    try:
        import yaml

        from ...data.common import repo_root

        for p in (repo_root() / "data/processed/data.yaml", repo_root() / "configs/data.yaml"):
            if p.exists():
                d = yaml.safe_load(p.read_text())
                if "names" in d:
                    return len(d["names"])
    except Exception:
        pass
    return None


@dataclass
class Timing:
    """Thoi gian mot lan xu ly, tach lam ba phan (FR-11)."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class RunOutput:
    detections: np.ndarray                    # (M, 6) x1 y1 x2 y2 conf cls
    timing: Timing = field(default_factory=Timing)


class BaseRunner(abc.ABC):
    """Lop cha cho pytorch / onnx / tensorrt runner."""

    name: str = "base"

    def __init__(self, weights: Path, imgsz: int = 640, conf: float = 0.25,
                 iou: float = 0.7, device: str = "cuda", nc: int | None = None):
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        # So lop: truyen tuong minh cho decode() thay vi de no doan truc tensor
        self.nc = nc if nc is not None else _num_classes()
        if not self.weights.exists():
            raise FileNotFoundError(f"khong thay trong so: {self.weights}")
        self._load()

    # ------------------------------------------------------------ bat buoc
    @abc.abstractmethod
    def _load(self) -> None:
        """Nap model. Goi mot lan trong __init__."""

    @abc.abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (1,3,H,W) float32 -> tensor tho (1, 4+nc, N). Da dong bo GPU khi tra ve."""

    # ------------------------------------------------------------ mac dinh
    def backend_info(self) -> dict:
        return {
            "runner": self.name,
            "weights": str(self.weights),
            "model_size_mb": round(self.weights.stat().st_size / 1e6, 2),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
        }

    def warmup(self, n: int = 20) -> None:
        x = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(n):
            self.infer(x)

    def run_array(self, img_bgr: np.ndarray) -> RunOutput:
        """Chay tren mot anh BGR, do rieng ba giai doan."""
        t = Timing()

        t0 = time.perf_counter()
        x, ratio, pad = preprocess(img_bgr, self.imgsz)
        t.preprocess_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        raw = self.infer(x)
        t.inference_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        det = decode(raw, self.conf, self.iou, nc=self.nc)
        det = scale_boxes(det, ratio, pad, img_bgr.shape[:2])
        t.postprocess_ms = (time.perf_counter() - t0) * 1000

        return RunOutput(det, t)

    def run_file(self, path: Path) -> RunOutput:
        return self.run_array(read_image(path))

    def close(self) -> None:  # noqa: B027 — hook tuy chon, khong bat buoc override
        """Giai phong tai nguyen. Mac dinh khong lam gi."""


def build_runner(backend: str, weights: Path, **kw) -> BaseRunner:
    """Tao runner theo ten backend: pytorch | onnx | tensorrt."""
    backend = backend.lower()
    if backend in ("pytorch", "pt", "torch"):
        from .pytorch_runner import PyTorchRunner

        return PyTorchRunner(weights, **kw)
    if backend in ("onnx", "onnxruntime", "ort"):
        from .onnx_runner import OnnxRunner

        return OnnxRunner(weights, **kw)
    if backend in ("tensorrt", "trt", "engine"):
        from .tensorrt_runner import TensorRTRunner

        return TensorRTRunner(weights, **kw)
    raise ValueError(f"backend khong ho tro: {backend} (nhan: pytorch|onnx|tensorrt)")


def weights_for(model_dir: Path, backend: str) -> Path:
    """models/yolov8n + 'onnx' -> models/yolov8n/best.onnx"""
    suffix = {"pytorch": ".pt", "pt": ".pt", "torch": ".pt",
              "onnx": ".onnx", "onnxruntime": ".onnx", "ort": ".onnx",
              "tensorrt": ".engine", "trt": ".engine", "engine": ".engine"}[backend.lower()]
    return Path(model_dir) / f"best{suffix}"
