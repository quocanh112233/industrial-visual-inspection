from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...postprocess import decode, scale_boxes
from ...preprocess import preprocess, read_image


def _rel(path: Path) -> str:
    from ...data.common import rel_to_root

    return rel_to_root(path)


def _num_classes() -> int | None:
    try:
        import yaml

        from ...data.common import repo_root

        for p in (repo_root() / "data/processed/data.yaml", repo_root() / "configs/data.yaml"):
            if p.exists():
                d = yaml.safe_load(p.read_text(encoding="utf-8"))
                if "names" in d:
                    return len(d["names"])
    except Exception:
        pass
    return None


@dataclass
class Timing:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class RunOutput:
    detections: np.ndarray
    timing: Timing = field(default_factory=Timing)


class BaseRunner(abc.ABC):

    name: str = "base"

    def __init__(self, weights: Path, imgsz: int = 640, conf: float = 0.25,
                 iou: float = 0.7, device: str = "cuda", nc: int | None = None,
                 scaleup: bool = True, multi_label: bool = False,
                 resize_to: int | None = None):
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.scaleup = scaleup
        self.multi_label = multi_label
        self.resize_to = resize_to
        self.nc = nc if nc is not None else _num_classes()
        if not self.weights.exists():
            raise FileNotFoundError(f"khong thay trong so: {self.weights}")
        self._load()

    @abc.abstractmethod
    def _load(self) -> None:
        pass

    @abc.abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        pass

    def backend_info(self) -> dict:
        return {
            "runner": self.name,
            "weights": _rel(self.weights),
            "model_size_mb": round(self.weights.stat().st_size / 1e6, 2),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "scaleup": self.scaleup,
            "resize_to": self.resize_to,
            "multi_label": self.multi_label,
        }

    def warmup(self, n: int = 20, img_bgr: np.ndarray | None = None) -> None:
        if img_bgr is None:
            img_bgr = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self.run_array(img_bgr)

    def run_array(self, img_bgr: np.ndarray) -> RunOutput:
        t = Timing()

        t0 = time.perf_counter()
        x, ratio, pad = preprocess(img_bgr, self.imgsz, scaleup=self.scaleup,
                                   resize_to=self.resize_to)
        t.preprocess_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        raw = self.infer(x)
        t.inference_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        det = decode(raw, self.conf, self.iou, nc=self.nc, multi_label=self.multi_label)
        det = scale_boxes(det, ratio, pad, img_bgr.shape[:2])
        t.postprocess_ms = (time.perf_counter() - t0) * 1000

        return RunOutput(det, t)

    def run_file(self, path: Path) -> RunOutput:
        return self.run_array(read_image(path))

    def close(self) -> None:  # noqa: B027
        pass


def build_runner(backend: str, weights: Path, **kw) -> BaseRunner:
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
    suffix = {"pytorch": ".pt", "pt": ".pt", "torch": ".pt",
              "onnx": ".onnx", "onnxruntime": ".onnx", "ort": ".onnx",
              "tensorrt": ".engine", "trt": ".engine", "engine": ".engine"}[backend.lower()]
    return Path(model_dir) / f"best{suffix}"
