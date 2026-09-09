"""FR-17 — Chon runtime qua bien moi truong.

    IVID_BACKEND   pytorch | onnx | tensorrt | mock   (mac dinh: tensorrt)
    IVID_MODEL     ten thu muc trong models/          (mac dinh: yolov8n)
    IVID_IMGSZ     kich thuoc dau vao                 (mac dinh: 640)
    IVID_CONF      nguong tin cay                     (mac dinh: 0.25)
    IVID_IOU       nguong IoU cua NMS                 (mac dinh: 0.7)
    IVID_DEVICE    cuda | cpu                         (mac dinh: cuda)
    IVID_MAX_UPLOAD_MB                                (mac dinh: 10)

Model duoc nap LUOI va loi nap duoc giu lai thay vi lam sap tien trinh: mot
dich vu khong nap duoc model van phai tra loi /health de he thong giam sat biet
chuyen gi dang xay ra. Sap ngay luc khoi dong thi chi thay container restart lien tuc.
"""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.common import repo_root


@dataclass
class Settings:
    backend: str = os.getenv("IVID_BACKEND", "tensorrt").lower()
    model: str = os.getenv("IVID_MODEL", "yolov8n")
    imgsz: int = int(os.getenv("IVID_IMGSZ", "640"))
    conf: float = float(os.getenv("IVID_CONF", "0.25"))
    iou: float = float(os.getenv("IVID_IOU", "0.7"))
    device: str = os.getenv("IVID_DEVICE", "cuda")
    max_upload_mb: float = float(os.getenv("IVID_MAX_UPLOAD_MB", "10"))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


class MockRunner:
    """Runner gia cho test va cho demo khi chua co trong so.

    Khong dung de do hieu nang. Ton tai de test API chay duoc tren CI, noi
    khong co GPU lan trong so model.
    """

    name = "mock"

    def __init__(self, imgsz: int = 640, **_):
        self.imgsz = imgsz
        self.weights = Path("mock")

    def run_array(self, img_bgr: np.ndarray):
        from ..benchmark.runners.base import RunOutput, Timing

        h, w = img_bgr.shape[:2]
        det = np.array([[w * 0.2, h * 0.2, w * 0.6, h * 0.7, 0.87, 1]], dtype=np.float32)
        return RunOutput(det, Timing(0.5, 1.0, 0.2))

    def warmup(self, n: int = 1) -> None:
        pass

    def backend_info(self) -> dict:
        return {"runner": "mock", "weights": "mock", "model_size_mb": 0.0,
                "imgsz": self.imgsz, "precision": "n/a"}

    def close(self) -> None:
        pass


class BackendHolder:
    """Giu runner + trang thai nap, an toan khi nhieu request goi cung luc."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.runner = None
        self.error: str | None = None
        self.names: dict[int, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ nap
    def load(self) -> None:
        with self._lock:
            if self.runner is not None:
                return
            s = self.settings
            try:
                self.names = self._load_names()
                if s.backend == "mock":
                    self.runner = MockRunner(imgsz=s.imgsz)
                else:
                    from ..benchmark.runners.base import build_runner, weights_for

                    w = weights_for(repo_root() / "models" / s.model, s.backend)
                    self.runner = build_runner(s.backend, w, imgsz=s.imgsz,
                                               conf=s.conf, iou=s.iou, device=s.device)
                    self.runner.warmup(5)
                self.error = None
            except Exception as e:
                self.runner = None
                self.error = f"{type(e).__name__}: {e}"

    def _load_names(self) -> dict[int, str]:
        import yaml

        for p in (repo_root() / "data/processed/data.yaml", repo_root() / "configs/data.yaml"):
            if p.exists():
                d = yaml.safe_load(p.read_text())
                if "names" in d:
                    return {int(k): v for k, v in d["names"].items()}
        return {}

    # -------------------------------------------------------------- thong tin
    @property
    def loaded(self) -> bool:
        return self.runner is not None

    def class_name(self, cid: int) -> str:
        return self.names.get(cid, str(cid))

    def model_version(self) -> str:
        """Van tat nhan dang model dang phuc vu — kich thuoc + thoi diem sua file."""
        s = self.settings
        if s.backend == "mock":
            return "mock"
        try:
            from ..benchmark.runners.base import weights_for

            w = weights_for(repo_root() / "models" / s.model, s.backend)
            st = w.stat()
            return f"{w.name}:{st.st_size}:{int(st.st_mtime)}"
        except Exception:
            return "?"

    def info(self) -> dict:
        s = self.settings
        d = {"backend": s.backend, "model": s.model, "imgsz": s.imgsz,
             "device": s.device, "loaded": self.loaded,
             "model_version": self.model_version(), "error": self.error}
        if self.runner is not None:
            with contextlib.suppress(Exception):
                d["backend_info"] = self.runner.backend_info()
        return d

    def close(self) -> None:
        if self.runner is not None:
            with contextlib.suppress(Exception):
                self.runner.close()
            self.runner = None
