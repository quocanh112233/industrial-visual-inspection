from __future__ import annotations

import numpy as np

from .base import BaseRunner


class PyTorchRunner(BaseRunner):
    name = "pytorch"

    def _load(self) -> None:
        import torch
        from ultralytics import YOLO

        self.torch = torch
        wanted = self.device
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("yeu cau CUDA nhung torch khong thay GPU")
        self._dev = torch.device(wanted)

        y = YOLO(str(self.weights))
        self.names = y.names
        self.model = y.model.float().to(self._dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def infer(self, x: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(x).to(self._dev, non_blocking=True)
        with self.torch.inference_mode():
            y = self.model(t)
        y = y[0] if isinstance(y, (list, tuple)) else y
        if self._dev.type == "cuda":
            self.torch.cuda.synchronize()
        return y.detach().cpu().numpy()

    def backend_info(self) -> dict:
        info = super().backend_info()
        info.update(
            framework=f"torch {self.torch.__version__}",
            device=str(self._dev),
            gpu=self.torch.cuda.get_device_name(0) if self._dev.type == "cuda" else "cpu",
            precision="fp32",
        )
        return info
