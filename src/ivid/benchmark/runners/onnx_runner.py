from __future__ import annotations

import numpy as np

from .base import BaseRunner

PREFERRED = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


class OnnxRunner(BaseRunner):
    name = "onnx"

    def __init__(self, *a, providers: list[str] | None = None,
                 allow_cpu_fallback: bool = True, **kw):
        self._requested_providers = providers
        self._allow_cpu = allow_cpu_fallback
        super().__init__(*a, **kw)

    def _load(self) -> None:
        import onnxruntime as ort

        self.ort = ort
        available = ort.get_available_providers()

        if self._requested_providers:
            chosen = [p for p in self._requested_providers if p in available]
        elif self.device.startswith("cuda"):
            chosen = [p for p in PREFERRED if p in available]
        else:
            chosen = ["CPUExecutionProvider"]

        if not chosen:
            raise RuntimeError(f"không provider nào dùng được. Có sẵn: {available}")

        gpu_eps = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
        self.gpu_available = bool(gpu_eps & set(chosen))
        if not self.gpu_available and self.device.startswith("cuda") and not self._allow_cpu:
            raise RuntimeError(
                f"ONNX Runtime không có GPU provider (chỉ có {available}).\n"
                "Trên Jetson phải cài wheel của jetson-ai-lab:\n"
                "  pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu\n"
                "Chạy với allow_cpu_fallback=True nếu thật sự muốn đo trên CPU."
            )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.weights), sess_options=so, providers=chosen)

        self.providers_active = self.sess.get_providers()
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.gpu_active = bool(gpu_eps & set(self.providers_active))

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(self.output_names, {self.input_name: x})[0]

    def backend_info(self) -> dict:
        info = super().backend_info()
        info.update(
            framework=f"onnxruntime {self.ort.__version__}",
            providers_available=self.ort.get_available_providers(),
            providers_active=self.providers_active,
            running_on_gpu=self.gpu_active,
            precision="fp32",
        )
        if not self.gpu_active:
            info["CANH_BAO"] = (
                "ONNX Runtime đang chạy trên CPU. Số liệu latency của định dạng này "
                "KHÔNG so sánh được với PyTorch/TensorRT chạy trên GPU — phải ghi rõ "
                "trong báo cáo thay vì để người đọc tưởng là ONNX chậm."
            )
        return info
