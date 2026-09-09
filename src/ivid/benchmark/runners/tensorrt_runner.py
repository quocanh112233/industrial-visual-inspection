"""Runner TensorRT — dung API TensorRT 10.x.

Cap phat bo nho GPU bang torch thay vi pycuda: Jetson da co san torch ban CUDA
cua NVIDIA, them pycuda chi tao them mot phu thuoc de vo phien ban.

Luu y ve stream: TensorRT canh bao neu enqueue tren default stream (no phai
chen them cudaStreamSynchronize). Runner dung stream rieng.
"""
from __future__ import annotations

import numpy as np

from .base import BaseRunner


class TensorRTRunner(BaseRunner):
    name = "tensorrt"

    def _load(self) -> None:
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch khong thay CUDA — khong chay duoc TensorRT qua torch buffer")

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.weights.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"khong nap duoc engine {self.weights}.\n"
                "Engine TensorRT gan voi phan cung + phien ban thu vien: neu no duoc "
                "build o may khac hoac o ban TensorRT khac thi phai build lai tai cho."
            )
        self.ctx = self.engine.create_execution_context()

        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.buffers: dict[str, torch.Tensor] = {}
        self.io_spec: list[dict] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.ctx.get_tensor_shape(name))
            if any(d < 0 for d in shape):                    # truc dong -> chot batch 1
                shape = tuple(1 if d < 0 else d for d in shape)
                if mode == trt.TensorIOMode.INPUT:
                    self.ctx.set_input_shape(name, shape)
                    shape = tuple(self.ctx.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            buf = torch.empty(shape, dtype=getattr(torch, np.dtype(np_dtype).name), device="cuda")
            self.buffers[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)
            self.io_spec.append({"name": name, "io": mode.name,
                                 "shape": list(shape), "dtype": np.dtype(np_dtype).name})

        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError(f"mong doi 1 input / 1 output, engine co {self.io_spec}")

        self._in, self._out = self.inputs[0], self.outputs[0]
        exp = self.buffers[self._in].shape
        if tuple(exp)[-2:] != (self.imgsz, self.imgsz):
            raise ValueError(
                f"engine nhan {tuple(exp)} nhung benchmark dang dung imgsz={self.imgsz}. "
                "Export lai ONNX voi dung imgsz roi build lai engine."
            )

        self.stream = torch.cuda.Stream()

    def infer(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        buf_in = self.buffers[self._in]
        src = torch.from_numpy(np.ascontiguousarray(x))
        with torch.cuda.stream(self.stream):
            buf_in.copy_(src.to(buf_in.dtype), non_blocking=True)
            self.ctx.execute_async_v3(stream_handle=self.stream.cuda_stream)
            out = self.buffers[self._out].clone()
        self.stream.synchronize()
        return out.float().cpu().numpy()

    def backend_info(self) -> dict:
        info = super().backend_info()
        dtypes = {t["dtype"] for t in self.io_spec}
        info.update(
            framework=f"tensorrt {self.trt.__version__}",
            gpu=self.torch.cuda.get_device_name(0),
            io=self.io_spec,
            # engine FP16 van co the co IO la FP32; precision that nam trong engine.
            # Ghi lai cach build de khong phai doan.
            precision="fp16 (theo configs/export.yaml)",
            io_dtypes=sorted(dtypes),
        )
        return info

    def close(self) -> None:
        self.buffers.clear()
        self.ctx = None
        self.engine = None
