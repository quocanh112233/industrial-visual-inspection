from __future__ import annotations

import numpy as np

from .base import BaseRunner


def _nap_cudart():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:                                   # pragma: no cover
        from cuda import cudart
    return cudart


def kiem_tra(ret):
    if not isinstance(ret, tuple):
        ret = (ret,)
    err, gia_tri = ret[0], ret[1:]
    if int(err) != 0:
        raise RuntimeError(f"CUDA loi {int(err)} ({getattr(err, 'name', err)})")
    if not gia_tri:
        return None
    return gia_tri[0] if len(gia_tri) == 1 else gia_tri


class TensorRTRunner(BaseRunner):
    name = "tensorrt"

    def _load(self) -> None:
        import tensorrt as trt

        cudart = _nap_cudart()
        self.trt, self.cudart = trt, cudart

        ver = int(kiem_tra(cudart.cudaRuntimeGetVersion()))
        err, so_gpu = cudart.cudaGetDeviceCount()
        if int(err) == 35:
            raise RuntimeError(
                f"cuda-python mang runtime CUDA {ver // 1000}.{ver % 1000 // 10} nhung "
                "driver tren may cu hon, nen khong loi goi CUDA nao chay duoc.\n"
                'Cai ban khop voi driver:  pip install "cuda-python<13"'
            )
        kiem_tra((err, so_gpu))
        if not so_gpu:
            raise RuntimeError("CUDA khong thay GPU nao — khong chay duoc TensorRT")

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
        self.dptr: dict[str, int] = {}
        self.nbytes: dict[str, int] = {}
        self.host: dict[str, np.ndarray] = {}
        self.io_spec: list[dict] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.ctx.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                shape = tuple(1 if d < 0 else d for d in shape)
                if mode == trt.TensorIOMode.INPUT:
                    self.ctx.set_input_shape(name, shape)
                    shape = tuple(self.ctx.get_tensor_shape(name))
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * np_dtype.itemsize

            self.dptr[name] = int(kiem_tra(cudart.cudaMalloc(nbytes)))
            self.nbytes[name] = nbytes
            self.host[name] = np.empty(shape, dtype=np_dtype)
            self.ctx.set_tensor_address(name, self.dptr[name])

            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)
            self.io_spec.append({"name": name, "io": mode.name,
                                 "shape": list(shape), "dtype": np_dtype.name})

        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError(f"mong doi 1 input / 1 output, engine co {self.io_spec}")

        self._in, self._out = self.inputs[0], self.outputs[0]
        exp = self.host[self._in].shape
        if tuple(exp)[-2:] != (self.imgsz, self.imgsz):
            raise ValueError(
                f"engine nhan {tuple(exp)} nhung benchmark dang dung imgsz={self.imgsz}. "
                "Export lai ONNX voi dung imgsz roi build lai engine."
            )

        self.stream = int(kiem_tra(cudart.cudaStreamCreate()))

    def infer(self, x: np.ndarray) -> np.ndarray:
        cudart = self.cudart
        H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

        vao = np.ascontiguousarray(x, dtype=self.host[self._in].dtype)
        kiem_tra(cudart.cudaMemcpyAsync(
            self.dptr[self._in], vao.ctypes.data, self.nbytes[self._in], H2D, self.stream))
        self.ctx.execute_async_v3(stream_handle=self.stream)
        ra = self.host[self._out]
        kiem_tra(cudart.cudaMemcpyAsync(
            ra.ctypes.data, self.dptr[self._out], self.nbytes[self._out], D2H, self.stream))
        kiem_tra(cudart.cudaStreamSynchronize(self.stream))

        return np.array(ra, dtype=np.float32)

    def backend_info(self) -> dict:
        info = super().backend_info()
        dtypes = {t["dtype"] for t in self.io_spec}
        props = kiem_tra(self.cudart.cudaGetDeviceProperties(0))
        ten = getattr(props, "name", b"")
        info.update(
            framework=f"tensorrt {self.trt.__version__}",
            gpu=ten.decode(errors="replace").strip("\x00") if isinstance(ten, bytes) else str(ten),
            cap_phat="cuda-python (cudart), khong dung torch",
            io=self.io_spec,
            precision="fp16 (theo configs/export.yaml)",
            io_dtypes=sorted(dtypes),
        )
        return info

    def close(self) -> None:
        cudart = getattr(self, "cudart", None)
        if cudart is None:
            return
        for ptr in getattr(self, "dptr", {}).values():
            cudart.cudaFree(ptr)
        if getattr(self, "stream", None):
            cudart.cudaStreamDestroy(self.stream)
        self.dptr, self.host, self.stream = {}, {}, None
        self.ctx = None
        self.engine = None
