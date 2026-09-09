"""Nap mot TensorRT engine, chay inference thu va do thoi gian.

Dung TensorRT 10.x API (get_tensor_name / execute_async_v3) — API cu
(get_binding_*/execute_v2) da bi bo tu TRT 10, va Jetson nay dang chay TRT 10.3.

Bo nho GPU duoc cap phat bang torch thay vi pycuda: Jetson da co san torch ban
CUDA cua NVIDIA, nen tranh duoc them mot phu thuoc de vo phien ban.

    python -m ivid.export.trt_infer_check models/smoke/yolov8n_op17.engine
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def load_engine(path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Khong deserialize duoc engine: {path}")
    return engine


def run(engine_path: Path, iters: int = 100, warmup: int = 20) -> dict:
    import numpy as np
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch khong thay CUDA — khong the chay TensorRT qua torch buffer")

    engine = load_engine(engine_path)
    ctx = engine.create_execution_context()

    tensors, buffers = [], {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(ctx.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        if any(d < 0 for d in shape):  # dynamic axis -> chot batch 1
            shape = tuple(1 if d < 0 else d for d in shape)
            if mode == trt.TensorIOMode.INPUT:
                ctx.set_input_shape(name, shape)
        buf = torch.empty(shape, dtype=getattr(torch, np.dtype(dtype).name), device="cuda")
        buffers[name] = buf
        ctx.set_tensor_address(name, buf.data_ptr())
        tensors.append(
            {"name": name, "io": mode.name, "shape": list(shape), "dtype": np.dtype(dtype).name}
        )

    # TensorRT canh bao neu dung default stream (no phai chen them
    # cudaStreamSynchronize). Tao stream rieng de tranh chi phi do.
    torch_stream = torch.cuda.Stream()
    stream = torch_stream.cuda_stream

    for t in tensors:
        if t["io"] == "INPUT":
            buffers[t["name"]].normal_()

    with torch.cuda.stream(torch_stream):
        for _ in range(warmup):
            ctx.execute_async_v3(stream_handle=stream)
    torch_stream.synchronize()

    lat = []
    with torch.cuda.stream(torch_stream):
        for _ in range(iters):
            t0 = time.perf_counter()
            ctx.execute_async_v3(stream_handle=stream)
            torch_stream.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)

    lat.sort()
    return {
        "engine": str(engine_path),
        "engine_size_mb": round(engine_path.stat().st_size / 1e6, 2),
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "tensors": tensors,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "p50": round(statistics.median(lat), 3),
            "p95": round(lat[int(0.95 * len(lat)) - 1], 3),
            "min": round(lat[0], 3),
            "max": round(lat[-1], 3),
            "mean": round(statistics.fmean(lat), 3),
        },
        "fps": round(1000.0 / statistics.median(lat), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine", type=Path)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", type=Path, help="ghi ket qua ra file JSON")
    a = ap.parse_args()

    if not a.engine.exists():
        print(f"[loi] khong thay engine: {a.engine}", file=sys.stderr)
        return 1

    res = run(a.engine, a.iters, a.warmup)
    print(json.dumps(res, indent=2))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2))
        print(f"\n-> da ghi {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
