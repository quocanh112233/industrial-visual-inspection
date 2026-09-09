#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -d "$ROOT/.venv" ]] && source "$ROOT/.venv/bin/activate"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

python - <<'PY'
import sys, traceback
ok = True
def check(name, fn):
    global ok
    try:
        print(f"  [ok]   {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"  [LOI]  {name}: {type(e).__name__}: {e}")

import numpy as np
print(f"\n=== phiên bản ===")
print(f"  numpy        {np.__version__}  ({np.__file__})")
for m in ("torch", "torchvision", "cv2", "onnxruntime", "tensorrt", "ultralytics",
          "onnx", "matplotlib", "scipy", "pandas"):
    try:
        mod = __import__(m)
        print(f"  {m:<12} {getattr(mod,'__version__','?')}")
    except Exception as e:
        ok = False
        print(f"  {m:<12} KHONG IMPORT DUOC: {e}")

print(f"\n=== phép thử thật (đây là phần hay hỏng) ===")
import torch
check("torch.from_numpy", lambda: tuple(torch.from_numpy(np.zeros((2,3), np.float32)).shape))
check("torch -> numpy",   lambda: torch.zeros(3).numpy().shape)
check("torch cuda",       lambda: (torch.cuda.is_available(), torch.cuda.get_device_name(0)))
check("torch cuda tính",  lambda: float((torch.ones(1000, device='cuda')*2).sum().item()))

import cv2
check("cv2 resize",   lambda: cv2.resize(np.zeros((10,10,3), np.uint8), (20,20)).shape)
check("cv2 imencode", lambda: cv2.imencode(".png", np.zeros((8,8,3), np.uint8))[0])

import onnxruntime as ort
check("ORT providers", lambda: ort.get_available_providers())

import tensorrt as trt
check("trt logger", lambda: type(trt.Logger(trt.Logger.ERROR)).__name__)

check("ultralytics YOLO", lambda: __import__("ultralytics").YOLO.__name__)

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot(np.arange(5), np.arange(5) ** 2)
    import io
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return f"vẽ được biểu đồ ({buf.tell()} bytes)"
check("matplotlib ve hinh", _mpl)

check("scipy", lambda: __import__("scipy.ndimage", fromlist=["x"]).maximum_filter(
    np.zeros((4, 4)), size=2).shape)
check("pandas", lambda: __import__("pandas").DataFrame({"a": np.arange(3)}).sum()["a"])

print()
if ok:
    print("=> Toàn bộ stack làm việc với nhau được.")
else:
    print("=> CÓ VẤN ĐỀ. Xem các dòng [LỖI] ở trên.")
sys.exit(0 if ok else 1)
PY
