#!/usr/bin/env bash
# Kiem tra cac thu vien co lam viec duoc VOI NHAU khong.
# Import duoc tung cai khong co nghia la chung tuong thich: numpy 2 va mot thu
# vien build voi numpy 1 se import binh thuong roi no ngay khi truyen mang.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -d "$ROOT/.venv" ]] && source "$ROOT/.venv/bin/activate"

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
print(f"\n=== phien ban ===")
print(f"  numpy        {np.__version__}  ({np.__file__})")
for m in ("torch", "torchvision", "cv2", "onnxruntime", "tensorrt", "ultralytics", "onnx"):
    try:
        mod = __import__(m)
        print(f"  {m:<12} {getattr(mod,'__version__','?')}")
    except Exception as e:
        ok = False
        print(f"  {m:<12} KHONG IMPORT DUOC: {e}")

print(f"\n=== phep thu that (day la phan hay hong) ===")
import torch
check("torch.from_numpy", lambda: tuple(torch.from_numpy(np.zeros((2,3), np.float32)).shape))
check("torch -> numpy",   lambda: torch.zeros(3).numpy().shape)
check("torch cuda",       lambda: (torch.cuda.is_available(), torch.cuda.get_device_name(0)))
check("torch cuda tinh",  lambda: float((torch.ones(1000, device='cuda')*2).sum().item()))

import cv2
check("cv2 resize",   lambda: cv2.resize(np.zeros((10,10,3), np.uint8), (20,20)).shape)
check("cv2 imencode", lambda: cv2.imencode(".png", np.zeros((8,8,3), np.uint8))[0])

import onnxruntime as ort
check("ORT providers", lambda: ort.get_available_providers())

import tensorrt as trt
check("trt logger", lambda: type(trt.Logger(trt.Logger.ERROR)).__name__)

check("ultralytics YOLO", lambda: __import__("ultralytics").YOLO.__name__)

print()
if ok:
    print("=> Toan bo stack lam viec voi nhau duoc.")
else:
    print("=> CO VAN DE. Xem cac dong [LOI] o tren.")
sys.exit(0 if ok else 1)
PY
