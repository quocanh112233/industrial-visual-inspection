#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"

warn() { printf '\033[1;33m[cảnh báo]\033[0m %s\n' "$*"; }
log()  { printf '\033[1;34m[ivid]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[lỗi]\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -m)" == "aarch64" ]] || die "Script này chỉ chạy trên Jetson (aarch64). Máy hiện tại: $(uname -m)"

if [[ ! -d "$VENV" ]]; then
  log "Tạo venv kế thừa system-site-packages (bắt buộc để thay torch/tensorrt của NVIDIA)..."
  python3 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -qU pip setuptools wheel

log "Kiểm tra torch/tensorrt nhìn thấy từ venv:"
python - <<'PY'
import importlib, sys
ok = True
try:
    import torch
    print(f"  torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        ok = False; print("  [lỗi] torch không thấy CUDA")
except Exception as e:
    ok = False; print("  [lỗi] không import được torch:", e)
try:
    import tensorrt as trt
    print(f"  tensorrt   {trt.__version__}")
except Exception as e:
    ok = False; print("  [lỗi] không import được tensorrt:", e)
try:
    import torchvision
    print(f"  torchvision {torchvision.__version__}")
except Exception as e:
    print("  [cảnh báo] chưa có torchvision — ultralytics cần nó:", e)
sys.exit(0 if ok else 1)
PY

log "Dọn dẹp onnxruntime ở user-site (~/.local) bằng python hệ thống..."
/usr/bin/python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

NUMPY_PIN="${NUMPY_PIN:-1.26.4}"
log "Ghim numpy==$NUMPY_PIN (bản mà L4T build các thư viện khác cùng)..."
python -m pip install "numpy==$NUMPY_PIN"

log "Cài onnxruntime-gpu vào venv cho JetPack 6 / CUDA 12.6..."
python -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
python -m pip install --force-reinstall --no-cache-dir \
  --index-url "$JETSON_INDEX" onnxruntime-gpu \
  || die "Không tải được onnxruntime-gpu từ $JETSON_INDEX (kiểm tra mạng)."

log "Cài ultralytics (--no-deps) + các phụ thuộc an toàn..."
python -m pip install "cuda-python<13"

python -m pip install --no-deps ultralytics
python -m pip install \
  opencv-python-headless pillow pyyaml requests scipy \
  matplotlib pandas psutil py-cpuinfo tqdm ultralytics-thop

log "Cài phụ thuộc của IVID..."
python -m pip install fastapi "uvicorn[standard]" python-multipart pydantic \
  onnx pytest ruff jetson-stats || true

log "Kiểm tra cuối:"
python - <<'PY'
import torch, tensorrt as trt, onnxruntime as ort
print(f"  torch          {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"  tensorrt       {trt.__version__}")
print(f"  onnxruntime    {ort.__version__}")
print(f"  ORT nạp từ     {ort.__file__}")
prov = ort.get_available_providers()
print(f"  ORT providers  {prov}")
if not any(p in prov for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")):
    print("""  [LỖI] ORT vẫn chỉ có CPU.
        Nguyên nhân hay gặp: bản CPU và bản GPU chồng lên nhau. Kiểm tra:
          ls -d ~/.local/lib/python3.10/site-packages/onnxruntime*
          ls -d .venv/lib/python3.10/site-packages/onnxruntime*
        Nếu còn bản CPU ở user-site, xoá thủ công rồi chạy lại script này.""")
else:
    print("  [ok] ORT có GPU provider")
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"  ultralytics    {ultralytics.__version__}")
except Exception as e:
    print("  [lỗi] ultralytics:", e)
PY

log "Kiểm tra tương thích giữa các thư viện:"
bash "$ROOT/scripts/check_stack.sh" || warn "Stack có vấn đề — xem ở trên trước khi chạy tiếp"

log "Xong. Kích hoạt bằng:  source $VENV/bin/activate"
