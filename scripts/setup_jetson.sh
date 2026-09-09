#!/usr/bin/env bash
# =============================================================================
# IVID - Dung moi truong Python tren Jetson Orin Nano (JetPack 6.2 / L4T R36.5.2)
#
#   bash scripts/setup_jetson.sh
#
# TAI SAO PHUC TAP HON "pip install -r requirements.txt":
#   Tren Jetson, torch va tensorrt la ban NVIDIA build rieng cho aarch64+CUDA,
#   cai san trong he thong. Neu chay 'pip install ultralytics' binh thuong,
#   pip se keo torch tu PyPI ve va GHI DE ban CUDA -> torch.cuda.is_available()
#   thanh False, mat toan bo GPU. Script nay tranh dieu do.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"

warn() { printf '\033[1;33m[canh bao]\033[0m %s\n' "$*"; }
log()  { printf '\033[1;34m[ivid]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[loi]\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -m)" == "aarch64" ]] || die "Script nay chi chay tren Jetson (aarch64). May hien tai: $(uname -m)"

# --- 1. venv KE THUA package he thong (de thay torch CUDA + tensorrt) --------
if [[ ! -d "$VENV" ]]; then
  log "Tao venv ke thua system-site-packages (bat buoc de thay torch/tensorrt cua NVIDIA)..."
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -qU pip setuptools wheel

log "Kiem tra torch/tensorrt nhin thay tu venv:"
python - <<'PY'
import importlib, sys
ok = True
try:
    import torch
    print(f"  torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        ok = False; print("  [loi] torch khong thay CUDA")
except Exception as e:
    ok = False; print("  [loi] khong import duoc torch:", e)
try:
    import tensorrt as trt
    print(f"  tensorrt   {trt.__version__}")
except Exception as e:
    ok = False; print("  [loi] khong import duoc tensorrt:", e)
try:
    import torchvision
    print(f"  torchvision {torchvision.__version__}")
except Exception as e:
    print("  [canh bao] chua co torchvision — ultralytics can no:", e)
sys.exit(0 if ok else 1)
PY

# --- 2. onnxruntime-gpu ban Jetson ------------------------------------------
# Ban tren PyPI KHONG co CUDA/TensorRT EP cho aarch64 -> cot "ONNX Runtime"
# trong bang benchmark se do nham tren CPU. Phai lay wheel cua jetson-ai-lab.
#
# BAY: 'onnxruntime' (CPU) va 'onnxruntime-gpu' cai DE LEN NHAU — ca hai cung
# giai nen vao thu muc  site-packages/onnxruntime/. Ai cai sau thi de len file
# .so cua nguoi truoc. Neu ban CPU cai sau, import se ra CPU du pip van liet ke
# ca hai. Ngoai ra 'pip uninstall' chay TRONG venv KHONG xoa duoc goi nam o
# ~/.local (user-site) — phai dung python he thong.
log "Don dep onnxruntime o user-site (~/.local) bang python he thong..."
/usr/bin/python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

# Ghim numpy TRUOC khi cai onnxruntime-gpu. Khong ghim thi pip tu chon ban moi
# nhat, va moi lan chay script lai ra mot phien ban khac — mat tinh tai lap.
#
# Ban duoi day DA DUOC KIEM CHUNG tren Jetson (scripts/check_stack.sh: torch
# 2.10 <-> numpy, cv2, ORT 1.24 deu qua). L4T cai san numpy 1.26.4, nhung 2.2.6
# van lam viec voi ca torch, opencv va ultralytics — nen khong can ha xuong.
# Doi so nay thi PHAI chay lai check_stack.sh de xac nhan.
NUMPY_PIN="${NUMPY_PIN:-2.2.6}"
log "Ghim numpy==$NUMPY_PIN (ban da kiem chung tren Jetson)..."
python -m pip install "numpy==$NUMPY_PIN"

log "Cai onnxruntime-gpu vao venv cho JetPack 6 / CUDA 12.6..."
python -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
python -m pip install --force-reinstall --no-cache-dir \
  --index-url "$JETSON_INDEX" onnxruntime-gpu \
  || die "Khong tai duoc onnxruntime-gpu tu $JETSON_INDEX (kiem tra mang)."

# --- 3. ultralytics KHONG keo theo torch ------------------------------------
log "Cai ultralytics (--no-deps) + cac phu thuoc an toan..."
python -m pip install --no-deps ultralytics
python -m pip install \
  opencv-python-headless pillow pyyaml requests scipy \
  matplotlib pandas psutil py-cpuinfo tqdm ultralytics-thop

# --- 4. phan con lai cua du an ----------------------------------------------
log "Cai phu thuoc cua IVID..."
python -m pip install fastapi "uvicorn[standard]" python-multipart pydantic \
  onnx pytest ruff jetson-stats || true

# --- 5. kiem tra lai ---------------------------------------------------------
log "Kiem tra cuoi:"
python - <<'PY'
import torch, tensorrt as trt, onnxruntime as ort
print(f"  torch          {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"  tensorrt       {trt.__version__}")
print(f"  onnxruntime    {ort.__version__}")
print(f"  ORT nap tu     {ort.__file__}")
prov = ort.get_available_providers()
print(f"  ORT providers  {prov}")
if not any(p in prov for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")):
    print("""  [LOI] ORT van chi co CPU.
        Nguyen nhan hay gap: ban CPU va ban GPU chong len nhau. Kiem tra:
          ls -d ~/.local/lib/python3.10/site-packages/onnxruntime*
          ls -d .venv/lib/python3.10/site-packages/onnxruntime*
        Neu con ban CPU o user-site, xoa thu cong roi chay lai script nay.""")
else:
    print("  [ok] ORT co GPU provider")
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"  ultralytics    {ultralytics.__version__}")
except Exception as e:
    print("  [loi] ultralytics:", e)
PY

log "Kiem tra tuong thich giua cac thu vien:"
bash "$ROOT/scripts/check_stack.sh" || warn "Stack co van de — xem o tren truoc khi chay tiep"

log "Xong. Kich hoat bang:  source $VENV/bin/activate"
