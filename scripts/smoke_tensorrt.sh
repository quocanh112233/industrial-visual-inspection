#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/models/smoke"
RES="$ROOT/results"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
OPSETS="12 17"
IMGSZ=640

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opsets) OPSETS="$2"; shift 2 ;;
    --imgsz)  IMGSZ="$2";  shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Tham so la: $1" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[ivid]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  [ok]\033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m  [that bai]\033[0m %s\n' "$*"; }

mkdir -p "$OUT" "$RES"
[[ -d "$ROOT/.venv" ]] && source "$ROOT/.venv/bin/activate"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

log "Phien ban moi truong"
python - "$RES/r1_env.json" <<'PY'
import json, platform, subprocess, sys
from pathlib import Path

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""

info = {
    "python": platform.python_version(),
    "arch": platform.machine(),
    "l4t": sh("head -1 /etc/nv_tegra_release"),
    "jetpack": (sh("apt-cache policy nvidia-jetpack 2>/dev/null | awk '/Installed/{print $2}'")
                or sh("apt-cache show nvidia-jetpack 2>/dev/null | awk '/^Version:/{print $2; exit}'")
                or "?"),
    "l4t_core": sh("dpkg-query --showformat='${Version}' --show nvidia-l4t-core 2>/dev/null"),
    "cuda": sh("/usr/local/cuda/bin/nvcc --version | tail -2 | head -1"),
    "nvpmodel_id": sh("awk -F: '{print $2+0}' /var/lib/nvpmodel/status 2>/dev/null"),
    "nvpmodel_name": sh("grep -oP '(?<=^< POWER_MODEL ID=)[0-9]+ NAME=\\S+' /etc/nvpmodel.conf 2>/dev/null | "
                        "awk -v id=\"$(awk -F: '{print $2+0}' /var/lib/nvpmodel/status 2>/dev/null)\" "
                        "'$1==id{sub(/NAME=/,\"\",$2); print $2}'"),
    "power_modes": sh("grep -oP '(?<=^< POWER_MODEL ID=).*(?= >)' /etc/nvpmodel.conf 2>/dev/null | tr '\\n' '|'"),
    "temp_c": sh("for z in /sys/devices/virtual/thermal/thermal_zone*; do "
                 "[ -f $z/type ] && echo \"$(cat $z/type)=$(( $(cat $z/temp)/1000 ))\"; done | tr '\\n' ' '"),
}
for mod in ("torch", "tensorrt", "onnx", "onnxruntime", "ultralytics"):
    try:
        m = __import__(mod)
        info[mod] = getattr(m, "__version__", "?")
    except Exception as e:
        info[mod] = f"KHONG CO ({type(e).__name__})"
try:
    import torch; info["torch_cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
except Exception:
    pass
try:
    import onnxruntime as ort; info["ort_providers"] = ort.get_available_providers()
except Exception:
    pass

for k, v in info.items():
    print(f"  {k:14s} {v}")
Path(sys.argv[1]).write_text(json.dumps(info, indent=2))
PY

command -v "$TRTEXEC" >/dev/null 2>&1 || [[ -x "$TRTEXEC" ]] || {
  bad "khong thay trtexec tai $TRTEXEC — dat bien TRTEXEC=/duong/dan/trtexec"; exit 1; }

cd "$OUT"
if [[ ! -f yolov8n.pt ]]; then
  log "Tai trong so pretrained yolov8n.pt"
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" >/dev/null 2>&1 \
    || curl -fL -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
  [[ -f "$ROOT/yolov8n.pt" ]] && mv "$ROOT/yolov8n.pt" .
fi
ls -lh yolov8n.pt

RESULTS_JSON="$RES/r1_smoke.json"
echo '{"imgsz": '"$IMGSZ"', "runs": []}' > "$RESULTS_JSON"

record() {
  python - "$RESULTS_JSON" "$1" "$2" "$3" "$4" <<'PY'
import json, sys
p, opset, stage, status, detail = sys.argv[1:6]
d = json.load(open(p))
d["runs"].append({"opset": int(opset), "stage": stage, "status": status, "detail": detail})
json.dump(d, open(p, "w"), indent=2)
PY
}

for OP in $OPSETS; do
  echo
  log "=============== OPSET $OP ==============="
  ONNX="$OUT/yolov8n_op${OP}.onnx"
  ENG="$OUT/yolov8n_op${OP}.engine"

  log "Export ONNX (opset $OP, batch 1, imgsz $IMGSZ)"
  if python - "$OP" "$IMGSZ" "$ONNX" <<'PY'
import shutil, sys
from pathlib import Path
from ultralytics import YOLO
opset, imgsz, dst = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
m = YOLO("yolov8n.pt")
out = m.export(format="onnx", opset=opset, imgsz=imgsz, batch=1, dynamic=False, simplify=True)
shutil.move(str(out), dst)
import onnx
onnx.checker.check_model(str(dst))
print(f"  onnx.checker PASS -> {dst} ({dst.stat().st_size/1e6:.1f} MB)")
PY
  then ok "ONNX opset $OP"; record "$OP" onnx ok "$(du -h "$ONNX" | cut -f1)"
  else bad "ONNX opset $OP"; record "$OP" onnx fail "export hoac onnx.checker loi"; continue
  fi

  log "Build TensorRT engine FP16 (co the mat 2-5 phut)"
  LOG="$OUT/trtexec_op${OP}.log"
  if "$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENG" --fp16 \
        --memPoolSize=workspace:2048MiB --skipInference > "$LOG" 2>&1
  then ok "engine opset $OP -> $(du -h "$ENG" | cut -f1)"; record "$OP" engine ok "$(du -h "$ENG" | cut -f1)"
  else bad "trtexec that bai — xem $LOG"; tail -15 "$LOG"; record "$OP" engine fail "xem $LOG"; continue
  fi

  log "Nap engine va chay 100 lan inference"
  if PYTHONPATH="$ROOT/src" python -m ivid.export.trt_infer_check "$ENG" \
        --json "$RES/r1_trt_op${OP}.json"
  then ok "inference opset $OP"; record "$OP" infer ok "results/r1_trt_op${OP}.json"
  else bad "inference opset $OP"; record "$OP" infer fail "xem log tren"
  fi
done

echo
log "=============== TONG KET R1 ==============="
python - "$RESULTS_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["runs"]
if not rows:
    print("  khong co ket qua"); raise SystemExit(1)
print(f"  {'opset':<8}{'stage':<10}{'status':<8}detail")
for r in rows:
    mark = "OK " if r["status"] == "ok" else "FAIL"
    print(f"  {r['opset']:<8}{r['stage']:<10}{mark:<8}{r['detail']}")
good = {r["opset"] for r in rows if r["stage"] == "infer" and r["status"] == "ok"}
print()
if good:
    print(f"  >>> OPSET DUNG DUOC: {sorted(good)}  -> chot vao configs/export.yaml")
else:
    print("  >>> KHONG opset nao qua duoc. R1 da xay ra — bao Claude kem log truoc khi train.")
PY
log "Chi tiet: $RESULTS_JSON  va  $RES/r1_env.json"
