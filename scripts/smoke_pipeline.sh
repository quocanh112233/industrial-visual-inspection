#!/usr/bin/env bash
# =============================================================================
# IVID — KIEM CHUNG BO DO, TRUOC KHI TRAIN
#
# Chay ca ba runner + latency.py + report.py bang model yolov8n PRETRAINED
# (chua fine-tune) ma smoke_tensorrt.sh da sinh ra.
#
# Detection sinh ra se vo nghia — model COCO nhin anh thep se doan lung tung.
# KHONG SAO: thu can kiem tra o day la BO DO co chay dung khong:
#   - ba runner nap duoc model va tra ve tensor dung hinh dang
#   - tien/hau xu ly dung chung khong lam vo cai nao
#   - latency.py do va tach duoc ba giai doan
#   - report.py sinh duoc bang va bieu do
#
# Phat hien loi o day ton 10 phut. Phat hien sau khi train xong ton mot ngay.
#
#   bash scripts/smoke_pipeline.sh
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE="$ROOT/models/smoke"
DEST="$ROOT/models/_smoke"
OUT="$ROOT/results/smoke_pipeline.json"

log()  { printf '\033[1;34m[ivid]\033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m[loi]\033[0m %s\n' "$*" >&2; }

[[ -d "$ROOT/.venv" ]] && source "$ROOT/.venv/bin/activate"
# Jetson chay locale C -> Python mac dinh ascii cho stdout
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

# --- 1. gom san pham cua smoke test R1 thanh mot "model" gia ---
OPSET="${OPSET:-17}"
mkdir -p "$DEST"
cp -f "$SMOKE/yolov8n.pt"                  "$DEST/best.pt"      2>/dev/null
cp -f "$SMOKE/yolov8n_op${OPSET}.onnx"     "$DEST/best.onnx"    2>/dev/null
cp -f "$SMOKE/yolov8n_op${OPSET}.engine"   "$DEST/best.engine"  2>/dev/null

missing=0
for f in best.pt best.onnx best.engine; do
  if [[ -f "$DEST/$f" ]]; then
    printf '  co  %-13s %s\n' "$f" "$(du -h "$DEST/$f" | cut -f1)"
  else
    printf '  THIEU %s\n' "$f"; missing=1
  fi
done
if [[ $missing -eq 1 ]]; then
  bad "Chay 'bash scripts/smoke_tensorrt.sh' truoc de sinh ra cac file nay."
  exit 1
fi

# --- 2. can co dataset da chuan bi de lay anh do ---
if [[ ! -f "$ROOT/data/processed/data.yaml" ]]; then
  log "Chua co dataset — chay 'make data' truoc"
  exit 1
fi

# --- 3. do nhanh: 1 phien, 30 anh, khong cho on dinh nhiet ---
log "Do thu ba runner (nc=80 vi day la model COCO pretrained)..."
PYTHONPATH="$ROOT/src" python -m ivid.benchmark.latency \
  --models _smoke \
  --backends pytorch onnx tensorrt \
  --sessions 1 --warmup 20 --settle 0 --cooldown 0 \
  --max-images 30 --nc 80 \
  --out results/smoke_pipeline.json
rc=$?

if [[ $rc -ne 0 ]]; then
  bad "latency.py that bai (ma loi $rc)"
  exit $rc
fi

# --- 4. sinh bao cao thu (khong dung mAP vi model chua fine-tune) ---
log "Sinh bao cao thu..."
PYTHONPATH="$ROOT/src" python -m ivid.benchmark.report \
  --input results/smoke_pipeline.json \
  --out results/smoke_pipeline_report.md \
  --images results/smoke_images

echo
log "=============== KET LUAN ==============="
PYTHONPATH="$ROOT/src" python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
runs = d.get("runs", {})
ok = [k for k, v in runs.items() if not v.get("skipped")]
bad = {k: v.get("reason") for k, v in runs.items() if v.get("skipped")}

print(f"  runner chay duoc : {len(ok)}/3  -> {', '.join(x.split('|')[1] for x in ok)}")
for k, why in bad.items():
    print(f"  KHONG chay duoc  : {k.split('|')[1]} — {why}")

for k in ok:
    v = runs[k]
    ms = v["median_stages"]
    chk = v["stage_sum_check"]
    delta = abs(chk["sum_of_parts_p50"] - chk["total_p50"])
    print(f"\n  {k.split('|')[1]:<9} total p50 {v['latency_p50_ms']:7.2f} ms  ({v['fps']:6.1f} FPS)")
    print(f"            pre {ms['preprocess']['p50']:.2f} + infer {ms['inference']['p50']:.2f} "
          f"+ post {ms['postprocess']['p50']:.2f}   (FR-11 lech {delta:.3f} ms)")
    bi = v["backend_info"]
    if bi.get("providers_active"):
        print(f"            ORT providers: {bi['providers_active']}")
    if bi.get("CANH_BAO"):
        print(f"            CANH BAO: {bi['CANH_BAO'][:80]}...")

print()
if len(ok) == 3:
    print("  >>> BO DO HOAT DONG DAY DU. San sang train that.")
else:
    print("  >>> Con runner chua chay duoc — sua truoc khi train.")
PY

log "Chi tiet: results/smoke_pipeline.json va results/smoke_pipeline_report.md"
log "Xoa model gia sau khi xong:  rm -rf models/_smoke"
