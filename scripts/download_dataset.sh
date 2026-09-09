#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
DEST="$RAW/NEU-DET"
CACHE="$RAW/.cache"

SOURCE="github"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --force)  FORCE=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Tham so la: $1" >&2; exit 2 ;;
  esac
done

GH_TARBALL="https://codeload.github.com/Marfbin/NEU-DET-with-yolov8/tar.gz/refs/heads/main"
KAGGLE_SLUG="ousmanesangary/neu-det"

log()  { printf '\033[1;34m[ivid]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[canh bao]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[loi]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ -d "$DEST" && $FORCE -eq 0 ]]; then
  log "Da co $DEST — bo qua buoc tai. Dung --force de tai lai."
else
  [[ $FORCE -eq 1 ]] && rm -rf "$DEST"
  mkdir -p "$CACHE"

  case "$SOURCE" in
  github)
    TARBALL="$CACHE/neu-det-github.tar.gz"
    if [[ ! -s "$TARBALL" ]]; then
      log "Tai tu GitHub mirror (~73 MB, khong can dang nhap)..."
      curl -fL --retry 3 --retry-delay 2 -o "$TARBALL.part" "$GH_TARBALL" \
        || die "Tai that bai. Kiem tra mang, hoac dung --source kaggle."
      mv "$TARBALL.part" "$TARBALL"
    else
      log "Dung tarball da cache: $TARBALL"
    fi

    log "Giai nen (chi lay thu muc dataset)..."
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    tar -xzf "$TARBALL" -C "$TMP" --wildcards '*/data/NEU-DET/*'
    SRCDIR="$(find "$TMP" -type d -name 'NEU-DET' | head -1)"
    [[ -n "$SRCDIR" ]] || die "Khong tim thay thu muc NEU-DET trong tarball."
    mkdir -p "$DEST"
    cp -r "$SRCDIR"/. "$DEST"/
    echo "github:Marfbin/NEU-DET-with-yolov8@main (YOLO txt, da chia san train/test)" > "$RAW/SOURCE.txt"
    ;;

  kaggle)
    command -v kaggle >/dev/null 2>&1 || die \
"Chua co lenh 'kaggle'. Cai va lay token:
  pip install kaggle
  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json"

    ZIP="$CACHE/neu-det-kaggle.zip"
    if [[ ! -s "$ZIP" ]]; then
      log "Tai tu Kaggle: $KAGGLE_SLUG (~30 MB, license CC0)..."
      kaggle datasets download -d "$KAGGLE_SLUG" -p "$CACHE" --force \
        || die "Kaggle tai that bai (kiem tra token trong ~/.kaggle/kaggle.json)."
      mv "$CACHE"/*.zip "$ZIP" 2>/dev/null || true
    fi
    mkdir -p "$DEST"
    unzip -q -o "$ZIP" -d "$DEST"
    echo "kaggle:$KAGGLE_SLUG (ban goc, annotation VOC XML)" > "$RAW/SOURCE.txt"
    ;;

  *) die "Nguon khong hop le: '$SOURCE' (chi nhan: github | kaggle)" ;;
  esac
fi

log "Kiem tra dataset vua tai..."
python3 - "$DEST" <<'PY'
import sys, collections
from pathlib import Path

root = Path(sys.argv[1])
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
txts = [p for p in root.rglob("*.txt")]
xmls = [p for p in root.rglob("*.xml")]

print(f"  thu muc goc : {root}")
print(f"  anh         : {len(imgs)}")
print(f"  nhan YOLO   : {len(txts)}")
print(f"  nhan VOC XML: {len(xmls)}")

if not imgs:
    sys.exit("  [loi] khong tim thay anh nao -> dataset hong")

try:
    from PIL import Image
    sizes = collections.Counter(Image.open(p).size for p in imgs[:300])
    print(f"  kich thuoc  : {dict(sizes)}")
except ImportError:
    print("  kich thuoc  : (bo qua - chua cai Pillow)")

prefix = collections.Counter(p.stem.rsplit("_", 1)[0] for p in imgs)
print(f"  lop theo ten file ({len(prefix)}):")
for k, v in sorted(prefix.items()):
    print(f"      {k:18s} {v}")

if txts:
    ids, boxes, oob, empty = collections.Counter(), 0, 0, 0
    for t in txts:
        lines = [l for l in t.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for l in lines:
            f = l.split()
            ids[int(f[0])] += 1; boxes += 1
            x, y, w, h = map(float, f[1:5])
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                oob += 1
    print(f"  tong bbox   : {boxes}")
    print(f"  bbox loi bien: {oob}")
    print(f"  nhan rong   : {empty}")
    print(f"  class id    : {dict(sorted(ids.items()))}")

expected = 1800
if len(imgs) != expected:
    print(f"  [canh bao] mong doi {expected} anh, thay {len(imgs)}")
else:
    print(f"  [ok] du {expected} anh")
PY

log "Xong. Dataset o: $DEST"
log "Nguon: $(cat "$RAW/SOURCE.txt")"
