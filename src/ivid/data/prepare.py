"""FR-01 — Chuan hoa NEU-DET ve dinh dang YOLO va chia train/val/test.

Nhan duoc ca hai kieu du lieu goc:

  * **VOC XML** (ban goc tren Kaggle): ANNOTATIONS/*.xml + IMAGES/*.jpg
  * **YOLO txt** (mirror GitHub): train/{images,labels} + test/{images,labels}

Du nguon nao, dau ra luon giong het nhau — day la diem mau chot: neu hai nguon
cho ra hai cach chia khac nhau thi so lieu benchmark khong so sanh duoc voi nhau.

Chay:
    PYTHONPATH=src python -m ivid.data.prepare --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

from .common import (
    IMG_EXT,
    SPLITS,
    class_names,
    load_config,
    primary_class,
    repo_root,
    sha256_of_files,
    write_json,
)


# --------------------------------------------------------------- doc nguon
def find_images(raw_dir: Path) -> list[Path]:
    return sorted(
        (p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXT),
        key=lambda p: p.name,
    )


def voc_to_yolo(xml_path: Path, names: list[str]) -> list[str]:
    """Doi mot file VOC XML thanh cac dong nhan YOLO (da chuan hoa 0..1)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = float(size.find("width").text), float(size.find("height").text)
    if w <= 0 or h <= 0:
        raise ValueError(f"{xml_path.name}: kich thuoc anh khong hop le ({w}x{h})")

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in names:
            raise ValueError(f"{xml_path.name}: lop la '{name}' (khong co trong configs/data.yaml)")
        b = obj.find("bndbox")
        x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        # VOC dem tu 1 va bao gom bien; kep vao trong anh truoc khi doi
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        if bw <= 0 or bh <= 0:
            continue  # box rong sau khi kep -> bo, validate.py se dem lai
        lines.append(f"{names.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def dedupe(lines: list[str]) -> tuple[list[str], int]:
    """Bo cac dong nhan trung y het nhau (cung lop, cung toa do).

    NEU-DET co 3 anh bi lap bbox. Ultralytics tu am tham bo chung luc train
    ("1 duplicate labels removed"), nghia la neu khong bo o day thi manifest se
    ghi so bbox KHAC voi so bbox model that su hoc — va con so trong bao cao
    khong con khop voi thuc te.
    """
    seen: set[tuple] = set()
    out, removed = [], 0
    for line in lines:
        f = line.split()
        try:
            key = (int(f[0]), *(round(float(v), 6) for v in f[1:5])) if len(f) == 5 else (line,)
        except ValueError:
            key = (line,)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(line)
    return out, removed


def label_for(img: Path, raw_dir: Path, names: list[str]) -> tuple[list[str], str]:
    """Tra ve (cac dong nhan YOLO, nguon: 'yolo'|'voc'|'missing')."""
    txt = img.parent.parent / "labels" / f"{img.stem}.txt"
    if not txt.exists():
        hits = list(raw_dir.rglob(f"{img.stem}.txt"))
        txt = hits[0] if hits else txt
    if txt.exists():
        return [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()], "yolo"

    xmls = list(raw_dir.rglob(f"{img.stem}.xml"))
    if xmls:
        return voc_to_yolo(xmls[0], names), "voc"

    return [], "missing"


# --------------------------------------------------------------- chia tap
def stratified_split(
    images: list[Path], ratios: dict[str, float], seed: int, stratify: bool
) -> dict[str, list[Path]]:
    """Chia co dinh theo seed. Chay hai lan cho ra ket qua giong het nhau
    vi danh sach da duoc sort truoc khi shuffle (FR-01)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        groups[primary_class(p) if stratify else "_all"].append(p)

    out: dict[str, list[Path]] = {s: [] for s in SPLITS}
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda p: p.name)
        random.Random(seed).shuffle(items)  # RNG rieng cho tung lop -> khong phu thuoc thu tu lop
        n = len(items)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        out["train"] += items[:n_train]
        out["val"] += items[n_train : n_train + n_val]
        out["test"] += items[n_train + n_val :]
    for s in SPLITS:
        out[s].sort(key=lambda p: p.name)
    return out


# --------------------------------------------------------------- ghi ra dia
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    a = ap.parse_args()

    root = repo_root()
    cfg = load_config(root / a.config if not Path(a.config).is_absolute() else a.config)
    names = class_names(cfg)
    raw_dir, out_dir = cfg["raw_dir"], cfg["out_dir"]

    if not raw_dir.exists():
        print(f"[loi] khong thay {raw_dir}\n      chay truoc: bash scripts/download_dataset.sh", file=sys.stderr)
        return 1

    images = find_images(raw_dir)
    if not images:
        print(f"[loi] khong co anh nao trong {raw_dir}", file=sys.stderr)
        return 1

    dup = {p.name for p in images if [q.name for q in images].count(p.name) > 1}
    if dup:
        print(f"[loi] ten anh bi trung giua cac thu muc con: {sorted(dup)[:5]}", file=sys.stderr)
        return 1

    print(f"[prepare] nguon    : {raw_dir}")
    print(f"[prepare] tim thay : {len(images)} anh")

    ratios = {k: float(cfg["split"][k]) for k in SPLITS}
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        print(f"[loi] ti le chia cong lai bang {total}, phai bang 1.0", file=sys.stderr)
        return 1

    splits = stratified_split(images, ratios, int(cfg["split"]["seed"]), bool(cfg["split"]["stratify"]))

    if out_dir.exists() and cfg.get("overwrite", True):
        shutil.rmtree(out_dir)
    for s in SPLITS:
        (out_dir / s / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "labels").mkdir(parents=True, exist_ok=True)

    src_kinds: dict[str, int] = defaultdict(int)
    n_boxes: dict[str, int] = defaultdict(int)
    n_empty: dict[str, int] = defaultdict(int)
    n_dupes: dict[str, int] = defaultdict(int)

    for split, items in splits.items():
        for img in items:
            shutil.copy2(img, out_dir / split / "images" / img.name)
            lines, kind = label_for(img, raw_dir, names)
            lines, removed = dedupe(lines)
            n_dupes[split] += removed
            src_kinds[kind] += 1
            n_boxes[split] += len(lines)
            if not lines:
                n_empty[split] += 1
            (out_dir / split / "labels" / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    # --- data.yaml cho ultralytics ---
    ds_yaml = out_dir / "data.yaml"
    ds_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {i: n for i, n in enumerate(names)},
            },
            sort_keys=False,
            allow_unicode=True,
        ), encoding="utf-8")

    # --- manifest tai lap (FR-06) ---
    manifest = {
        "dataset": cfg["dataset_name"],
        "raw_dir": str(raw_dir),
        "raw_source": (raw_dir.parent / "SOURCE.txt").read_text(encoding="utf-8").strip()
        if (raw_dir.parent / "SOURCE.txt").exists()
        else "?",
        "seed": int(cfg["split"]["seed"]),
        "ratios": ratios,
        "stratify": bool(cfg["split"]["stratify"]),
        "names": names,
        "annotation_source": dict(src_kinds),
        "counts": {s: len(splits[s]) for s in SPLITS},
        "boxes": dict(n_boxes),
        "duplicate_boxes_removed": dict(n_dupes),
        "images_without_boxes": dict(n_empty),
        "total_images": len(images),
        "dataset_sha256": sha256_of_files(images),
        "split_sha256": {
            s: sha256_of_files([out_dir / s / "labels" / f"{p.stem}.txt" for p in splits[s]])
            for s in SPLITS
        },
    }
    write_json(root / "results" / "dataset_manifest.json", manifest)

    print(f"[prepare] nguon nhan: {dict(src_kinds)}")
    for s in SPLITS:
        print(f"[prepare] {s:<6}: {len(splits[s]):5d} anh  {n_boxes[s]:5d} bbox  "
              f"{n_empty[s]:3d} anh khong co bbox  {n_dupes[s]:2d} bbox trung lap da bo")
    print(f"[prepare] tong     : {sum(len(v) for v in splits.values())} anh (goc {len(images)})")
    print(f"[prepare] ghi      : {ds_yaml}")
    print(f"[prepare] manifest : results/dataset_manifest.json")
    print(f"[prepare] hash     : {manifest['dataset_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
