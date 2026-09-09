"""FR-02 — Kiem tra tinh toan ven du lieu sau khi chuan bi.

Tim va dem cac bat thuong sau, tren ca ba tap:

  1. anh khong co file nhan
  2. nhan tro toi anh khong ton tai
  3. anh co file nhan nhung rong (khong co bbox nao)
  4. bounding box vuot bien anh, hoac rong/am
  5. class id nam ngoai danh sach lop
  6. anh hong / khong doc duoc
  7. anh trung ten giua cac tap (ro ri du lieu train sang test)
  8. lop trong nhan khong khop tien to ten file (rieng NEU-DET)

Muc 8 khong phai loi — NEU-DET co anh chua nhieu loai loi cung luc. No duoc
bao cao de biet muc do "nhieu nhan mot anh", vi con so do anh huong toi cach
doc mAP theo tung lop sau nay.

Chay:
    PYTHONPATH=src python -m ivid.data.validate --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import IMG_EXT, SPLITS, class_names, load_config, primary_class, repo_root, write_json

EPS = 1e-6


def check_split(split_dir: Path, names: list[str]) -> dict:
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT) if img_dir.exists() else []
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    issues: dict[str, list] = defaultdict(list)

    for stem in sorted(img_stems - lbl_stems):
        issues["anh_khong_co_nhan"].append(stem)
    for stem in sorted(lbl_stems - img_stems):
        issues["nhan_khong_co_anh"].append(stem)

    # anh hong
    try:
        from PIL import Image

        for p in images:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                issues["anh_hong"].append(f"{p.name}: {type(e).__name__}")
    except ImportError:
        issues["_canh_bao"].append("chua cai Pillow -> bo qua kiem tra anh hong")

    n_boxes = 0
    per_class: dict[int, int] = defaultdict(int)
    multi_label = 0

    for lp in labels:
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        if not lines:
            issues["nhan_rong"].append(lp.stem)
            continue
        ids_here = set()
        for ln, line in enumerate(lines, 1):
            f = line.split()
            if len(f) != 5:
                issues["dong_nhan_sai_dinh_dang"].append(f"{lp.name}:{ln}")
                continue
            try:
                cid = int(f[0])
                cx, cy, bw, bh = map(float, f[1:])
            except ValueError:
                issues["dong_nhan_sai_dinh_dang"].append(f"{lp.name}:{ln}")
                continue

            n_boxes += 1
            ids_here.add(cid)
            per_class[cid] += 1

            if not 0 <= cid < len(names):
                issues["class_id_ngoai_pham_vi"].append(f"{lp.name}:{ln} -> id={cid}")
            if bw <= 0 or bh <= 0:
                issues["bbox_rong_hoac_am"].append(f"{lp.name}:{ln}")
            if (cx - bw / 2 < -EPS or cx + bw / 2 > 1 + EPS
                    or cy - bh / 2 < -EPS or cy + bh / 2 > 1 + EPS):
                issues["bbox_vuot_bien"].append(f"{lp.name}:{ln}")

        # muc 8: anh co nhieu loai loi
        prefix = primary_class(lp)
        expected = names.index(prefix) if prefix in names else None
        if expected is not None and ids_here - {expected}:
            multi_label += 1

    return {
        "n_images": len(images),
        "n_labels": len(labels),
        "n_boxes": n_boxes,
        "per_class": {names[k]: v for k, v in sorted(per_class.items()) if 0 <= k < len(names)},
        "anh_nhieu_loai_loi": multi_label,
        "issues": {k: v for k, v in issues.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="results/data_report.md")
    a = ap.parse_args()

    root = repo_root()
    cfg = load_config(root / a.config if not Path(a.config).is_absolute() else a.config)
    names = class_names(cfg)
    out_dir = cfg["out_dir"]

    if not out_dir.exists():
        print(f"[loi] khong thay {out_dir}\n      chay truoc: python -m ivid.data.prepare", file=sys.stderr)
        return 1

    report = {s: check_split(out_dir / s, names) for s in SPLITS}

    # muc 7: ro ri giua cac tap
    stems = {s: {p.stem for p in (out_dir / s / "images").iterdir()} for s in SPLITS}
    leaks = {}
    for i, a1 in enumerate(SPLITS):
        for a2 in SPLITS[i + 1:]:
            common = stems[a1] & stems[a2]
            if common:
                leaks[f"{a1}∩{a2}"] = sorted(common)[:20]

    total_issues = sum(
        len(v) for s in SPLITS for k, v in report[s]["issues"].items() if not k.startswith("_")
    ) + sum(len(v) for v in leaks.values())

    # ---------------------------------------------------------------- bao cao
    L: list[str] = []
    L.append("# Báo cáo kiểm tra dữ liệu (FR-02)\n")
    L.append(f"*Sinh tự động lúc {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
             f"bởi `ivid.data.validate`*\n")
    L.append(f"Thư mục kiểm tra: `{out_dir}`\n")

    L.append("## Tổng quan\n")
    L.append("| Tập | Ảnh | File nhãn | Bbox | Ảnh nhiều loại lỗi |")
    L.append("|---|---:|---:|---:|---:|")
    for s in SPLITS:
        r = report[s]
        L.append(f"| {s} | {r['n_images']} | {r['n_labels']} | {r['n_boxes']} | {r['anh_nhieu_loai_loi']} |")
    tot_i = sum(report[s]["n_images"] for s in SPLITS)
    tot_b = sum(report[s]["n_boxes"] for s in SPLITS)
    L.append(f"| **tổng** | **{tot_i}** | | **{tot_b}** | |\n")

    L.append("## Bất thường\n")
    kinds = [
        ("anh_khong_co_nhan", "Ảnh không có file nhãn"),
        ("nhan_khong_co_anh", "Nhãn trỏ tới ảnh không tồn tại"),
        ("nhan_rong", "File nhãn rỗng (ảnh không có bbox nào)"),
        ("bbox_vuot_bien", "Bounding box vượt biên ảnh"),
        ("bbox_rong_hoac_am", "Bounding box rỗng hoặc kích thước âm"),
        ("class_id_ngoai_pham_vi", "Class id ngoài danh sách lớp"),
        ("dong_nhan_sai_dinh_dang", "Dòng nhãn sai định dạng"),
        ("anh_hong", "Ảnh hỏng / không đọc được"),
    ]
    L.append("| Loại bất thường | train | val | test | tổng |")
    L.append("|---|---:|---:|---:|---:|")
    for key, label in kinds:
        cells = [len(report[s]["issues"].get(key, [])) for s in SPLITS]
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | **{sum(cells)}** |")
    L.append(f"| Ảnh trùng giữa các tập (rò rỉ) | | | | **{sum(len(v) for v in leaks.values())}** |\n")

    # chi tiet vai vi du neu co
    for s in SPLITS:
        for key, label in kinds:
            v = report[s]["issues"].get(key, [])
            if v:
                L.append(f"<details><summary>{label} — {s} ({len(v)})</summary>\n")
                L.append("```")
                L.extend(v[:50])
                if len(v) > 50:
                    L.append(f"... còn {len(v) - 50} mục nữa")
                L.append("```\n</details>\n")
    if leaks:
        L.append("<details><summary>Ảnh trùng giữa các tập</summary>\n")
        L.append("```")
        for k, v in leaks.items():
            L.append(f"{k}: {v}")
        L.append("```\n</details>\n")

    L.append("## Phân bố bbox theo lớp\n")
    L.append("| Lớp | " + " | ".join(SPLITS) + " | tổng |")
    L.append("|---|" + "---:|" * (len(SPLITS) + 1))
    for n in names:
        cells = [report[s]["per_class"].get(n, 0) for s in SPLITS]
        L.append(f"| {n} | " + " | ".join(str(c) for c in cells) + f" | **{sum(cells)}** |")

    L.append("\n## Kết luận\n")
    if total_issues == 0:
        L.append("✅ **Không phát hiện bất thường nào.** Dữ liệu sẵn sàng để train.")
    else:
        L.append(f"⚠️ **Phát hiện {total_issues} bất thường** — xem chi tiết bên trên.")

    out_path = root / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    write_json(root / "results" / "data_report.json", {"splits": report, "leaks": leaks,
                                                       "total_issues": total_issues})

    for s in SPLITS:
        r = report[s]
        print(f"[validate] {s:<6}: {r['n_images']:5d} anh  {r['n_boxes']:5d} bbox  "
              f"{r['anh_nhieu_loai_loi']:3d} anh nhieu loai loi")
    print(f"[validate] tong bat thuong: {total_issues}")
    print(f"[validate] bao cao -> {a.out}")
    return 0 if total_issues == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
