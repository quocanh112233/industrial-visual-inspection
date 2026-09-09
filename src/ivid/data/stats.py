"""FR-03 — Thong ke phan bo lop va hinh dang bounding box.

Ngoai bang dem mau theo lop (yeu cau toi thieu cua FR-03), script con tinh
kich thuoc va ti le khung — nhung con so nay giai thich VI SAO mot so lop kho:
khung rat lon phu kin anh (crazing) hoac rat det (scratches) deu lam giam mAP,
va do la phan tich can co trong bao cao thay vi chi noi "lop nay mAP thap".

Chay:
    PYTHONPATH=src python -m ivid.data.stats --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .common import IMG_EXT, SPLITS, class_names, load_config, repo_root, write_json


def gather(out_dir: Path, names: list[str]) -> dict:
    per_split: dict[str, dict] = {}
    for s in SPLITS:
        img_dir, lbl_dir = out_dir / s / "images", out_dir / s / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)

        boxes_by_class: dict[int, int] = defaultdict(int)
        imgs_by_class: dict[int, set] = defaultdict(set)
        areas_by_class: dict[int, list] = defaultdict(list)
        ars_by_class: dict[int, list] = defaultdict(list)
        boxes_per_image: list[int] = []

        for p in images:
            lp = lbl_dir / f"{p.stem}.txt"
            lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()] if lp.exists() else []
            boxes_per_image.append(len(lines))
            for line in lines:
                f = line.split()
                cid = int(f[0])
                _, _, bw, bh = map(float, f[1:5])
                boxes_by_class[cid] += 1
                imgs_by_class[cid].add(p.stem)
                areas_by_class[cid].append(bw * bh)
                ars_by_class[cid].append(bw / bh if bh > 0 else 0.0)

        per_split[s] = {
            "n_images": len(images),
            "n_boxes": sum(boxes_by_class.values()),
            "boxes_per_image_mean": round(statistics.fmean(boxes_per_image), 2) if boxes_per_image else 0,
            "boxes_by_class": {names[k]: v for k, v in sorted(boxes_by_class.items())},
            "images_by_class": {names[k]: len(v) for k, v in sorted(imgs_by_class.items())},
            "area_median_by_class": {
                names[k]: round(statistics.median(v), 4) for k, v in sorted(areas_by_class.items())
            },
            "aspect_median_by_class": {
                names[k]: round(statistics.median(v), 2) for k, v in sorted(ars_by_class.items())
            },
        }
    return per_split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="docs/dataset.md")
    a = ap.parse_args()

    root = repo_root()
    cfg = load_config(root / a.config if not Path(a.config).is_absolute() else a.config)
    names = class_names(cfg)
    out_dir = cfg["out_dir"]
    if not out_dir.exists():
        print(f"[loi] khong thay {out_dir} — chay ivid.data.prepare truoc", file=sys.stderr)
        return 1

    st = gather(out_dir, names)
    tot_box = {n: sum(st[s]["boxes_by_class"].get(n, 0) for s in SPLITS) for n in names}
    tot_img = {n: sum(st[s]["images_by_class"].get(n, 0) for s in SPLITS) for n in names}

    lo, hi = min(tot_box.values()), max(tot_box.values())
    imbalance = hi / lo if lo else float("inf")

    L: list[str] = []
    L.append(f"# Dataset — {cfg['dataset_name']}\n")
    L.append("*Sinh tự động bởi `ivid.data.stats` — chạy lại cho ra file giống hệt.*\n")
    L.append("Ảnh xám bề mặt thép cán nóng, kích thước gốc **200×200**, 6 loại lỗi. "
             "Chia train/val/test 70/15/15 phân tầng theo lớp, seed cố định "
             f"(`{cfg['split']['seed']}`) — xem `results/dataset_manifest.json`.\n")

    L.append("## Số ảnh mỗi tập\n")
    L.append("| Tập | Ảnh | Bbox | Bbox/ảnh |")
    L.append("|---|---:|---:|---:|")
    for s in SPLITS:
        L.append(f"| {s} | {st[s]['n_images']} | {st[s]['n_boxes']} | {st[s]['boxes_per_image_mean']} |")
    L.append(f"| **tổng** | **{sum(st[s]['n_images'] for s in SPLITS)}** | "
             f"**{sum(st[s]['n_boxes'] for s in SPLITS)}** | |\n")

    L.append("## Số bbox theo lớp (FR-03)\n")
    L.append("| Lớp | train | val | test | tổng | % |")
    L.append("|---|---:|---:|---:|---:|---:|")
    grand = sum(tot_box.values())
    for n in names:
        c = [st[s]["boxes_by_class"].get(n, 0) for s in SPLITS]
        L.append(f"| `{n}` | {c[0]} | {c[1]} | {c[2]} | **{tot_box[n]}** | {100*tot_box[n]/grand:.1f}% |")
    L.append(f"| **tổng** | | | | **{grand}** | |\n")

    L.append("## Số ảnh chứa mỗi lớp\n")
    L.append("| Lớp | train | val | test | tổng |")
    L.append("|---|---:|---:|---:|---:|")
    for n in names:
        c = [st[s]["images_by_class"].get(n, 0) for s in SPLITS]
        L.append(f"| `{n}` | {c[0]} | {c[1]} | {c[2]} | **{tot_img[n]}** |")
    L.append("")
    L.append("> Tổng cột này lớn hơn 1800 vì một ảnh có thể chứa nhiều loại lỗi.\n")

    L.append("## Hình dạng bounding box (tập train)\n")
    L.append("| Lớp | Diện tích trung vị (tỉ lệ ảnh) | Tỉ lệ rộng/cao trung vị |")
    L.append("|---|---:|---:|")
    for n in names:
        area = st["train"]["area_median_by_class"].get(n, 0)
        ar = st["train"]["aspect_median_by_class"].get(n, 0)
        L.append(f"| `{n}` | {area:.3f} | {ar:.2f} |")
    L.append("")

    L.append("## Nhận xét\n")
    L.append(f"- **Mất cân bằng lớp: {imbalance:.2f}×** giữa lớp nhiều nhất "
             f"(`{max(tot_box, key=tot_box.get)}`, {hi} bbox) và ít nhất "
             f"(`{min(tot_box, key=tot_box.get)}`, {lo} bbox).")
    if imbalance < 1.5:
        L.append("  Mức này nhẹ, chưa cần lấy mẫu lại hay đánh trọng số lớp.")
    elif imbalance < 3:
        L.append("  Mức này đáng chú ý — nếu mAP của lớp ít mẫu thấp hẳn thì đây là "
                 "một nguyên nhân, cần nói rõ trong báo cáo thay vì chỉ báo con số.")
    else:
        L.append("  Mức này nặng — cân nhắc đánh trọng số lớp hoặc tăng cường dữ liệu.")
    big = [n for n in names if st["train"]["area_median_by_class"].get(n, 0) > 0.25]
    if big:
        L.append(f"- Các lớp {', '.join('`'+n+'`' for n in big)} có khung chiếm hơn 25% diện tích ảnh. "
                 "Khung rất lớn và chồng lấn khiến IoU nhạy cảm, mAP@0.5:0.95 sẽ thấp hơn "
                 "mAP@0.5 nhiều — đây là đặc tính dataset, không phải lỗi model.")
    thin = [n for n in names if not 0.4 < st["train"]["aspect_median_by_class"].get(n, 1) < 2.5]
    if thin:
        L.append(f"- Các lớp {', '.join('`'+n+'`' for n in thin)} có khung rất dẹt. "
                 "Anchor-free như YOLOv8 xử lý được, nhưng đáng theo dõi recall của chúng.")
    L.append("- Ảnh gốc **200×200** trong khi benchmark chạy ở 640×640 (phóng to 3.2×). "
             "Điều này phải nêu rõ trong báo cáo: nó mở ra khả năng giảm `imgsz` để "
             "tăng tốc mà gần như không mất mAP — một hướng tối ưu đáng đo.")

    out_path = root / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(root / "results" / "data_stats.json",
               {"splits": st, "total_boxes_by_class": tot_box,
                "total_images_by_class": tot_img, "imbalance_ratio": round(imbalance, 3)})

    print(f"[stats] mat can bang lop : {imbalance:.2f}x "
          f"({max(tot_box, key=tot_box.get)}={hi} / {min(tot_box, key=tot_box.get)}={lo})")
    for n in names:
        print(f"[stats]   {n:<18} {tot_box[n]:5d} bbox  trong {tot_img[n]:5d} anh")
    print(f"[stats] ghi -> {a.out} va results/data_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
