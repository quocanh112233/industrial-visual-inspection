"""FR-09 — Xac minh ba dinh dang cho ra detection giong nhau.

Chay cung N anh qua ca ba runner voi cung tien/hau xu ly, roi so tung cap:
bao nhieu detection khop, bao nhieu chi mot ben co, IoU trung binh, lech diem
tin cay lon nhat.

SRS ghi ro: "neu TensorRT lech nhieu thi ghi nhan, khong che giau". Script vi
vay khong bao gio 'that bai' vi lech — no bao cao con so va de nguoi doc danh gia.

    PYTHONPATH=src python -m ivid.export.verify_parity --name yolov8n
"""
from __future__ import annotations

import argparse
import statistics
import sys
from itertools import combinations
from pathlib import Path

import yaml

from ..benchmark.runners.base import build_runner, weights_for
from ..data.common import IMG_EXT, repo_root, write_json
from ..postprocess import match_detections
from ..preprocess import read_image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--backends", nargs="*", default=["pytorch", "onnx", "tensorrt"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--iou-match", type=float, default=0.5,
                    help="IoU toi thieu de coi hai box la cung mot detection")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text())
    pc = cfg["parity"]
    n = a.n or int(pc["n_images_full"])
    imgsz = int(cfg["onnx"]["imgsz"])

    d = yaml.safe_load((root / a.data).read_text())
    img_dir = Path(d["path"]) / d.get("test", "test/images")
    images = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT),
                    key=lambda p: p.name)[:n]
    if not images:
        print("[loi] khong co anh test", file=sys.stderr)
        return 1

    runners = {}
    for b in a.backends:
        w = weights_for(root / "models" / a.name, b)
        if not w.exists():
            print(f"[parity] bo qua {b}: khong thay {w.name}")
            continue
        try:
            runners[b] = build_runner(b, w, imgsz=imgsz,
                                      conf=float(pc["conf"]), iou=float(pc["iou"]))
            print(f"[parity] nap {b}: {w.name}")
        except Exception as e:
            print(f"[parity] bo qua {b}: {type(e).__name__}: {e}")

    if len(runners) < 2:
        print("[loi] can it nhat 2 dinh dang de so sanh", file=sys.stderr)
        return 1

    print(f"[parity] chay {len(images)} anh qua {len(runners)} dinh dang...")
    dets: dict[str, list] = {b: [] for b in runners}
    for p in images:
        img = read_image(p)
        for b, r in runners.items():
            dets[b].append(r.run_array(img).detections)

    pairs: dict[str, dict] = {}
    for b1, b2 in combinations(runners.keys(), 2):
        per_image = [match_detections(dets[b1][i], dets[b2][i], a.iou_match)
                     for i in range(len(images))]
        tot_m = sum(x["matched"] for x in per_image)
        tot_a = sum(x["only_a"] for x in per_image)
        tot_b = sum(x["only_b"] for x in per_image)
        ious = [x["mean_iou"] for x in per_image if x["matched"]]
        confs = [x["max_conf_diff"] for x in per_image if x["matched"]]
        differing = [images[i].name for i, x in enumerate(per_image)
                     if x["only_a"] or x["only_b"]]
        pairs[f"{b1}_vs_{b2}"] = {
            "n_images": len(images),
            "detections": {b1: sum(len(x) for x in dets[b1]),
                           b2: sum(len(x) for x in dets[b2])},
            "matched": tot_m,
            f"only_{b1}": tot_a,
            f"only_{b2}": tot_b,
            "match_rate": round(tot_m / max(1, tot_m + tot_a + tot_b), 4),
            "images_with_difference": len(differing),
            "mean_iou_of_matched": round(statistics.fmean(ious), 4) if ious else 0.0,
            "max_conf_diff": round(max(confs), 4) if confs else 0.0,
            "examples_differing": differing[:10],
        }

    report = {"model": a.name, "n_images": len(images), "iou_match": a.iou_match,
              "conf": float(pc["conf"]), "iou_nms": float(pc["iou"]),
              "backends": list(runners.keys()),
              "total_detections": {b: sum(len(x) for x in dets[b]) for b in runners},
              "pairs": pairs}
    write_json(root / "results" / "parity_check.json", report)

    print(f"\n[parity] {'cap so sanh':<26}{'khop':>7}{'chi A':>7}{'chi B':>7}"
          f"{'ti le khop':>12}{'IoU tb':>9}{'lech conf':>11}")
    for k, v in pairs.items():
        b1, b2 = k.split("_vs_")
        print(f"         {k:<26}{v['matched']:>7}{v[f'only_{b1}']:>7}{v[f'only_{b2}']:>7}"
              f"{v['match_rate']*100:>11.1f}%{v['mean_iou_of_matched']:>9.3f}"
              f"{v['max_conf_diff']:>11.3f}")

    print("\n[parity] ghi -> results/parity_check.json")
    worst = min((v["match_rate"] for v in pairs.values()), default=1.0)
    if worst < 0.95:
        print(f"[parity] Ti le khop thap nhat {worst*100:.1f}%. Day la SO LIEU CAN BAO CAO, "
              "khong phai loi can giau — ghi vao docs/benchmark-report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
