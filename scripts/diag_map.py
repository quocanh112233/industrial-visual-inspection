from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.benchmark.accuracy import test_set  # noqa: E402
from ivid.benchmark.metrics import IOU_THRESHOLDS, compute_map, load_ground_truth  # noqa: E402
from ivid.benchmark.runners.base import build_runner, weights_for  # noqa: E402
from ivid.data.common import class_names, load_config, repo_root  # noqa: E402
from ivid.preprocess import read_image  # noqa: E402


def score_with_ultralytics(dets: list[np.ndarray], gts: list[np.ndarray]) -> dict | None:
    try:
        import torch
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import ap_per_class, box_iou
    except Exception as e:                                   # noqa: BLE001
        print(f"    (bo qua doi chieu: khong nap duoc ultralytics — {e})")
        return None

    holder = type("_V", (), {})()
    holder.iouv = torch.tensor(IOU_THRESHOLDS, dtype=torch.float32)

    tp_all, conf_all, pcls_all, tcls_all = [], [], [], []
    for det, gt in zip(dets, gts, strict=True):
        tcls_all.append(gt[:, 4] if len(gt) else np.zeros(0, np.float32))
        if len(det) == 0:
            continue
        conf_all.append(det[:, 4])
        pcls_all.append(det[:, 5])
        if len(gt) == 0:
            tp_all.append(np.zeros((len(det), len(IOU_THRESHOLDS)), dtype=bool))
            continue
        iou = box_iou(torch.from_numpy(gt[:, :4]), torch.from_numpy(det[:, :4]))
        tp = BaseValidator.match_predictions(
            holder, torch.from_numpy(det[:, 5]), torch.from_numpy(gt[:, 4]), iou)
        tp_all.append(tp.cpu().numpy())

    tp = np.concatenate(tp_all)
    res = ap_per_class(tp, np.concatenate(conf_all), np.concatenate(pcls_all),
                       np.concatenate(tcls_all), plot=False)
    ap = next((x for x in res if isinstance(x, np.ndarray) and x.ndim == 2
               and x.shape[1] == len(IOU_THRESHOLDS)), None)
    if ap is None:
        print("    (bo qua doi chieu: khong tim thay mang AP trong ket qua ap_per_class)")
        return None
    return {"mAP50": round(float(ap[:, 0].mean()), 4),
            "mAP50_95": round(float(ap.mean()), 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--backend", default="pytorch")
    p.add_argument("--data", default="data/processed/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--muc-tieu", type=float, default=0.7621,
                   help="con so ultralytics.val() bao cho ban .pt")
    a = p.parse_args()

    root = repo_root()
    names = class_names(load_config(root / "configs/data.yaml"))
    nc = len(names)
    images, label_dir = test_set(root / a.data, "test", a.max_images)
    gts = load_ground_truth(images, label_dir)
    imgs_bgr = [read_image(q) for q in images]
    print(f"[diag] {len(images)} anh, {sum(len(g) for g in gts)} bbox ground truth, {nc} lop")
    print(f"[diag] model {a.model}/{a.backend}, conf={a.conf} iou={a.iou}")
    print(f"[diag] moc doi chieu: ultralytics.val() = {a.muc_tieu:.4f}\n")

    w = weights_for(root / "models" / a.model, a.backend)
    if not w.exists():
        print(f"[loi] khong thay {w}", file=sys.stderr)
        return 1

    ket_qua = []
    for scaleup in (True, False):
        for multi_label in (False, True):
            nhan = f"scaleup={str(scaleup):<5} multi_label={str(multi_label):<5}"
            runner = build_runner(a.backend, w, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                                  nc=nc, scaleup=scaleup, multi_label=multi_label)
            runner.warmup(3, imgs_bgr[0])
            dets = [runner.run_array(im).detections for im in imgs_bgr]
            runner.close()

            m = compute_map(dets, gts, nc)
            lech = m["mAP50"] - a.muc_tieu
            print(f"[B] {nhan} -> mAP@0.5 {m['mAP50']:.4f} ({lech:+.4f})  "
                  f"mAP@0.5:0.95 {m['mAP50_95']:.4f}  ({m['n_detections']} detection)")
            ket_qua.append((scaleup, multi_label, m, dets))

    print("\n[A] Doi chieu PHEP TINH mAP tren cung mot tap detection:")
    for scaleup, multi_label, m, dets in ket_qua:
        ul = score_with_ultralytics(dets, gts)
        if ul is None:
            break
        d50 = m["mAP50"] - ul["mAP50"]
        print(f"    scaleup={str(scaleup):<5} multi_label={str(multi_label):<5} "
              f"| cua ta {m['mAP50']:.4f} | ma ultralytics {ul['mAP50']:.4f} "
              f"| lech {d50:+.4f} -> {'KHOP' if abs(d50) <= 0.005 else 'LECH'}")

    tot = min(ket_qua, key=lambda k: abs(k[2]["mAP50"] - a.muc_tieu))
    print(f"\n[diag] gan moc doi chieu nhat: scaleup={tot[0]} multi_label={tot[1]} "
          f"-> {tot[2]['mAP50']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
