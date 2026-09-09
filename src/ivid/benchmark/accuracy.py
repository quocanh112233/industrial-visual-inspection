"""FR-13 — Do mAP rieng cho tung dinh dang tren cung tap test.

Cot loi cua ca du an: neu chi do latency, khong ai biet TensorRT FP16 danh doi
bao nhieu do chinh xac. SRS §3.1 noi ro — "phai dinh luong, khong gia dinh la
bang nhau".

VI SAO KHONG DUNG ultralytics.val()

Ban dau buoc nay goi `ultralytics.val()` cho ca ba dinh dang, voi ly do "cung
mot ham cham diem thi khong the lech vi cach cham". Do la sai lam: goi CUNG MOT
HAM khong co nghia la dung CUNG MOT PHEP DO. No bao:

    yolov8n PyTorch  mAP@0.5 0.7621   mAP@0.5:0.95 0.4348
    yolov8n ONNX     mAP@0.5 0.7312   mAP@0.5:0.95 0.3520   (-0.031 / -0.083)

Chenh lech ay khong phai do ONNX kem hon: no la FP32, diem so lop chi lech
1.371e-06 so voi PyTorch, va parity @ conf 0.001 cho 8629 vs 8624 detection
khop 98.9%. Nguyen nhan that da truy ra duoc bang scripts/diag_rect.py —
ultralytics dat rect=True cho ban .pt nhung ep rect=False cho moi dinh dang
xuat (engine/validator.py). O che do rect voi pad=0.5, khung anh khong phai
640 ma la ceil(640/32 + 0.5) * 32 = 672: anh 640 nam giua mot khung 672 co
vien xam 16 px moi ben. Do lai tren CUNG mot ban .pt:

    rect=True  imgsz=640   mAP@0.5 0.7621   mAP@0.5:0.95 0.4348
    rect=False imgsz=640   mAP@0.5 0.7310   mAP@0.5:0.95 0.3521
    rect=False imgsz=672   mAP@0.5 0.7279   mAP@0.5:0.95 0.3410

Dong thu ba loai bo cach giai thich "do phan giai cao hon": phong thang anh len
672 con KEM hon. Thu tao ra khoang cach la VIEN XAM cua che do rect.

Engine ONNX/TensorRT co dau vao co dinh 640x640 nen khong tai lap duoc che do
ay. Vi vay 0.7621 la con so cua trinh danh gia chu khong phai con so chay duoc
tren day chuyen — dung luan diem SRS 3.1.

Module nay tu chay tap test qua chinh runner cua du an — von dung chung
ivid.preprocess va ivid.postprocess cho ca ba dinh dang — roi tinh mAP bang
ivid.benchmark.metrics. Chenh lech con lai chi den tu ban than runtime.

Ban .pt duoc doi chieu voi ultralytics.val(rect=False) — DUNG moc, cung che do
khung anh — de xac nhan phep tinh mAP tu viet la dung (--cross-check). Phep tinh
ay con duoc kiem chung doc lap bang scripts/diag_map.py: cham cung mot tap
detection bang chinh ma ultralytics (match_predictions + ap_per_class) cho ket
qua lech 0.0004.

    PYTHONPATH=src python -m ivid.benchmark.accuracy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from ..data.common import IMG_EXT, class_names, load_config, repo_root, write_json
from ..preprocess import read_image
from .metrics import compute_map, load_ground_truth
from .runners.base import build_runner, weights_for

BACKEND_LABEL = {"pytorch": "PyTorch", "onnx": "ONNX Runtime", "tensorrt": "TensorRT FP16"}
NAMES: list[str] = []


def test_set(data_yaml: Path, split: str, limit: int | None = None):
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(d["path"])
    img_dir = root / d.get(split, f"{split}/images")
    imgs = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT),
                  key=lambda p: p.name)
    if limit:
        imgs = imgs[:limit]
    return imgs, img_dir.parent / "labels"


def evaluate_backend(model: str, backend: str, images: list[Path], gts: list,
                     nc: int, imgsz: int, conf: float, iou: float, root: Path,
                     resize_to: int | None = None) -> dict:
    w = weights_for(root / "models" / model, backend)
    if not w.exists():
        return {"skipped": True, "reason": f"khong thay {w.name}"}
    try:
        runner = build_runner(backend, w, imgsz=imgsz, conf=conf, iou=iou, nc=nc,
                              resize_to=resize_to)
    except Exception as e:
        return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}

    imgs_bgr = [read_image(p) for p in images]
    runner.warmup(5, imgs_bgr[0])
    t0 = time.perf_counter()
    dets = [runner.run_array(im).detections for im in imgs_bgr]
    wall = time.perf_counter() - t0
    info = runner.backend_info()
    runner.close()

    m = compute_map(dets, gts, nc)
    m.update(skipped=False, model=model, backend=backend, class_names=NAMES,
             backend_label=BACKEND_LABEL.get(backend, backend),
             conf=conf, iou=iou, imgsz=imgsz, resize_to=resize_to,
             wall_seconds=round(wall, 2), backend_info=info)
    return m


def cross_check_pt(model: str, data: Path, imgsz: int, conf: float, iou: float,
                   root: Path) -> dict:
    """Chay ultralytics.val(rect=False) tren ban .pt de doi chieu phep tinh mAP.

    PHAI ep rect=False. Mac dinh ultralytics dung rect=True cho .pt, tuc cham
    diem o khung 672x672 co vien xam — mot che do ma engine 640x640 khong tai
    lap duoc. So sanh voi che do ay la so sanh nham moc: no tung lam cross-check
    bao "LECH LON 0.0304" trong khi phep tinh mAP hoan toan dung.
    """
    from ..train.evaluate import evaluate as ul_evaluate

    w = root / "models" / model / "best.pt"
    if not w.exists():
        return {"skipped": True, "reason": "khong thay best.pt"}
    try:
        r = ul_evaluate(w, data, "test", imgsz, batch=1, device=None, conf=conf,
                        iou=iou, rect=False)
        return {"skipped": False, "mAP50": r["overall"]["mAP50"],
                "mAP50_95": r["overall"]["mAP50_95"]}
    except Exception as e:
        return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="thap de duong PR day du — mAP quet moi nguong, khong dung 1 nguong")
    ap.add_argument("--iou", type=float, default=0.7, help="nguong IoU cua NMS")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="ghi de imgsz trong config (kich thuoc dau vao model)")
    ap.add_argument("--resize-to", type=int, default=None,
                    help="thu anh ve kich thuoc nay roi dem xam ra imgsz. "
                         "imgsz=672 --resize-to 640 tai lap che do rect cua ultralytics")
    ap.add_argument("--cross-check", action="store_true",
                    help="doi chieu ban .pt voi ultralytics.val() de xac nhan phep tinh mAP")
    ap.add_argument("--out", default="results/benchmark.json")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    models = a.models or cfg["models"]
    backends = a.backends or cfg["backends"]
    imgsz = int(a.imgsz or cfg["imgsz"])
    if a.resize_to is None and a.imgsz is None and cfg.get("resize_to"):
        a.resize_to = int(cfg["resize_to"])      # chi lay tu config khi khong ghi de imgsz
    global NAMES
    NAMES = class_names(load_config(root / a.data_config))
    nc = len(NAMES)

    data = root / a.data
    if not data.exists():
        print(f"[loi] khong thay {data}", file=sys.stderr)
        return 1

    images, label_dir = test_set(data, cfg["split"], a.max_images)
    print(f"[acc] tap test : {len(images)} anh, {nc} lop")
    print(f"[acc] tham so  : imgsz={imgsz} conf={a.conf} iou={a.iou}")
    print("[acc] doc ground truth...")
    gts = load_ground_truth(images, label_dir)
    print(f"[acc] ground truth: {sum(len(g) for g in gts)} bbox")

    out_path = root / a.out
    payload = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.setdefault("accuracy", {})
    payload["accuracy_method"] = {
        "cach_do": "runner cua du an (ivid.preprocess + ivid.postprocess dung chung) "
                   "+ ivid.benchmark.metrics.compute_map",
        "vi_sao_khong_dung_ultralytics_val": (
            "ultralytics.val() doi pipeline danh gia tuy theo dinh dang model, nen no do "
            "ca su khac biet cua pipeline lan cua runtime. Xem docstring "
            "ivid/benchmark/accuracy.py va results/parity_conf001.json."),
        "conf": a.conf, "iou_nms": a.iou, "imgsz": imgsz, "resize_to": a.resize_to,
        "cong_thuc_AP": "noi suy 101 diem, khop voi ultralytics compute_ap(method='interp')",
    }

    for model in models:
        for backend in backends:
            key = f"{model}|{backend}"
            print(f"\n[acc] === {model} / {BACKEND_LABEL.get(backend, backend)} ===")
            r = evaluate_backend(model, backend, images, gts, nc, imgsz, a.conf, a.iou,
                                 root, a.resize_to)
            payload["accuracy"][key] = r
            if r.get("skipped"):
                print(f"    BO QUA: {r['reason']}")
            else:
                print(f"    mAP@0.5 {r['mAP50']:.4f}   mAP@0.5:0.95 {r['mAP50_95']:.4f}   "
                      f"P {r['precision']:.4f}  R {r['recall']:.4f}   "
                      f"({r['n_detections']} detection, {r['wall_seconds']}s)")
            write_json(out_path, payload)

    # --- doi chieu phep tinh mAP tu viet voi ultralytics, tren ban .pt ---
    if a.cross_check:
        payload["accuracy_cross_check"] = {
            "ghi_chu": "ultralytics.val() duoc ep rect=False de cung che do khung anh "
                       "640x640 voi engine trien khai. Mac dinh rect=True cua no cham o "
                       "khung 672x672 co vien xam va cho mAP cao hon ~0.031 — xem "
                       "scripts/diag_rect.py va docstring ivid/train/evaluate.py.",
        }
        for model in models:
            print(f"\n[acc] === doi chieu {model}/.pt voi ultralytics.val(rect=False) ===")
            ul = cross_check_pt(model, data, imgsz, a.conf, a.iou, root)
            ours = payload["accuracy"].get(f"{model}|pytorch", {})
            if not ul.get("skipped") and not ours.get("skipped"):
                d50 = round(ours["mAP50"] - ul["mAP50"], 4)
                d95 = round(ours["mAP50_95"] - ul["mAP50_95"], 4)
                ul.update(ours_mAP50=ours["mAP50"], ours_mAP50_95=ours["mAP50_95"],
                          delta_mAP50=d50, delta_mAP50_95=d95, khop=abs(d50) <= 0.02)
                print(f"    ultralytics {ul['mAP50']:.4f} | cua ta {ours['mAP50']:.4f} "
                      f"| lech {d50:+.4f}  -> {'KHOP' if ul['khop'] else 'LECH LON'}")
            payload["accuracy_cross_check"][model] = ul
        write_json(out_path, payload)

    print("\n[acc] Chenh lech so voi PyTorch (moc goc):")
    for model in models:
        base = payload["accuracy"].get(f"{model}|pytorch")
        if not base or base.get("skipped"):
            continue
        for backend in backends:
            r = payload["accuracy"].get(f"{model}|{backend}")
            if not r or r.get("skipped"):
                continue
            print(f"    {model:<9} {BACKEND_LABEL.get(backend, backend):<15} "
                  f"mAP@0.5 {r['mAP50']:.4f} ({r['mAP50'] - base['mAP50']:+.4f})   "
                  f"mAP@0.5:0.95 {r['mAP50_95']:.4f} ({r['mAP50_95'] - base['mAP50_95']:+.4f})")

    print(f"\n[acc] ghi -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
