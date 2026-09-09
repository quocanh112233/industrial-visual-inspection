from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ..data.common import IMG_EXT, repo_root, write_json
from ..preprocess import read_image
from . import device_info
from .resources import ResourceMonitor, measure_idle
from .runners.base import build_runner, weights_for


def pctl(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s))) - 1))
    return s[k]


def summarize(values: list[float], percentiles: list[int]) -> dict:
    if not values:
        return {}
    out = {f"p{p}": round(pctl(values, p), 3) for p in percentiles}
    out.update(
        min=round(min(values), 3),
        max=round(max(values), 3),
        mean=round(statistics.fmean(values), 3),
        std=round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        n=len(values),
    )
    return out


def _warm_opencv(n: int = 50) -> None:
    from ..preprocess import preprocess

    dummy = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    for _ in range(n):
        preprocess(dummy, 640)


def test_images(data_yaml: Path, split: str, limit: int | None) -> list[Path]:
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    img_dir = Path(d["path"]) / d.get(split, f"{split}/images")
    imgs = sorted((p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT),
                  key=lambda p: p.name)
    return imgs[:limit] if limit else imgs


def run_one_session(runner, images_bgr: list[np.ndarray]) -> dict:
    pre, inf, post, tot = [], [], [], []
    n_det = 0
    for img in images_bgr:
        out = runner.run_array(img)
        pre.append(out.timing.preprocess_ms)
        inf.append(out.timing.inference_ms)
        post.append(out.timing.postprocess_ms)
        tot.append(out.timing.total_ms)
        n_det += len(out.detections)
    return {"preprocess": pre, "inference": inf, "postprocess": post,
            "total": tot, "n_detections": n_det}


def benchmark_backend(model: str, backend: str, images: list[Path], cfg: dict,
                      root: Path) -> dict:
    weights = weights_for(root / "models" / model, backend)
    if not weights.exists():
        return {"skipped": True, "reason": f"không thấy {weights.relative_to(root)}"}

    kw = dict(imgsz=int(cfg["imgsz"]), conf=float(cfg["conf"]), iou=float(cfg["iou"]),
              resize_to=(int(cfg["resize_to"]) if cfg.get("resize_to") else None))
    if cfg.get("nc"):
        kw["nc"] = int(cfg["nc"])
    if backend == "onnx":
        kw["allow_cpu_fallback"] = bool(cfg.get("allow_onnx_cpu_fallback", True))

    idle = measure_idle(1.5)
    t0 = time.perf_counter()
    try:
        runner = build_runner(backend, weights, **kw)
    except Exception as e:
        return {"skipped": True, "reason": f"{type(e).__name__}: {e}"}
    load_s = time.perf_counter() - t0

    images_bgr = [read_image(p) for p in images]

    print(f"    warm-up {cfg['warmup']} lần (cả đường ống, bằng ảnh thật)...")
    runner.warmup(int(cfg["warmup"]), images_bgr[0])

    sessions: list[dict] = []
    percentiles = [int(p) for p in cfg["percentiles"]]
    n_sessions = int(cfg["sessions"])

    with ResourceMonitor() as rm:
        for s in range(n_sessions):
            if s > 0 and cfg["cooldown_seconds"]:
                print(f"    để nguội {cfg['cooldown_seconds']}s trước phiên {s+1}...")
                time.sleep(float(cfg["cooldown_seconds"]))
            temps_before = device_info.temperatures()
            t0 = time.perf_counter()
            raw = run_one_session(runner, images_bgr)
            wall = time.perf_counter() - t0
            temps_after = device_info.temperatures()

            sess = {
                "session": s + 1,
                "wall_seconds": round(wall, 2),
                "n_images": len(images_bgr),
                "n_detections": raw["n_detections"],
                "fps_wall": round(len(images_bgr) / wall, 2) if wall > 0 else 0.0,
                "stages": {k: summarize(raw[k], percentiles)
                           for k in ("preprocess", "inference", "postprocess", "total")},
                "temp_before_mean_c": round(sum(temps_before.values()) / len(temps_before), 1)
                if temps_before else None,
                "temp_after_mean_c": round(sum(temps_after.values()) / len(temps_after), 1)
                if temps_after else None,
            }
            sessions.append(sess)
            st = sess["stages"]
            print(f"    phien {s+1}/{n_sessions}: total p50={st['total']['p50']:.2f} ms  "
                  f"infer p50={st['inference']['p50']:.2f} ms  "
                  f"{sess['fps_wall']:.1f} FPS  {sess['temp_after_mean_c']}C")

    res = rm.summary()
    runner_info = runner.backend_info()
    runner.close()

    def med(stage: str, key: str) -> float:
        vals = [s["stages"][stage][key] for s in sessions if key in s["stages"][stage]]
        return round(statistics.median(vals), 3) if vals else float("nan")

    median_stages = {
        stage: {key: med(stage, key)
                for key in list(sessions[0]["stages"][stage].keys()) if key != "n"}
        for stage in ("preprocess", "inference", "postprocess", "total")
    }
    p50_by_session = [s["stages"]["total"]["p50"] for s in sessions]
    spread = (max(p50_by_session) - min(p50_by_session)) / min(p50_by_session) * 100 \
        if len(p50_by_session) > 1 and min(p50_by_session) > 0 else 0.0

    return {
        "skipped": False,
        "model": model,
        "backend": backend,
        "backend_info": runner_info,
        "load_seconds": round(load_s, 2),
        "model_size_mb": runner_info["model_size_mb"],
        "sessions": sessions,
        "median_stages": median_stages,
        "latency_p50_ms": median_stages["total"]["p50"],
        "latency_p95_ms": median_stages["total"]["p95"],
        "fps": round(1000.0 / median_stages["total"]["p50"], 1)
        if median_stages["total"]["p50"] else 0.0,
        "stage_sum_check": {
            "sum_of_parts_p50": round(median_stages["preprocess"]["p50"]
                                      + median_stages["inference"]["p50"]
                                      + median_stages["postprocess"]["p50"], 3),
            "total_p50": median_stages["total"]["p50"],
        },
        "session_p50_spread_percent": round(spread, 2),
        "resources": res,
        "idle_before_load": idle,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--data", default="data/processed/data.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--sessions", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--settle", type=int, default=None, help="giây chờ ổn định nhiệt trước khi đo")
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--nc", type=int, default=None,
                    help="ghi đè số lớp (vd 80 khi đo thử bằng model COCO pretrained)")
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--allow-non-jetson", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="cho phép ghi đè kết quả đã đo với nhiều ảnh/phiên hơn")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    for key, val in (("sessions", a.sessions), ("warmup", a.warmup),
                     ("settle_seconds", a.settle), ("cooldown_seconds", a.cooldown),
                     ("max_images", a.max_images), ("nc", a.nc)):
        if val is not None:
            cfg[key] = val
    models = a.models or cfg["models"]
    backends = a.backends or cfg["backends"]

    dev = device_info.collect("trước khi benchmark")
    if not dev["is_jetson"] and not a.allow_non_jetson:
        print("[lỗi] Ràng buộc C1: mọi phép đo hiệu năng phải chạy TRÊN JETSON.\n"
              "      Thêm --allow-non-jetson nếu chỉ muốn thử cho chạy được.", file=sys.stderr)
        return 1

    data = root / a.data
    if not data.exists():
        print(f"[lỗi] không thấy {data} — chạy 'make data' trước", file=sys.stderr)
        return 1
    images = test_images(data, cfg["split"], cfg.get("max_images"))
    if not images:
        print("[lỗi] tập test rong", file=sys.stderr)
        return 1

    print(f"[bench] thiet bi : {dev.get('board_model') or dev['hostname']} ({dev['arch']})")
    pm = dev["power_mode"]
    print(f"[bench] chế độ nguồn: {pm['current_name']} (id={pm['current_id']}) "
          f"trọng số {list(pm['available'].values())}")
    print(f"[bench] nhiet do : {dev['temperature_mean_c']} C")
    print(f"[bench] ảnh      : {len(images)} ({cfg['split']}), cùng thứ tự cho mọi định dạng")
    print(f"[bench] phien    : {cfg['sessions']} x (warm-up {cfg['warmup']})")

    if cfg.get("settle_seconds"):
        print(f"[bench] chờ ổn định nhiệt {cfg['settle_seconds']}s (SRS §7.1 buoc 1)...")
        time.sleep(float(cfg["settle_seconds"]))
        dev["temperatures_after_settle_c"] = device_info.temperatures()

    out_path = root / a.out
    payload = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.setdefault("runs", {})
    payload["device"] = dev
    payload["config"] = cfg
    payload["images"] = {"split": cfg["split"], "count": len(images),
                         "first": images[0].name, "last": images[-1].name}
    payload["benchmarked_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _warm_opencv()

    for model in models:
        for backend in backends:
            key = f"{model}|{backend}"
            print(f"\n[bench] === {model} / {backend} ===")
            old = payload["runs"].get(key)
            if old and not old.get("skipped") and not a.force:
                old_n = old.get("sessions", [{}])[0].get("n_images", 0)
                old_s = len(old.get("sessions", []))
                new_n, new_s = len(images), int(cfg["sessions"])
                if (old_n, old_s) > (new_n, new_s):
                    print(f"    BỎ QUA: đã có phép đo đầy đủ hơn ({old_n} ảnh x {old_s} phien) "
                          f"— lần này chỉ {new_n} ảnh x {new_s} phien.")
                    print("            Dùng --force để ghi đè, hoặc --out để ghi ra file khác.")
                    continue

            res = benchmark_backend(model, backend, images, cfg, root)
            payload["runs"][key] = res
            if res.get("skipped"):
                print(f"    BO QUA: {res['reason']}")
            else:
                ms = res["median_stages"]
                print(f"    -> total p50 {res['latency_p50_ms']} ms | p95 {res['latency_p95_ms']} ms "
                      f"| {res['fps']} FPS | model {res['model_size_mb']} MB")
                print(f"       pre {ms['preprocess']['p50']} + infer {ms['inference']['p50']} "
                      f"+ post {ms['postprocess']['p50']} ms")
                if res["session_p50_spread_percent"] > 10:
                    print(f"    [cảnh báo] p50 lệch {res['session_p50_spread_percent']}% giữa các phiên "
                          f"(> 10%, NFR-02 không đạt) — kiểm tra throttling nhiệt")
            write_json(out_path, payload)

    payload["device_after"] = device_info.collect("sau khi benchmark")
    write_json(out_path, payload)
    print(f"\n[bench] ghi -> {a.out}")
    print("[bench] buoc tiep: python -m ivid.benchmark.report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
