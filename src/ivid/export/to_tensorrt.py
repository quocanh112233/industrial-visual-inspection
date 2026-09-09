"""FR-08 — Build TensorRT engine FP16 tu file ONNX. BAT BUOC chay tren Jetson.

Engine TensorRT gan chat voi phan cung va phien ban thu vien: engine build tren
may khac se khong nap duoc, hoac nap duoc nhung sai. Vi vay script tu chan neu
khong phai aarch64, tru khi ep bang --allow-non-jetson.

Dung `trtexec` thay vi TensorRT Python API co chu dich: day dung dung binary ma
smoke test R1 da kiem chung, va log build (chon layer, chien luoc chien thuat)
duoc luu lai de doi chieu khi so lieu benchmark co bat thuong.

    PYTHONPATH=src python -m ivid.export.to_tensorrt --name yolov8n
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ..data.common import repo_root, write_json


def device_metadata() -> dict:
    """Ghi lai dieu kien build (FR-14 doi hoi cho luc do; luc build cung nen co,
    vi engine build o che do nguon thap co the chon chien thuat khac)."""
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    md = {
        "arch": platform.machine(),
        "l4t": sh("head -1 /etc/nv_tegra_release"),
        "cuda": sh("/usr/local/cuda/bin/nvcc --version | tail -2 | head -1"),
        "nvpmodel_id": sh("awk -F: '{print $2+0}' /var/lib/nvpmodel/status 2>/dev/null"),
        "power_modes": sh("grep -oP '(?<=^< POWER_MODEL ID=).*(?= >)' /etc/nvpmodel.conf 2>/dev/null"
                          " | tr '\\n' '|'"),
    }
    for mod in ("tensorrt", "torch", "onnx"):
        try:
            md[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            md[mod] = "khong cai"
    return md


def build(onnx_path: Path, engine_path: Path, trtexec: str, precision: str,
          workspace_mib: int, log_path: Path, extra: list[str]) -> tuple[bool, float, str]:
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        # TensorRT 10 da bo --workspace; day la cu phap thay the
        f"--memPoolSize=workspace:{workspace_mib}MiB",
        "--skipInference",          # do latency o buoc rieng, khong tin so cua trtexec
        *extra,
    ]
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    elif precision.lower() == "int8":
        cmd.append("--int8")
    elif precision.lower() not in ("fp32", ""):
        raise ValueError(f"precision khong ho tro: {precision}")

    print(f"[trt] {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n===== stderr =====\n" + proc.stderr, encoding="utf-8")
    return proc.returncode == 0, dt, proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--precision", default=None, help="ghi de fp16/fp32/int8")
    ap.add_argument("--allow-non-jetson", action="store_true",
                    help="cho phep build ngoai Jetson (engine se KHONG dung duoc tren Jetson)")
    ap.add_argument("--skip-check", action="store_true", help="bo qua buoc nap engine chay thu")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    tc = cfg["tensorrt"]
    precision = a.precision or tc["precision"]

    if platform.machine() != "aarch64" and not a.allow_non_jetson:
        print("[loi] Dang khong chay tren Jetson (aarch64).\n"
              "      Engine TensorRT khong di chuyen duoc giua cac may — build o day\n"
              "      thi tren Jetson se khong nap duoc. Chay lai script nay tren Jetson,\n"
              "      hoac them --allow-non-jetson neu ban that su muon.", file=sys.stderr)
        return 1

    onnx_path = root / "models" / a.name / "best.onnx"
    if not onnx_path.exists():
        print(f"[loi] khong thay {onnx_path}\n"
              f"      chay truoc: python -m ivid.export.to_onnx --name {a.name}", file=sys.stderr)
        return 1

    trtexec = tc["trtexec"]
    if not (Path(trtexec).exists() or shutil.which(trtexec)):
        print(f"[loi] khong thay trtexec tai {trtexec}", file=sys.stderr)
        return 1

    engine_path = onnx_path.with_suffix(".engine")
    log_path = root / "results" / f"trtexec_{a.name}.log"

    extra: list[str] = []
    if tc.get("builder_optimization_level") is not None:
        extra.append(f"--builderOptimizationLevel={int(tc['builder_optimization_level'])}")

    print(f"[trt] onnx      : {onnx_path}")
    print(f"[trt] precision : {precision}   workspace: {tc['workspace_mib']} MiB")
    print("[trt] build co the mat 3-8 phut, dung ngat...")

    ok, dt, out = build(onnx_path, engine_path, trtexec, precision,
                        int(tc["workspace_mib"]), log_path, extra)

    report = {
        "model": a.name,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": precision,
        "workspace_mib": int(tc["workspace_mib"]),
        "build_ok": ok,
        "build_seconds": round(dt, 1),
        "trtexec_log": str(log_path.relative_to(root)),
        "device": device_metadata(),
    }

    if not ok:
        print(f"[loi] trtexec that bai sau {dt:.0f}s — xem {log_path}", file=sys.stderr)
        for line in out.splitlines()[-20:]:
            print("   ", line, file=sys.stderr)
        write_json(root / "results" / f"export_{a.name}_tensorrt.json", report)
        return 1

    report["engine_size_mb"] = round(engine_path.stat().st_size / 1e6, 2)
    m = re.search(r"Engine built in ([\d.]+) sec", out)
    if m:
        report["trtexec_reported_build_sec"] = float(m.group(1))

    print(f"[trt] engine OK sau {dt:.0f}s -> {engine_path} ({report['engine_size_mb']} MB)")

    if not a.skip_check:
        print("[trt] nap engine va chay thu...")
        from .trt_infer_check import run as trt_run

        try:
            chk = trt_run(engine_path, iters=50, warmup=20)
            report["load_check"] = chk
            print(f"[trt] chay thu OK: p50 {chk['latency_ms']['p50']} ms "
                  f"({chk['fps']} FPS, chi inference thuan)")
            print("[trt] io: " + ", ".join(f"{t['name']}{t['shape']}" for t in chk["tensors"]))
        except Exception as e:
            report["load_check"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[loi] nap engine that bai: {e}", file=sys.stderr)
            write_json(root / "results" / f"export_{a.name}_tensorrt.json", report)
            return 1

    write_json(root / "results" / f"export_{a.name}_tensorrt.json", report)
    print(f"[trt] bao cao -> results/export_{a.name}_tensorrt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
