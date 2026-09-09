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

from ..data.common import rel_to_root, repo_root, write_json


def device_metadata() -> dict:
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
            md[mod] = "không cài"
    return md


def build(onnx_path: Path, engine_path: Path, trtexec: str, precision: str,
          workspace_mib: int, log_path: Path, extra: list[str]) -> tuple[bool, float, str]:
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}MiB",
        "--skipInference",
        *extra,
    ]
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    elif precision.lower() == "int8":
        cmd.append("--int8")
    elif precision.lower() not in ("fp32", ""):
        raise ValueError(f"precision không ho tro: {precision}")

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
    ap.add_argument("--precision", default=None, help="ghi đè fp16/fp32/int8")
    ap.add_argument("--allow-non-jetson", action="store_true",
                    help="cho phép build ngoài Jetson (engine sẽ KHÔNG chạy được trên Jetson)")
    ap.add_argument("--skip-check", action="store_true", help="bỏ qua bước nạp engine chạy thử")
    a = ap.parse_args()

    root = repo_root()
    cfg = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    tc = cfg["tensorrt"]
    precision = a.precision or tc["precision"]

    if platform.machine() != "aarch64" and not a.allow_non_jetson:
        print("[lỗi] Đang không chạy trên Jetson (aarch64).\n"
              "      Engine TensorRT không di chuyển được giữa các máy — build ở đây\n"
              "      thì trên Jetson sẽ không nạp được. Chạy lại script này trên Jetson,\n"
              "      hoặc thêm --allow-non-jetson nếu bạn thật sự muốn.", file=sys.stderr)
        return 1

    onnx_path = root / "models" / a.name / "best.onnx"
    if not onnx_path.exists():
        print(f"[lỗi] không thấy {onnx_path}\n"
              f"      chạy trước: python -m ivid.export.to_onnx --name {a.name}", file=sys.stderr)
        return 1

    trtexec = tc["trtexec"]
    if not (Path(trtexec).exists() or shutil.which(trtexec)):
        print(f"[lỗi] không thấy trtexec tại {trtexec}", file=sys.stderr)
        return 1

    engine_path = onnx_path.with_suffix(".engine")
    log_path = root / "results" / f"trtexec_{a.name}.log"

    extra: list[str] = []
    if tc.get("builder_optimization_level") is not None:
        extra.append(f"--builderOptimizationLevel={int(tc['builder_optimization_level'])}")

    print(f"[trt] onnx      : {onnx_path}")
    print(f"[trt] precision : {precision}   workspace: {tc['workspace_mib']} MiB")
    print("[trt] build có thể mất 3-8 phút, đừng ngắt...")

    ok, dt, out = build(onnx_path, engine_path, trtexec, precision,
                        int(tc["workspace_mib"]), log_path, extra)

    report = {
        "model": a.name,
        "onnx": rel_to_root(onnx_path),
        "engine": rel_to_root(engine_path),
        "precision": precision,
        "workspace_mib": int(tc["workspace_mib"]),
        "build_ok": ok,
        "build_seconds": round(dt, 1),
        "trtexec_log": str(log_path.relative_to(root)),
        "device": device_metadata(),
    }

    if not ok:
        print(f"[lỗi] trtexec thất bại sau {dt:.0f}s — xem {log_path}", file=sys.stderr)
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
        print("[trt] nạp engine và chạy thử...")
        from .trt_infer_check import run as trt_run

        try:
            chk = trt_run(engine_path, iters=50, warmup=20)
            report["load_check"] = chk
            print(f"[trt] chạy thử OK: p50 {chk['latency_ms']['p50']} ms "
                  f"({chk['fps']} FPS, chỉ inference thuan)")
            print("[trt] io: " + ", ".join(f"{t['name']}{t['shape']}" for t in chk["tensors"]))
        except Exception as e:
            report["load_check"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[lỗi] nạp engine thất bại: {e}", file=sys.stderr)
            write_json(root / "results" / f"export_{a.name}_tensorrt.json", report)
            return 1

    write_json(root / "results" / f"export_{a.name}_tensorrt.json", report)
    print(f"[trt] báo cáo -> results/export_{a.name}_tensorrt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
