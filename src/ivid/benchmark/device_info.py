"""FR-14 — Ghi lai dieu kien do.

Mot bang benchmark khong kem dieu kien do la mot bang khong kiem chung duoc:
cung mot Jetson chay o 15W va o MAXN_SUPER cho ra so lieu chenh nhau rat nhieu,
va so do luc chip da nong 80C khac han luc vua khoi dong. Vi vay moi ket qua
benchmark deu phai di kem khoi metadata nay.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sh(cmd: str, timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def power_modes() -> dict:
    """Doc /etc/nvpmodel.conf: cac che do co san + che do dang dung."""
    modes: dict[int, str] = {}
    conf = Path("/etc/nvpmodel.conf")
    if conf.exists():
        for m in re.finditer(r"^< POWER_MODEL ID=(\d+) NAME=(\S+) >", conf.read_text(encoding="utf-8"), re.M):
            modes[int(m.group(1))] = m.group(2)

    current = None
    status = Path("/var/lib/nvpmodel/status")
    if status.exists():
        m = re.search(r"pmode:(\d+)", status.read_text(encoding="utf-8"))
        if m:
            current = int(m.group(1))
    if current is None:
        m = re.search(r"NV Power Mode:\s*(\S+)", _sh("nvpmodel -q 2>/dev/null"))
        if m:
            for k, v in modes.items():
                if v == m.group(1):
                    current = k

    return {
        "available": {str(k): v for k, v in sorted(modes.items())},
        "current_id": current,
        "current_name": modes.get(current, "?") if current is not None else "?",
    }


def temperatures() -> dict[str, float]:
    out: dict[str, float] = {}
    for z in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
        try:
            v = int((z / "temp").read_text(encoding="utf-8")) / 1000.0
        except Exception:
            continue
        # mot so zone tren x86 bao gia tri vo nghia (0.05C); bo di de trung binh
        # khong bi keo lech. Tren Jetson moi zone deu that.
        if 5.0 < v < 150.0:
            out[(z / "type").read_text(encoding="utf-8").strip()] = v
    return out


def clocks() -> dict:
    """Tan so CPU/GPU hien tai (MHz)."""
    out: dict[str, float] = {}
    cpu = []
    for p in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        try:
            cpu.append(int(p.read_text(encoding="utf-8")) / 1000.0)
        except Exception:
            continue
    if cpu:
        out["cpu_mhz_mean"] = round(sum(cpu) / len(cpu), 1)
        out["cpu_mhz_max_core"] = round(max(cpu), 1)
    for g in ("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",
              "/sys/class/devfreq/17000000.gpu/cur_freq"):
        p = Path(g)
        if p.exists():
            try:
                out["gpu_mhz"] = round(int(p.read_text(encoding="utf-8")) / 1e6, 1)
                break
            except Exception:
                pass
    return out


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for mod in ("torch", "torchvision", "tensorrt", "onnx", "onnxruntime", "ultralytics", "numpy", "cv2"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "khong cai"
    try:
        import torch

        out["torch_cuda_available"] = str(torch.cuda.is_available())
        out["torch_cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import onnxruntime as ort

        out["ort_providers"] = ort.get_available_providers()
    except Exception:
        pass
    return out


def collect(note: str = "") -> dict:
    """Toan bo metadata thiet bi cho mot phien do."""
    temps = temperatures()
    is_jetson = Path("/etc/nv_tegra_release").exists()
    info = {
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "is_jetson": is_jetson,
        "hostname": platform.node(),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "l4t_release": Path("/etc/nv_tegra_release").read_text(encoding="utf-8").splitlines()[0]
        if is_jetson else None,
        "board_model": Path("/proc/device-tree/model").read_text(encoding="utf-8").strip("\x00")
        if Path("/proc/device-tree/model").exists() else None,
        "jetpack": _sh("apt-cache policy nvidia-jetpack 2>/dev/null | awk '/Installed/{print $2}'")
        or _sh("apt-cache show nvidia-jetpack 2>/dev/null | awk '/^Version:/{print $2; exit}'"),
        "cuda": _sh("/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2 | head -1"),
        "power_mode": power_modes(),
        "jetson_clocks_locked": "1" in _sh("cat /sys/devices/system/cpu/cpu0/cpufreq/"
                                           "scaling_governor 2>/dev/null | grep -c performance"),
        "clocks": clocks(),
        "temperatures_c": temps,
        "temperature_mean_c": round(sum(temps.values()) / len(temps), 1) if temps else None,
        "memory": _mem(),
        "libraries": library_versions(),
    }
    return info


def _mem() -> dict:
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[k] = round(int(v.strip().split()[0]) / 1e6, 2)  # GB
    except Exception:
        pass
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="ghi ra file JSON")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    info = collect(a.note)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
