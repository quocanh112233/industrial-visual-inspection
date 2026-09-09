"""FR-12 — Do RAM va bo nho GPU dinh diem trong luc chay, va kich thuoc model.

Tren Jetson, CPU va GPU dung CHUNG mot vung DRAM vat ly. Vi vay "GPU memory"
o day khong phai VRAM roi nhu tren card PCIe — no la phan bo nho ma driver cap
cho ngu canh CUDA, nam trong cung 8GB. Bao cao phai noi ro dieu do, neu khong
nguoi doc se so sanh nham voi so lieu cua may ban.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _system_used_mb() -> float:
    try:
        vals = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            k, v = line.split(":", 1)
            vals[k] = int(v.strip().split()[0]) / 1024.0
        return round(vals.get("MemTotal", 0) - vals.get("MemAvailable", 0), 1)
    except Exception:
        return 0.0


def _torch_gpu_mb() -> tuple[float, float]:
    """(dang cap phat, dinh diem) theo MB — chi co y nghia voi runner dung torch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (round(torch.cuda.memory_allocated() / 1e6, 1),
                round(torch.cuda.max_memory_allocated() / 1e6, 1))
    except Exception:
        return 0.0, 0.0


class ResourceMonitor:
    """Lay mau bo nho trong nen trong khi benchmark chay.

        with ResourceMonitor() as rm:
            ...chay benchmark...
        print(rm.summary())
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_rss: list[float] = []
        self.samples_sys: list[float] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples_rss.append(_rss_mb())
            self.samples_sys.append(_system_used_mb())
            self._stop.wait(self.interval)

    def __enter__(self) -> ResourceMonitor:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._baseline_sys = _system_used_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        alloc, peak = _torch_gpu_mb()
        return {
            "process_rss_peak_mb": round(max(self.samples_rss), 1) if self.samples_rss else 0.0,
            "process_rss_mean_mb": round(sum(self.samples_rss) / len(self.samples_rss), 1)
            if self.samples_rss else 0.0,
            "system_used_peak_mb": round(max(self.samples_sys), 1) if self.samples_sys else 0.0,
            "system_used_baseline_mb": round(self._baseline_sys, 1),
            "system_used_delta_mb": round(max(self.samples_sys) - self._baseline_sys, 1)
            if self.samples_sys else 0.0,
            "torch_gpu_allocated_mb": alloc,
            "torch_gpu_peak_mb": peak,
            "n_samples": len(self.samples_rss),
            "ghi_chu": "Tren Jetson, CPU va GPU dung chung DRAM — 'gpu_peak' nam trong "
                       "cung tong bo nho he thong, khong phai VRAM roi.",
        }


def model_size_mb(path: Path) -> float:
    return round(Path(path).stat().st_size / 1e6, 2)


def measure_idle(seconds: float = 2.0) -> dict:
    """Muc bo nho nen truoc khi nap model, de tru ra."""
    end = time.time() + seconds
    sys_s, rss_s = [], []
    while time.time() < end:
        sys_s.append(_system_used_mb())
        rss_s.append(_rss_mb())
        time.sleep(0.1)
    return {"system_used_mb": round(sum(sys_s) / len(sys_s), 1),
            "process_rss_mb": round(sum(rss_s) / len(rss_s), 1)}
