from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..data.common import rel_to_root, repo_root, write_json


def lib_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "torchvision", "ultralytics", "numpy", "onnx", "onnxruntime", "tensorrt"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = "không cài"
    try:
        import torch

        out["cuda_available"] = str(torch.cuda.is_available())
        out["cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def git_commit(root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="vi du configs/train_yolov8n.yaml")
    ap.add_argument("--project", default=None,
                    help="thư mục chứa các run (Colab: /content/drive/MyDrive/ivid/runs)")
    ap.add_argument("--name", default=None, help="ghi đè tên run trong config")
    ap.add_argument("--device", default=None, help="0 | cpu | 0,1 (mặc định: ultralytics từ chon)")
    ap.add_argument("--resume", action="store_true", help="chạy tiếp từ last.pt của run cùng tên")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in cấu hình, không train")
    a = ap.parse_args()

    root = repo_root()
    cfg_path = Path(a.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        print(f"[lỗi] không thấy config: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    name = a.name or cfg["name"]
    project = Path(a.project) if a.project else root / "runs"
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = root / data_path
    if not data_path.exists():
        print(f"[lỗi] không thấy {data_path}\n"
              f"      chạy trước: make data", file=sys.stderr)
        return 1

    targs = dict(cfg.get("train", {}))
    targs.update(
        data=str(data_path),
        seed=int(cfg["seed"]),
        project=str(project),
        name=name,
        exist_ok=True,
        resume=a.resume,
    )
    if a.device is not None:
        targs["device"] = a.device

    os.environ.setdefault("PYTHONHASHSEED", str(cfg["seed"]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    print(f"[train] config  : {cfg_path.relative_to(root) if cfg_path.is_relative_to(root) else cfg_path}")
    print(f"[train] model   : {cfg['model']}")
    print(f"[train] data    : {data_path}")
    print(f"[train] run     : {project / name}")
    print("[train] tham số : " + " ".join(f"{k}={v}" for k, v in sorted(targs.items())
                                           if k in ("epochs", "imgsz", "batch", "seed", "amp", "cache")))
    if a.dry_run:
        print(json.dumps(targs, indent=2, default=str))
        return 0

    from ultralytics import YOLO

    started = datetime.now(timezone.utc)
    model = YOLO(cfg["model"])

    try:
        n_params = sum(x.numel() for x in model.model.parameters())
    except Exception:
        n_params = None
    if n_params:
        print(f"[train] tham số : {n_params:,}")

    model.train(**targs)
    finished = datetime.now(timezone.utc)

    run_dir = project / name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[lỗi] không sinh ra {best}", file=sys.stderr)
        return 1

    dest = root / "models" / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest / "best.pt")

    ds_manifest = root / "results" / "dataset_manifest.json"
    manifest = {
        "run_name": name,
        "config_file": str(cfg_path.relative_to(root)) if cfg_path.is_relative_to(root) else str(cfg_path),
        "config": cfg,
        "train_args": {k: str(v) for k, v in sorted(targs.items())},
        "seed": int(cfg["seed"]),
        "pretrained_weights": str(cfg["model"]),
        "n_parameters": n_params,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "duration_minutes": round((finished - started).total_seconds() / 60, 1),
        "libraries": lib_versions(),
        "git_commit": git_commit(root),
        "dataset": json.loads(ds_manifest.read_text(encoding="utf-8")) if ds_manifest.exists() else "thieu",
        "weights": {
            "best": rel_to_root(dest / "best.pt"),
            "size_mb": round((dest / "best.pt").stat().st_size / 1e6, 2),
        },
        "run_dir": rel_to_root(run_dir),
    }
    write_json(root / "results" / f"train_{name}_manifest.json", manifest)

    csv = run_dir / "results.csv"
    if csv.exists():
        shutil.copy2(csv, root / "results" / f"train_{name}_curve.csv")

    print(f"\n[train] xong sau {manifest['duration_minutes']} phut")
    print(f"[train] best.pt -> models/{name}/best.pt ({manifest['weights']['size_mb']} MB)")
    print(f"[train] manifest -> results/train_{name}_manifest.json")
    print(f"[train] buoc tiep: PYTHONPATH=src python -m ivid.train.evaluate --name {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
