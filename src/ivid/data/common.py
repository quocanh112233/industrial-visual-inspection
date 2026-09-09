from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def rel_to_root(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(repo_root()))
    except ValueError:
        return str(p)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = repo_root()
    for key in ("raw_dir", "out_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else root / p
    return cfg


def class_names(cfg: dict[str, Any]) -> list[str]:
    return [cfg["names"][i] for i in sorted(cfg["names"])]


def primary_class(image_path: Path) -> str:
    return image_path.stem.rsplit("_", 1)[0]


def sha256_of_files(paths: list[Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        with p.open("rb") as f:
            while block := f.read(chunk):
                h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
