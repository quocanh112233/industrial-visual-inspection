"""Tien ich dung chung cho buoc chuan bi du lieu."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def repo_root() -> Path:
    """Thu muc goc repo (chua configs/ va src/)."""
    return Path(__file__).resolve().parents[3]


def rel_to_root(path: str | Path) -> str:
    """Duong dan tuong doi so voi goc repo, neu no nam trong repo.

    Cac file JSON/Markdown trong results/ va docs/ DUOC COMMIT. Neu chung chua
    duong dan tuyet doi thi may dev ghi '/home/quocanh/...' con Jetson ghi
    '/home/jetson/...' — hai may sinh ra hai noi dung khac nhau tu cung mot du
    lieu, va 'git pull' bao xung dot. Da xay ra that.
    """
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
    """names: {0: a, 1: b} -> [a, b], theo dung thu tu id."""
    return [cfg["names"][i] for i in sorted(cfg["names"])]


def primary_class(image_path: Path) -> str:
    """NEU-DET dat ten <lop>_<so>.jpg — tien to la lop chinh cua anh.

    Dung de chia tap co phan tang (stratify). Mot anh co the chua nhieu loai
    loi; day chi la lop dai dien de chia cho deu, khong phai nhan.
    """
    return image_path.stem.rsplit("_", 1)[0]


def sha256_of_files(paths: list[Path], chunk: int = 1 << 20) -> str:
    """Hash on dinh cua mot tap file: phu thuoc ten tuong doi + noi dung,
    khong phu thuoc thu tu duyet thu muc cua he dieu hanh (FR-06)."""
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
