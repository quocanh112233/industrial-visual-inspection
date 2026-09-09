"""Bat cac loi lam ma nguon khong den duoc may khac.

Co that: '.gitignore' tung co dong 'data/' (khong dau '/' o dau). Git hieu do la
"moi thu muc ten data o moi do sau", nen no chan luon src/ivid/data/ — toan bo
FR-01..03 khong bao gio duoc commit. Tren may dev moi thu chay binh thuong;
tren Jetson thi 'ModuleNotFoundError: No module named ivid.data'.

Loai loi nay khong bao gio lo ra khi test tren may da co san file.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, timeout=30).stdout


def in_git_repo() -> bool:
    return (ROOT / ".git").exists()


@pytest.mark.skipif(not in_git_repo(), reason="khong phai git repo")
def test_moi_file_nguon_deu_duoc_git_theo_doi():
    tracked = set(git("ls-files").splitlines())
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")}
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "Cac file nguon sau ton tai tren dia nhung git KHONG theo doi — "
        "rat co the bi .gitignore chan nham:\n  " + "\n  ".join(missing)
        + "\n\nKiem tra bang: git check-ignore -v <file>"
    )


@pytest.mark.skipif(not in_git_repo(), reason="khong phai git repo")
def test_gitignore_khong_chan_nham_thu_muc_ma_nguon():
    for name in ("src/ivid/data", "src/ivid/train", "src/ivid/export",
                 "src/ivid/benchmark", "src/ivid/serve", "tests", "configs", "scripts"):
        out = git("check-ignore", "-v", f"{name}/x.py")
        assert not out.strip(), f"'{name}' bi .gitignore chan: {out.strip()}"


@pytest.mark.skipif(not in_git_repo(), reason="khong phai git repo")
def test_gitignore_van_chan_du_lieu_va_trong_so():
    """Sua loi tren khong duoc lam mat tac dung chan file nang."""
    for path in ("data/raw/x.jpg", "data/processed/data.yaml",
                 "models/yolov8n/best.pt", "models/yolov8n/best.engine"):
        assert git("check-ignore", "-v", path).strip(), f"'{path}' PHAI bi chan"


def test_moi_thu_muc_con_cua_ivid_deu_la_package():
    """Thieu __init__.py thi import se hong theo cach kho doan."""
    missing = [str(d.relative_to(ROOT)) for d in (ROOT / "src/ivid").rglob("*")
               if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists()]
    assert not missing, "thieu __init__.py o: " + ", ".join(missing)


@pytest.mark.parametrize("mod", [
    "ivid.preprocess", "ivid.postprocess",
    "ivid.data.common", "ivid.data.prepare", "ivid.data.validate", "ivid.data.stats",
    "ivid.benchmark.device_info", "ivid.benchmark.resources", "ivid.benchmark.report",
    "ivid.benchmark.runners.base",
    "ivid.serve.schemas", "ivid.serve.backends",
])
def test_module_import_duoc_khong_can_gpu(mod):
    """Cac module nay phai import duoc tren may khong co torch/tensorrt —
    neu khong, CI khong chay duoc va Docker build cung se hong."""
    importlib.import_module(mod)
