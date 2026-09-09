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


# --------------------------------------------------------------- encoding
def test_moi_cho_doc_ghi_text_deu_khai_bao_encoding():
    """Jetson chay locale C/POSIX: Python mac dinh dung ascii cho file va stdout.
    Mot dau '—' trong bao cao la du de lam vo ca phien benchmark.
    """
    import re

    bad = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        s = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\.(read|write)_text\(", s):
            k, depth = m.end(), 1
            start = k
            while depth:
                if s[k] == "(":
                    depth += 1
                elif s[k] == ")":
                    depth -= 1
                k += 1
            if "encoding=" not in s[start:k - 1]:
                bad.append(f"{p.relative_to(ROOT)}:{s[:m.start()].count(chr(10)) + 1}")
    assert not bad, "thieu encoding='utf-8' o:\n  " + "\n  ".join(bad)


def test_write_json_giu_duoc_ky_tu_tieng_viet(tmp_path):
    from ivid.data.common import write_json

    payload = {"ghi_chu": "Trên Jetson, CPU và GPU dùng chung DRAM — không phải VRAM rời",
               "do_lech": "±0.5°C", "ket_luan": "✅ đạt"}
    f = tmp_path / "x.json"
    write_json(f, payload)
    import json

    assert json.loads(f.read_text(encoding="utf-8")) == payload


# ---------------------------------------------------------------- warm-up
def test_warmup_chay_ca_duong_ong_khong_chi_inference():
    """Neu warm-up chi goi infer(), backend do TRUOC se ganh chi phi khoi tao
    OpenCV va trong nhu tien xu ly cua no cham hon — trong khi ca ba backend
    dung chung mot ham preprocess. Da xay ra that: 10.2 ms vs 5.6 ms.
    """
    import inspect

    from ivid.benchmark.runners.base import BaseRunner

    src = inspect.getsource(BaseRunner.warmup)
    assert "run_array" in src, "warmup phai goi run_array, khong duoc chi goi infer"
