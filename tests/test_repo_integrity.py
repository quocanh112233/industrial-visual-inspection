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


@pytest.mark.skipif(not in_git_repo(), reason="không phải git repo")
def test_moi_file_nguon_deu_duoc_git_theo_doi():
    tracked = set(git("ls-files").splitlines())
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")}
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "Các file nguồn sau tồn tại trên đĩa nhưng git KHÔNG theo dõi — "
        "rất có thể bị .gitignore chặn nhầm:\n  " + "\n  ".join(missing)
        + "\n\nKiểm tra bằng: git check-ignore -v <file>"
    )


@pytest.mark.skipif(not in_git_repo(), reason="không phải git repo")
def test_gitignore_khong_chan_nham_thu_muc_ma_nguon():
    for name in ("src/ivid/data", "src/ivid/train", "src/ivid/export",
                 "src/ivid/benchmark", "src/ivid/serve", "tests", "configs", "scripts"):
        out = git("check-ignore", "-v", f"{name}/x.py")
        assert not out.strip(), f"'{name}' bị .gitignore chặn: {out.strip()}"


@pytest.mark.skipif(not in_git_repo(), reason="không phải git repo")
def test_gitignore_van_chan_du_lieu_va_trong_so():
    for path in ("data/raw/x.jpg", "data/processed/data.yaml",
                 "models/yolov8n/best.pt", "models/yolov8n/best.engine"):
        assert git("check-ignore", "-v", path).strip(), f"'{path}' PHẢI bị chặn"


def test_moi_thu_muc_con_cua_ivid_deu_la_package():
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
    importlib.import_module(mod)


def test_moi_cho_doc_ghi_text_deu_khai_bao_encoding():
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


def test_warmup_chay_ca_duong_ong_khong_chi_inference():
    import inspect

    from ivid.benchmark.runners.base import BaseRunner

    src = inspect.getsource(BaseRunner.warmup)
    assert "run_array" in src, "warmup phải gọi run_array, không được chỉ gọi infer"


def _doc_files() -> list[Path]:
    files = list((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]
    files += list((ROOT / "notebooks").glob("*.ipynb"))
    return [f for f in files if f.exists()]


def _commands_in_docs() -> tuple[set[str], set[str]]:
    import re

    mods, scripts = set(), set()
    for md in _doc_files():
        t = md.read_text(encoding="utf-8")
        mods |= set(re.findall(r"python3? -m (ivid\.[a-zA-Z0-9_.]+)", t))
        scripts |= set(re.findall(r"bash (scripts/[a-zA-Z0-9_]+\.sh)", t))
    return mods, scripts


def test_tai_lieu_khong_tro_toi_module_da_bien_mat():
    mods, _ = _commands_in_docs()
    missing = [m for m in mods
               if not (ROOT / "src" / (m.replace(".", "/") + ".py")).exists()]
    assert not missing, "tài liệu gọi module không tồn tại: " + ", ".join(sorted(missing))


def test_tai_lieu_khong_tro_toi_script_da_bien_mat():
    _, scripts = _commands_in_docs()
    missing = [s for s in scripts if not (ROOT / s).exists()]
    assert not missing, "tài liệu gọi script không tồn tại: " + ", ".join(sorted(missing))


def test_moi_config_duoc_nhac_trong_tai_lieu_deu_ton_tai():
    import re

    refs = set()
    for md in _doc_files():
        refs |= set(re.findall(r"(configs/[a-zA-Z0-9_]+\.yaml)",
                               md.read_text(encoding="utf-8")))
    missing = [r for r in refs if not (ROOT / r).exists()]
    assert not missing, "tài liệu nhắc config không tồn tại: " + ", ".join(sorted(missing))


def test_makefile_target_duoc_nhac_trong_tai_lieu_deu_ton_tai():
    import re

    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-z][a-z-]*):", mk, re.M))
    refs = set()
    for md in _doc_files():
        refs |= set(re.findall(r"\bmake ([a-z][a-z-]*)", md.read_text(encoding="utf-8")))
    missing = sorted(refs - targets)
    assert not missing, "tài liệu gọi 'make' target không có: " + ", ".join(missing)


def test_notebook_colab_hop_le_va_bat_gpu():
    import json

    p = ROOT / "notebooks" / "ivid_colab.ipynb"
    assert p.exists(), "thieu notebooks/ivid_colab.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"].get("accelerator") == "GPU", "notebook phải khai báo accelerator GPU"
    assert any(c["cell_type"] == "code" for c in nb["cells"])
    text = p.read_text(encoding="utf-8")
    for leak in ("ghp_", "github_pat_", "AIza", "-----BEGIN"):
        assert leak not in text, f"notebook chưa chuoi giong secret: {leak}"


def test_khong_co_duong_dan_tuyet_doi_trong_file_duoc_commit():
    import re

    if not in_git_repo():
        pytest.skip("không phải git repo")

    tracked = set(git("ls-files").splitlines())
    suspicious = re.compile(r"(/home/|/Users/|C:\\\\)")
    bad = []
    for rel in sorted(tracked):
        if not (rel.startswith(("results/", "docs/")) and rel.endswith((".json", ".md"))):
            continue
        if "train_" in rel and "manifest" in rel:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in suspicious.finditer(text):
            line = text[:m.start()].count("\n") + 1
            bad.append(f"{rel}:{line}")
            break
    assert not bad, ("đường dẫn tuyệt đối trong file được commit:\n  " + "\n  ".join(bad)
                     + "\n\nDùng ivid.data.common.rel_to_root() khi ghi đường dẫn.")


def test_rel_to_root_tra_ve_duong_dan_tuong_doi():
    from ivid.data.common import rel_to_root

    assert rel_to_root(ROOT / "results" / "x.json") == "results/x.json"
    assert rel_to_root(ROOT / "src/ivid/data/prepare.py") == "src/ivid/data/prepare.py"
    assert rel_to_root("/tmp/ngoai-repo.json") == "/tmp/ngoai-repo.json"
