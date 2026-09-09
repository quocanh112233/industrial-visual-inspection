"""Kiem tra cac file cau hinh tu noi dung khop voi ten cua chinh no.

Co that: configs/train_yolov8s.yaml duoc tao bang 'sed' tu ban yolov8n, nhung
bieu thuc thay the neo '$' o cuoi dong trong khi dong that con comment phia sau
-> khong khop -> sed IM LANG khong thay the gi. Ket qua: file ten 'yolov8s'
nhung ben trong ghi 'model: yolov8n.pt', va mot phien train 37 phut cho ra
YOLOv8n thu ba thay vi YOLOv8s.

Loi bi che them mot lop nua vi notebook truyen '--name yolov8s', nen thu muc va
ten file ket qua deu trong dung. Chi co so tham so trong log la lo ra.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_CONFIGS = sorted((ROOT / "configs").glob("train_*.yaml"))
ARCH = re.compile(r"yolo(?:v)?\d+[nsmlx]")


def arch_of(text: str) -> str | None:
    m = ARCH.search(text)
    return m.group(0) if m else None


@pytest.mark.parametrize("cfg_path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_ten_run_khop_voi_ten_file(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    expected = cfg_path.stem.removeprefix("train_")
    assert cfg["name"] == expected, (
        f"{cfg_path.name}: name='{cfg['name']}' nhung ten file noi la '{expected}'")


@pytest.mark.parametrize("cfg_path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_trong_so_pretrained_khop_voi_kien_truc_trong_ten_file(cfg_path: Path):
    """train_yolov8s.yaml PHAI dung yolov8s.pt, khong duoc dung yolov8n.pt."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    want = arch_of(cfg_path.stem)
    got = arch_of(str(cfg["model"]))
    assert want and got, f"{cfg_path.name}: khong doc duoc kien truc"
    assert got == want, (
        f"{cfg_path.name}: ten file noi '{want}' nhung model='{cfg['model']}'. "
        "Day la loi da xay ra that va lam mat 37 phut train.")


@pytest.mark.parametrize("cfg_path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_config_train_co_du_khoa_bat_buoc(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for key in ("name", "model", "data", "seed", "train"):
        assert key in cfg, f"{cfg_path.name}: thieu khoa '{key}'"
    for key in ("epochs", "imgsz", "batch"):
        assert key in cfg["train"], f"{cfg_path.name}: thieu train.{key}"


@pytest.mark.parametrize("cfg_path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_moi_config_dung_cung_seed(cfg_path: Path):
    """Khac seed thi khong so sanh duoc hai model voi nhau."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["seed"] == 1337, f"{cfg_path.name}: seed={cfg['seed']}, mong doi 1337"


@pytest.mark.parametrize("cfg_path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_cache_khong_dung_ram(cfg_path: Path):
    """cache='ram' lam ultralytics canh bao ket qua khong tai lap duoc (FR-06)."""
    cache = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["train"].get("cache")
    assert cache != "ram", f"{cfg_path.name}: cache='ram' pha tinh tai lap"


def test_co_du_ca_hai_model_theo_SRS():
    names = {yaml.safe_load(p.read_text(encoding="utf-8"))["name"] for p in TRAIN_CONFIGS}
    assert {"yolov8n", "yolov8s"} <= names, f"thieu config, hien co: {sorted(names)}"


def test_config_data_va_export_co_thu_tu_lop_giong_nhau():
    """Thu tu 6 lop la hop dong: no quyet dinh class id trong nhan, ONNX va engine."""
    data = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    names = [data["names"][i] for i in sorted(data["names"])]
    assert names == ["crazing", "inclusion", "patches", "pitted_surface",
                     "rolled-in_scale", "scratches"]
