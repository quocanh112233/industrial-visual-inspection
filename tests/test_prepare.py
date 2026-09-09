"""Test cho buoc chuan bi du lieu (FR-01).

Cac test khong can dataset that: chung tu sinh mot dataset gia nho, nen chay
duoc tren CI (noi khong tai NEU-DET ve).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ivid.data.common import primary_class, sha256_of_files  # noqa: E402
from ivid.data.prepare import stratified_split, voc_to_yolo  # noqa: E402

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def fake_images(per_class: int = 100) -> list[Path]:
    return [Path(f"{c}_{i}.jpg") for c in CLASSES for i in range(1, per_class + 1)]


def test_primary_class_tach_dung_ten_co_gach_ngang():
    # 'rolled-in_scale_42.jpg' de bay: co ca '-' va nhieu '_'
    assert primary_class(Path("rolled-in_scale_42.jpg")) == "rolled-in_scale"
    assert primary_class(Path("crazing_1.jpg")) == "crazing"
    assert primary_class(Path("pitted_surface_300.jpg")) == "pitted_surface"


def test_chia_tap_lap_lai_duoc():
    """FR-01: chay hai lan cho ra cung mot cach chia."""
    imgs = fake_images()
    a = stratified_split(imgs, RATIOS, seed=1337, stratify=True)
    b = stratified_split(imgs, RATIOS, seed=1337, stratify=True)
    assert a == b


def test_doi_seed_thi_doi_cach_chia():
    imgs = fake_images()
    a = stratified_split(imgs, RATIOS, seed=1337, stratify=True)
    b = stratified_split(imgs, RATIOS, seed=7, stratify=True)
    assert a != b


def test_khong_mat_anh_va_khong_trung_giua_cac_tap():
    """FR-01: tong so anh khop, khong co anh nao nam o hai tap."""
    imgs = fake_images()
    s = stratified_split(imgs, RATIOS, seed=1337, stratify=True)
    got = [p for v in s.values() for p in v]
    assert len(got) == len(imgs)
    assert set(got) == set(imgs)
    for k1 in s:
        for k2 in s:
            if k1 < k2:
                assert not set(s[k1]) & set(s[k2])


def test_chia_deu_tung_lop_khi_bat_stratify():
    s = stratified_split(fake_images(100), RATIOS, seed=1337, stratify=True)
    for split, expect in (("train", 70), ("val", 15), ("test", 15)):
        for c in CLASSES:
            n = sum(1 for p in s[split] if primary_class(p) == c)
            assert n == expect, f"{split}/{c}: mong {expect}, duoc {n}"


def test_thu_tu_dau_vao_khong_anh_huong_ket_qua():
    """Thu tu duyet thu muc khac nhau giua cac may khong duoc lam doi cach chia."""
    imgs = fake_images()
    a = stratified_split(imgs, RATIOS, seed=1337, stratify=True)
    b = stratified_split(list(reversed(imgs)), RATIOS, seed=1337, stratify=True)
    assert a == b


def test_voc_to_yolo_doi_dung_toa_do(tmp_path: Path):
    xml = tmp_path / "crazing_1.xml"
    xml.write_text(
        """<annotation><size><width>200</width><height>200</height></size>
        <object><name>scratches</name>
          <bndbox><xmin>50</xmin><ymin>60</ymin><xmax>150</xmax><ymax>160</ymax></bndbox>
        </object></annotation>"""
    )
    lines = voc_to_yolo(xml, CLASSES)
    assert len(lines) == 1
    cid, cx, cy, bw, bh = lines[0].split()
    assert int(cid) == CLASSES.index("scratches") == 5
    assert float(cx) == pytest.approx(0.5)     # (50+150)/2/200
    assert float(cy) == pytest.approx(0.55)    # (60+160)/2/200
    assert float(bw) == pytest.approx(0.5)
    assert float(bh) == pytest.approx(0.5)


def test_voc_to_yolo_kep_box_vuot_bien(tmp_path: Path):
    xml = tmp_path / "x_1.xml"
    xml.write_text(
        """<annotation><size><width>200</width><height>200</height></size>
        <object><name>crazing</name>
          <bndbox><xmin>-10</xmin><ymin>0</ymin><xmax>250</xmax><ymax>200</ymax></bndbox>
        </object></annotation>"""
    )
    cid, cx, cy, bw, bh = voc_to_yolo(xml, CLASSES)[0].split()
    assert float(cx) - float(bw) / 2 >= -1e-6
    assert float(cx) + float(bw) / 2 <= 1 + 1e-6


def test_voc_to_yolo_bao_loi_khi_lop_la(tmp_path: Path):
    xml = tmp_path / "x_1.xml"
    xml.write_text(
        """<annotation><size><width>200</width><height>200</height></size>
        <object><name>khong_ton_tai</name>
          <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
        </object></annotation>"""
    )
    with pytest.raises(ValueError, match="khong_ton_tai"):
        voc_to_yolo(xml, CLASSES)


def test_hash_khong_phu_thuoc_thu_tu(tmp_path: Path):
    files = []
    for name, body in (("b.txt", "bbb"), ("a.txt", "aaa"), ("c.txt", "ccc")):
        p = tmp_path / name
        p.write_text(body)
        files.append(p)
    assert sha256_of_files(files) == sha256_of_files(list(reversed(files)))


# ------------------------------------------------------------------ preprocess
def test_letterbox_giu_ti_le_va_ra_dung_kich_thuoc():
    """Anh khong vuong phai duoc dem vien, khong bi keo gian."""
    pytest.importorskip("cv2")
    import numpy as np

    from ivid.preprocess import letterbox

    img = np.zeros((100, 200, 3), dtype=np.uint8)   # rong gap doi cao
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape[:2] == (640, 640)
    assert r == pytest.approx(3.2)                   # 640/200
    assert px == 0 and py == 160                     # dem tren duoi, khong dem trai phai


def test_preprocess_ra_dung_dinh_dang_model_can():
    pytest.importorskip("cv2")
    import numpy as np

    from ivid.preprocess import preprocess

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    x, r, pad = preprocess(img, 640)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert r == pytest.approx(3.2)


def test_preprocess_doi_bgr_sang_rgb():
    pytest.importorskip("cv2")
    import numpy as np

    from ivid.preprocess import preprocess

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255                               # kenh B day
    x, _, _ = preprocess(img, 640)
    # sau khi doi sang RGB, kenh 2 (B) moi la kenh day
    assert x[0, 2].mean() == pytest.approx(1.0, abs=1e-3)
    assert x[0, 0].mean() == pytest.approx(0.0, abs=1e-3)
