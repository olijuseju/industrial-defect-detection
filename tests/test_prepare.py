import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from defect_detection.data.config import DataConfig
from defect_detection.data.prepare import flat_name, prepare_dataset
from defect_detection.data.split import build_split

CONFIG = DataConfig(
    category="widget",
    merge_gap_px=0,
    min_area_px=0,
    test_fraction=0.25,
    folds=2,
    seed=3,
    negative_ratio=1.0,
)


@pytest.fixture
def category_dir(tmp_path: Path) -> Path:
    """2 defectos x 4 imágenes de 64x64 con un defecto en [10:30, 10:30], y 12 buenas."""
    root = tmp_path / "raw/widget"
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    image = np.dstack([mask] * 3)
    for defect in ("dent", "scratch"):
        (root / "test" / defect).mkdir(parents=True)
        (root / "ground_truth" / defect).mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(str(root / "test" / defect / f"{i:03}.png"), image)
            cv2.imwrite(str(root / "ground_truth" / defect / f"{i:03}_mask.png"), mask)
    for origin, count in (("train", 8), ("test", 4)):
        (root / origin / "good").mkdir(parents=True)
        for i in range(count):
            cv2.imwrite(str(root / origin / "good" / f"{i:03}.png"), np.zeros_like(image))
    return root


def read_list(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_flat_name_keeps_good_origins_apart():
    assert flat_name("test/crack/000.png") == "crack_000"
    assert flat_name("train/good/000.png") == "good_train_000"
    assert flat_name("test/good/000.png") == "good_test_000"


def test_prepare_writes_images_and_yolo_labels(category_dir: Path, tmp_path: Path):
    out = tmp_path / "processed/widget"

    prepare_dataset(category_dir, out, build_split(category_dir, CONFIG), CONFIG)

    images = sorted(p.stem for p in (out / "images").iterdir())
    labels = sorted(p.stem for p in (out / "labels").iterdir())
    assert images == labels
    assert len(images) == 16  # 8 defectuosas + 8 negativos
    # Caja (10, 10, 20, 20) en 64x64; "scratch" es la clase 1 por orden alfabético.
    assert (out / "labels/scratch_000.txt").read_text() == "1 0.312500 0.312500 0.312500 0.312500\n"
    assert (out / "labels/dent_000.txt").read_text().startswith("0 ")
    good = next(p for p in (out / "labels").iterdir() if p.name.startswith("good_"))
    assert good.read_text() == ""


def test_prepare_writes_fold_lists_and_dataset_yaml(category_dir: Path, tmp_path: Path):
    out = tmp_path / "processed/widget"

    prepare_dataset(category_dir, out, build_split(category_dir, CONFIG), CONFIG)

    test = read_list(out / "test.txt")
    train0, val0 = read_list(out / "fold0_train.txt"), read_list(out / "fold0_val.txt")
    assert len(test) == 4 and len(train0) == 6 and len(val0) == 6
    assert not set(test) & set(train0) and not set(test) & set(val0) and not set(train0) & set(val0)
    # Con dos folds, lo que valida en uno entrena en el otro.
    assert val0 == read_list(out / "fold1_train.txt")
    assert all((out / line).is_file() for line in test + train0 + val0)
    assert yaml.safe_load((out / "fold0.yaml").read_text()) == {
        "train": "fold0_train.txt",
        "val": "fold0_val.txt",
        "test": "test.txt",
        "names": {0: "dent", 1: "scratch"},
    }


def test_defective_image_without_boxes_fails(category_dir: Path, tmp_path: Path):
    strict = dataclasses.replace(CONFIG, min_area_px=10_000)

    with pytest.raises(ValueError, match="No boxes left"):
        prepare_dataset(category_dir, tmp_path / "out", build_split(category_dir, CONFIG), strict)


def test_prepare_rebuilds_from_scratch(category_dir: Path, tmp_path: Path):
    out = tmp_path / "processed/widget"
    split = build_split(category_dir, CONFIG)
    prepare_dataset(category_dir, out, split, CONFIG)
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    (out / "images/stale.png").touch()
    (out / "fold9.yaml").touch()

    prepare_dataset(category_dir, out, split, CONFIG)

    after = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert after == before
