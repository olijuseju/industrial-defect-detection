import dataclasses
from pathlib import Path

import pytest

from defect_detection.data.config import DataConfig
from defect_detection.data.split import TEST, build_split, class_counts, load_or_create_split

CONFIG = DataConfig(
    category="widget",
    merge_gap_px=0,
    min_area_px=0,
    test_fraction=0.2,
    folds=4,
    seed=7,
    negative_ratio=1.0,
)


@pytest.fixture
def category_dir(tmp_path: Path) -> Path:
    """Árbol con 2 defectos x 10 imágenes y 40 buenas; el split solo mira las rutas."""
    root = tmp_path / "widget"
    for defect in ("dent", "scratch"):
        (root / "test" / defect).mkdir(parents=True)
        (root / "ground_truth" / defect).mkdir(parents=True)
        for i in range(10):
            (root / "test" / defect / f"{i:03}.png").touch()
            (root / "ground_truth" / defect / f"{i:03}_mask.png").touch()
    for split, count in (("train", 30), ("test", 10)):
        (root / split / "good").mkdir(parents=True)
        for i in range(count):
            (root / split / "good" / f"{i:03}.png").touch()
    return root


def all_images(split: dict) -> list[str]:
    return [image for images in split["partitions"].values() for image in images]


def test_split_is_deterministic_and_depends_on_seed(category_dir: Path):
    assert build_split(category_dir, CONFIG) == build_split(category_dir, CONFIG)
    other = dataclasses.replace(CONFIG, seed=8)
    assert (
        build_split(category_dir, other)["partitions"]
        != (build_split(category_dir, CONFIG)["partitions"])
    )


def test_every_image_is_assigned_exactly_once(category_dir: Path):
    images = all_images(build_split(category_dir, CONFIG))

    assert len(images) == len(set(images)) == 40  # 20 defectuosas + 20 negativos
    assert sum("good" not in image for image in images) == 20


def test_split_is_stratified_and_folds_are_balanced(category_dir: Path):
    counts = class_counts(build_split(category_dir, CONFIG))

    assert counts[TEST] == {"dent": 2, "scratch": 2, "good": 4}
    for fold in ("fold0", "fold1", "fold2", "fold3"):
        assert counts[fold] == {"dent": 2, "scratch": 2, "good": 4}


def test_negative_ratio_controls_number_of_good_images(category_dir: Path):
    split = build_split(category_dir, dataclasses.replace(CONFIG, negative_ratio=0.5))

    assert sum("/good/" in image for image in all_images(split)) == 10


def test_too_many_negatives_requested_fails(category_dir: Path):
    with pytest.raises(ValueError, match="only 40 exist"):
        build_split(category_dir, dataclasses.replace(CONFIG, negative_ratio=3.0))


def test_existing_split_file_is_reused(category_dir: Path, tmp_path: Path):
    path = tmp_path / "splits/widget.json"
    first = load_or_create_split(path, category_dir, CONFIG)
    (category_dir / "test/dent/000.png").unlink()  # el disco cambia, el split guardado no

    assert load_or_create_split(path, category_dir, CONFIG) == first


def test_existing_split_with_other_params_fails_unless_forced(category_dir: Path, tmp_path: Path):
    path = tmp_path / "widget.json"
    load_or_create_split(path, category_dir, CONFIG)
    changed = dataclasses.replace(CONFIG, folds=5)

    with pytest.raises(ValueError, match="folds"):
        load_or_create_split(path, category_dir, changed)

    assert load_or_create_split(path, category_dir, changed, force=True)["folds"] == 5


def test_config_loads_from_yaml(tmp_path: Path):
    path = tmp_path / "data.yaml"
    path.write_text(
        "category: widget\nmerge_gap_px: 0\nmin_area_px: 0\ntest_fraction: 0.2\n"
        "folds: 4\nseed: 7\nnegative_ratio: 1.0\n"
    )

    assert DataConfig.from_yaml(path) == CONFIG
