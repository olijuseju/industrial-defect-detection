from pathlib import Path

import cv2
import numpy as np
import pytest

from defect_detection.data.masks import box_to_yolo, mask_to_boxes
from defect_detection.data.mvtec import defect_classes, iter_defect_samples
from defect_detection.data.preview import write_contact_sheets


def blank_mask(size: int = 100) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


def test_single_blob_gives_tight_box():
    mask = blank_mask()
    mask[20:30, 40:60] = 255

    assert mask_to_boxes(mask) == [(40, 20, 20, 10)]


def test_empty_mask_gives_no_boxes():
    assert mask_to_boxes(blank_mask()) == []


def test_nearby_fragments_merge_into_one_tight_box():
    mask = blank_mask()
    mask[20:30, 10:20] = 255
    mask[20:30, 25:35] = 255  # hueco de 5 px

    assert len(mask_to_boxes(mask)) == 2
    # La caja fusionada abarca los píxeles originales, sin el margen de la dilatación.
    assert mask_to_boxes(mask, merge_gap_px=9) == [(10, 20, 25, 10)]


def test_distant_fragments_stay_separate():
    mask = blank_mask()
    mask[10:20, 10:20] = 255
    mask[70:80, 70:80] = 255

    assert mask_to_boxes(mask, merge_gap_px=9) == [(10, 10, 10, 10), (70, 70, 10, 10)]


def test_small_fragments_are_dropped():
    mask = blank_mask()
    mask[10:30, 10:30] = 255
    mask[80:82, 80:82] = 255  # 4 px

    assert mask_to_boxes(mask, min_area_px=5) == [(10, 10, 20, 20)]


def test_min_area_counts_original_pixels_of_merged_group():
    mask = blank_mask()
    mask[50:52, 50:52] = 255
    mask[50:52, 54:56] = 255  # dos fragmentos de 4 px: 8 px juntos

    assert mask_to_boxes(mask, min_area_px=6) == []
    assert mask_to_boxes(mask, merge_gap_px=5, min_area_px=6) == [(50, 50, 6, 2)]


def test_box_to_yolo_normalizes_center_and_size():
    assert box_to_yolo((40, 20, 20, 10), width=100, height=200) == pytest.approx(
        (0.5, 0.125, 0.2, 0.05)
    )


@pytest.fixture
def category_dir(tmp_path: Path) -> Path:
    root = tmp_path / "widget"
    for defect in ("scratch", "dent"):
        (root / "test" / defect).mkdir(parents=True)
        (root / "ground_truth" / defect).mkdir(parents=True)
        for stem in ("000", "001"):
            mask = blank_mask(64)
            mask[10:30, 10:30] = 255
            cv2.imwrite(str(root / "test" / defect / f"{stem}.png"), np.dstack([mask] * 3))
            cv2.imwrite(str(root / "ground_truth" / defect / f"{stem}_mask.png"), mask)
    return root


def test_iter_defect_samples_pairs_images_with_masks(category_dir: Path):
    samples = list(iter_defect_samples(category_dir))

    assert defect_classes(category_dir) == ["dent", "scratch"]
    assert [(d, i.name, m.name) for d, i, m in samples] == [
        ("dent", "000.png", "000_mask.png"),
        ("dent", "001.png", "001_mask.png"),
        ("scratch", "000.png", "000_mask.png"),
        ("scratch", "001.png", "001_mask.png"),
    ]


def test_iter_defect_samples_fails_on_missing_mask(category_dir: Path):
    (category_dir / "ground_truth/dent/001_mask.png").unlink()

    with pytest.raises(FileNotFoundError, match="001_mask.png"):
        list(iter_defect_samples(category_dir))


def test_write_contact_sheets_one_per_defect(category_dir: Path, tmp_path: Path):
    sheets = write_contact_sheets(category_dir, tmp_path / "preview", 0, 0, tile_px=32, columns=3)

    assert [p.name for p in sheets] == ["dent_gap0_area0.jpg", "scratch_gap0_area0.jpg"]
    assert cv2.imread(str(sheets[0])).shape == (32, 96, 3)
