import zipfile
from pathlib import Path

import pytest

from defect_detection.data.download import ChecksumMismatchError, sha256_file
from defect_detection.data.package import package_dataset, unpack_dataset


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "processed/widget"
    for relative in (
        "images/dent_000.png",
        "labels/dent_000.txt",
        "test.txt",
        "fold0_train.txt",
        "fold0.yaml",
        "preview/dent_gap0_area0.jpg",
        "labels.cache",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    return root


def test_package_contains_dataset_but_not_previews_or_caches(dataset_dir: Path, tmp_path: Path):
    zip_path = package_dataset(dataset_dir, tmp_path / "widget.zip", tmp_path / "checksums.json")

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [
            "widget/fold0.yaml",
            "widget/fold0_train.txt",
            "widget/images/dent_000.png",
            "widget/labels/dent_000.txt",
            "widget/test.txt",
        ]


def test_package_is_reproducible(dataset_dir: Path, tmp_path: Path):
    first = package_dataset(dataset_dir, tmp_path / "widget.zip", tmp_path / "c")
    first_hash = sha256_file(first)
    (dataset_dir / "images/dent_000.png").touch()  # cambia la fecha, no el contenido

    assert sha256_file(package_dataset(dataset_dir, first, tmp_path / "c")) == first_hash


def test_unpack_restores_the_dataset(dataset_dir: Path, tmp_path: Path):
    checksums = tmp_path / "checksums.json"
    zip_path = package_dataset(dataset_dir, tmp_path / "widget.zip", checksums)

    unpack_dataset(zip_path, tmp_path / "elsewhere", checksums)

    assert (tmp_path / "elsewhere/widget/images/dent_000.png").read_text() == "images/dent_000.png"
    assert not (tmp_path / "elsewhere/widget/preview").exists()


def test_unpack_rejects_a_zip_that_is_not_the_recorded_one(dataset_dir: Path, tmp_path: Path):
    checksums = tmp_path / "checksums.json"
    zip_path = package_dataset(dataset_dir, tmp_path / "widget.zip", checksums)
    zip_path.write_bytes(zip_path.read_bytes() + b"x")

    with pytest.raises(ChecksumMismatchError):
        unpack_dataset(zip_path, tmp_path / "elsewhere", checksums)

    assert not (tmp_path / "elsewhere").exists()


def test_unpack_requires_a_recorded_checksum(dataset_dir: Path, tmp_path: Path):
    zip_path = package_dataset(dataset_dir, tmp_path / "widget.zip", tmp_path / "checksums.json")

    with pytest.raises(KeyError, match="widget.zip"):
        unpack_dataset(zip_path, tmp_path / "elsewhere", tmp_path / "other.json")
