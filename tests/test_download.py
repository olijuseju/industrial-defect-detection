import io
import json
import tarfile
from pathlib import Path

import pytest

from defect_detection.data.download import (
    ChecksumMismatchError,
    download_file,
    ensure_dataset,
    extract_categories,
    sha256_file,
    verify_or_record_checksum,
)


def make_tar(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return make_tar(
        tmp_path / "mini.tar",
        {
            "hazelnut/train/good/000.png": b"good",
            "hazelnut/test/crack/000.png": b"crack",
            "bottle/train/good/000.png": b"bottle",
            "readme.txt": b"readme",
        },
    )


def test_extract_only_requested_category(archive: Path, tmp_path: Path):
    raw_dir = tmp_path / "raw"

    extract_categories(archive, raw_dir, ["hazelnut"])

    assert (raw_dir / "hazelnut/test/crack/000.png").read_bytes() == b"crack"
    assert sorted(p.name for p in raw_dir.iterdir()) == ["hazelnut"]


def test_extract_missing_category_fails_and_leaves_nothing(archive: Path, tmp_path: Path):
    raw_dir = tmp_path / "raw"

    with pytest.raises(ValueError, match="zipper"):
        extract_categories(archive, raw_dir, ["hazelnut", "zipper"])

    assert list(raw_dir.iterdir()) == []


def test_extract_rejects_path_traversal(tmp_path: Path):
    evil = make_tar(tmp_path / "evil.tar", {"hazelnut/../../evil.txt": b"x"})
    raw_dir = tmp_path / "raw"

    with pytest.raises(tarfile.FilterError):
        extract_categories(evil, raw_dir, ["hazelnut"])

    assert not (tmp_path / "evil.txt").exists()
    assert list(raw_dir.iterdir()) == []


def test_checksum_recorded_then_verified(archive: Path, tmp_path: Path):
    checksums = tmp_path / "configs/checksums.json"

    recorded = verify_or_record_checksum(archive, checksums)

    assert json.loads(checksums.read_text()) == {"mini.tar": sha256_file(archive)}
    assert verify_or_record_checksum(archive, checksums) == recorded


def test_checksum_mismatch_raises(archive: Path, tmp_path: Path):
    checksums = tmp_path / "checksums.json"
    verify_or_record_checksum(archive, checksums)
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(ChecksumMismatchError):
        verify_or_record_checksum(archive, checksums)


def test_download_file_streams_to_dest(archive: Path, tmp_path: Path):
    dest = tmp_path / "archives/mini.tar"

    download_file(archive.as_uri(), dest)

    assert sha256_file(dest) == sha256_file(archive)
    assert not dest.with_name("mini.tar.part").exists()


def test_download_failure_leaves_no_partial_file(tmp_path: Path):
    dest = tmp_path / "archives/missing.tar"

    with pytest.raises(OSError):
        download_file((tmp_path / "does-not-exist.tar").as_uri(), dest)

    assert list(dest.parent.iterdir()) == []


def test_ensure_dataset_from_archive_is_idempotent(archive: Path, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    checksums = tmp_path / "checksums.json"

    ensure_dataset(raw_dir, checksums, categories=["hazelnut"], archive=archive)
    archive.unlink()
    # Con la categoría ya extraída no debe necesitar ni el archivo ni una URL.
    ensure_dataset(raw_dir, checksums, categories=["hazelnut"])

    assert (raw_dir / "hazelnut/train/good/000.png").exists()


def test_ensure_dataset_from_url_downloads_and_extracts(archive: Path, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    checksums = tmp_path / "checksums.json"

    ensure_dataset(raw_dir, checksums, categories=["hazelnut", "bottle"], url=archive.as_uri())

    assert (raw_dir / "archives/mini.tar").exists()
    assert (raw_dir / "hazelnut").is_dir()
    assert (raw_dir / "bottle").is_dir()
    assert "mini.tar" in json.loads(checksums.read_text())


def test_ensure_dataset_requires_a_source(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        ensure_dataset(tmp_path / "raw", tmp_path / "checksums.json", categories=["hazelnut"])
