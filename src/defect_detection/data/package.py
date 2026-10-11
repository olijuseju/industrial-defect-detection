"""Empaquetado del dataset procesado para llevarlo a otra máquina (p. ej. Colab).

El dataset no está en git, así que viaja como un zip. Su SHA256 se registra en el mismo
JSON que el del archivo de MVTec: quien lo desempaqueta comprueba que entrena con
exactamente los mismos datos que se generaron aquí.
"""

import logging
import zipfile
from pathlib import Path

from defect_detection.data.download import record_checksum, verify_checksum

logger = logging.getLogger(__name__)

_DIRECTORIES = ("images", "labels")
_ROOT_SUFFIXES = (".txt", ".yaml")
# Fecha fija en las entradas: el zip depende solo del contenido, no de cuándo se creó.
_FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def dataset_files(dataset_dir: Path) -> list[Path]:
    """Ficheros que forman el dataset; deja fuera previews y cachés de Ultralytics."""
    files = [p for name in _DIRECTORIES for p in (dataset_dir / name).rglob("*") if p.is_file()]
    files += [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix in _ROOT_SUFFIXES]
    return sorted(files)


def package_dataset(dataset_dir: Path, zip_path: Path, checksums_path: Path) -> Path:
    """Crea un zip reproducible de ``dataset_dir`` y registra su SHA256.

    Las rutas del zip empiezan por el nombre de la carpeta (``hazelnut/images/...``).
    """
    files = dataset_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No dataset files in {dataset_dir}; run scripts/prepare_data.py")

    part = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(part, "w") as archive:
        for file in files:
            name = (Path(dataset_dir.name) / file.relative_to(dataset_dir)).as_posix()
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
            info.external_attr = 0o644 << 16
            # Los PNG ya están comprimidos: recomprimirlos solo gasta tiempo.
            info.compress_type = (
                zipfile.ZIP_STORED if file.suffix == ".png" else zipfile.ZIP_DEFLATED
            )
            archive.writestr(info, file.read_bytes())
    part.replace(zip_path)
    logger.info("Packaged %d files into %s", len(files), zip_path)
    record_checksum(zip_path, checksums_path)
    return zip_path


def unpack_dataset(zip_path: Path, processed_dir: Path, checksums_path: Path) -> None:
    """Verifica el SHA256 de ``zip_path`` y lo extrae en ``processed_dir``."""
    verify_checksum(zip_path, checksums_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(processed_dir)
    logger.info("Unpacked %s into %s", zip_path, processed_dir)
