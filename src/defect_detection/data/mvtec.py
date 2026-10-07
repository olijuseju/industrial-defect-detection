"""Estructura en disco de una categoría de MVTec AD.

``<categoría>/test/<defecto>/000.png`` tiene su máscara en
``<categoría>/ground_truth/<defecto>/000_mask.png``. Las imágenes sin defecto están en
``train/good`` y ``test/good`` y no tienen máscara.
"""

from collections.abc import Iterator
from pathlib import Path

GOOD = "good"


def defect_classes(category_dir: Path) -> list[str]:
    """Tipos de defecto de la categoría, en orden alfabético."""
    return sorted(p.name for p in (category_dir / "ground_truth").iterdir() if p.is_dir())


def mask_path(image: Path) -> Path:
    """Ruta de la máscara de una imagen defectuosa ``<categoría>/test/<defecto>/<n>.png``."""
    category_dir = image.parents[2]
    return category_dir / "ground_truth" / image.parent.name / f"{image.stem}_mask.png"


def iter_defect_samples(category_dir: Path) -> Iterator[tuple[str, Path, Path]]:
    """Recorre las imágenes defectuosas como ``(defecto, imagen, máscara)``."""
    for defect in defect_classes(category_dir):
        for image in sorted((category_dir / "test" / defect).glob("*.png")):
            mask = mask_path(image)
            if not mask.is_file():
                raise FileNotFoundError(f"Missing mask for {image}: {mask}")
            yield defect, image, mask


def good_images(category_dir: Path) -> list[Path]:
    """Imágenes sin defecto de la categoría (``train/good`` y ``test/good``)."""
    return sorted(
        image
        for split in ("train", "test")
        for image in (category_dir / split / GOOD).glob("*.png")
    )


def class_of(image: Path) -> str:
    """Clase de una imagen: el nombre de su carpeta (un tipo de defecto o ``good``)."""
    return image.parent.name
