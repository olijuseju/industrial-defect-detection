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


def iter_defect_samples(category_dir: Path) -> Iterator[tuple[str, Path, Path]]:
    """Recorre las imágenes defectuosas como ``(defecto, imagen, máscara)``."""
    for defect in defect_classes(category_dir):
        for image in sorted((category_dir / "test" / defect).glob("*.png")):
            mask = category_dir / "ground_truth" / defect / f"{image.stem}_mask.png"
            if not mask.is_file():
                raise FileNotFoundError(f"Missing mask for {image}: {mask}")
            yield defect, image, mask
