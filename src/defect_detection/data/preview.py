"""Hojas de contacto con las cajas pintadas, para elegir los umbrales de ``mask_to_boxes``."""

import logging
from pathlib import Path

import cv2
import numpy as np

from defect_detection.data.masks import Box, mask_to_boxes
from defect_detection.data.mvtec import iter_defect_samples

logger = logging.getLogger(__name__)

_BOX_COLOR = (0, 255, 0)
_MASK_COLOR = (0, 0, 255)


def draw_labels(image: np.ndarray, mask: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Pinta sobre una copia de la imagen el contorno de la máscara y las cajas."""
    out = image.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, _MASK_COLOR, 2)
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), _BOX_COLOR, 3)
    return out


def write_contact_sheets(
    category_dir: Path,
    out_dir: Path,
    merge_gap_px: int,
    min_area_px: int,
    tile_px: int = 384,
    columns: int = 6,
) -> list[Path]:
    """Escribe una hoja por tipo de defecto con todas sus imágenes etiquetadas."""
    tiles: dict[str, list[np.ndarray]] = {}
    for defect, image_path, mask_path in iter_defect_samples(category_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        boxes = mask_to_boxes(mask, merge_gap_px, min_area_px)
        if not boxes:
            logger.warning("No boxes left for %s", image_path)
        tile = cv2.resize(draw_labels(image, mask, boxes), (tile_px, tile_px))
        caption = f"{image_path.stem}: {len(boxes)}"
        cv2.putText(tile, caption, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.setdefault(defect, []).append(tile)

    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for defect, defect_tiles in tiles.items():
        rows = -(-len(defect_tiles) // columns)
        blank = np.zeros_like(defect_tiles[0])
        padded = defect_tiles + [blank] * (rows * columns - len(defect_tiles))
        sheet = np.vstack([np.hstack(padded[r * columns : (r + 1) * columns]) for r in range(rows)])
        path = out_dir / f"{defect}_gap{merge_gap_px}_area{min_area_px}.jpg"
        cv2.imwrite(str(path), sheet)
        logger.info("Wrote %s (%d images)", path, len(defect_tiles))
        sheets.append(path)
    return sheets
