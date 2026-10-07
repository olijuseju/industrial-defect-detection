"""Conversión de máscaras de segmentación de MVTec AD a cajas para YOLO."""

import cv2
import numpy as np

# Caja en píxeles: (x, y, ancho, alto), con (x, y) la esquina superior izquierda.
Box = tuple[int, int, int, int]


def mask_to_boxes(mask: np.ndarray, merge_gap_px: int = 0, min_area_px: int = 0) -> list[Box]:
    """Devuelve una caja por cada defecto de una máscara binaria (0 = fondo).

    Un mismo defecto puede venir partido en varios fragmentos. Los que distan menos de
    ``merge_gap_px`` se agrupan dilatando la máscara antes de buscar componentes conexas.
    La caja se calcula con los píxeles originales del grupo, así que no se infla con la
    dilatación. Se descartan los grupos con menos de ``min_area_px`` píxeles originales.
    """
    binary = (mask > 0).astype(np.uint8)
    grouped = binary
    if merge_gap_px > 0:
        size = merge_gap_px | 1  # el núcleo debe ser impar para quedar centrado
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        grouped = cv2.dilate(binary, kernel)

    count, labels = cv2.connectedComponents(grouped, connectivity=8)
    labels[binary == 0] = 0

    boxes: list[Box] = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < max(min_area_px, 1):
            continue
        x, y = int(xs.min()), int(ys.min())
        boxes.append((x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1))
    return boxes


def box_to_yolo(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    """Convierte una caja en píxeles a ``(cx, cy, w, h)`` normalizado a [0, 1]."""
    x, y, w, h = box
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
