"""Genera el dataset en formato Ultralytics YOLO a partir de una categoría de MVTec AD.

Estructura de salida (``data/processed/<categoría>/``)::

    images/            todas las imágenes del split, en plano
    labels/            una etiqueta .txt por imagen (vacía si no tiene defecto)
    test.txt           lista de imágenes de test
    foldN_train.txt    listas de train y val cuando el fold N hace de validación
    foldN_val.txt
    foldN.yaml         dataset de Ultralytics para ese fold

Las particiones son listas de rutas y no carpetas para tener k folds sin duplicar las
imágenes. Las rutas empiezan por ``./`` y los YAML no fijan ``path``: Ultralytics las
resuelve respecto al propio fichero, así que el directorio se puede mover (p. ej. a Colab).
"""

import logging
import shutil
from pathlib import Path

import cv2
import yaml

from defect_detection.data.config import DataConfig
from defect_detection.data.masks import box_to_yolo, mask_to_boxes
from defect_detection.data.mvtec import GOOD, class_of, defect_classes, mask_path
from defect_detection.data.split import TEST, fold_name

logger = logging.getLogger(__name__)


def flat_name(relative_image: str) -> str:
    """Nombre único en plano: ``test/crack/000.png`` -> ``crack_000``.

    Las buenas llevan además su carpeta de origen (``good_train_000``) porque
    ``train/good`` y ``test/good`` repiten numeración.
    """
    origin, cls, filename = Path(relative_image).parts
    stem = Path(filename).stem
    return f"{GOOD}_{origin}_{stem}" if cls == GOOD else f"{cls}_{stem}"


def label_lines(mask_file: Path, class_id: int, config: DataConfig) -> list[str]:
    """Líneas YOLO (``clase cx cy w h``) de una imagen defectuosa."""
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask {mask_file}")
    boxes = mask_to_boxes(mask, config.merge_gap_px, config.min_area_px)
    if not boxes:
        # Sin cajas entraría en el dataset como imagen de fondo, es decir, como negativo falso.
        raise ValueError(f"No boxes left for defective image, check thresholds: {mask_file}")
    height, width = mask.shape
    return [
        f"{class_id} " + " ".join(f"{value:.6f}" for value in box_to_yolo(box, width, height))
        for box in boxes
    ]


def _write_list(path: Path, names: list[str]) -> None:
    path.write_text("".join(f"./images/{name}.png\n" for name in sorted(names)), encoding="utf-8")


def prepare_dataset(category_dir: Path, out_dir: Path, split: dict, config: DataConfig) -> None:
    """Escribe imágenes, etiquetas, listas y YAML del dataset según ``split``.

    Reconstruye la salida desde cero en cada ejecución, de modo que el resultado depende
    solo de los datos crudos, el split y la configuración.
    """
    classes = defect_classes(category_dir)
    images_dir, labels_dir = out_dir / "images", out_dir / "labels"
    for directory in (images_dir, labels_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
    for stale in [*out_dir.glob("*.txt"), *out_dir.glob("*.yaml"), *out_dir.glob("*.cache")]:
        stale.unlink()

    names: dict[str, list[str]] = {}
    boxes = 0
    for partition, images in split["partitions"].items():
        names[partition] = []
        for relative in images:
            source = category_dir / relative
            name = flat_name(relative)
            cls = class_of(source)
            lines = (
                [] if cls == GOOD else label_lines(mask_path(source), classes.index(cls), config)
            )
            shutil.copyfile(source, images_dir / f"{name}.png")
            (labels_dir / f"{name}.txt").write_text(
                "".join(f"{line}\n" for line in lines), encoding="utf-8"
            )
            names[partition].append(name)
            boxes += len(lines)

    _write_list(out_dir / "test.txt", names[TEST])
    folds = [fold_name(i) for i in range(split["folds"])]
    for fold in folds:
        train = [name for other in folds if other != fold for name in names[other]]
        _write_list(out_dir / f"{fold}_train.txt", train)
        _write_list(out_dir / f"{fold}_val.txt", names[fold])
        dataset = {
            "train": f"{fold}_train.txt",
            "val": f"{fold}_val.txt",
            "test": "test.txt",
            "names": dict(enumerate(classes)),
        }
        (out_dir / f"{fold}.yaml").write_text(
            yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
        )

    total = sum(len(partition) for partition in names.values())
    logger.info(
        "Prepared %s: %d images, %d boxes, %d folds, classes %s",
        out_dir,
        total,
        boxes,
        len(folds),
        classes,
    )
