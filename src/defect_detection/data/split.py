"""Split reproducible de una categoría: test fijo estratificado + k folds.

Con ~17 imágenes defectuosas por clase, un único train/val/test deja 2-3 por clase en val
y las métricas salen muy ruidosas. Por eso se aparta un test fijo y el resto se reparte en
folds: el baseline valida con uno y la validación cruzada queda disponible sin rehacer
los datos.

El resultado se guarda en un JSON que se versiona (solo contiene rutas, no imágenes), de
modo que el split no depende de que este código o el generador aleatorio cambien.
"""

import json
import logging
import random
from pathlib import Path

from defect_detection.data.config import DataConfig
from defect_detection.data.mvtec import GOOD, class_of, good_images, iter_defect_samples

logger = logging.getLogger(__name__)

TEST = "test"

_PARAM_KEYS = ("category", "test_fraction", "folds", "seed", "negative_ratio")


def fold_name(index: int) -> str:
    return f"fold{index}"


def build_split(category_dir: Path, config: DataConfig) -> dict:
    """Asigna cada imagen a ``test`` o a un fold, estratificando por clase.

    Las imágenes buenas son un estrato más: se muestrean ``negative_ratio`` por cada
    defectuosa y se reparten igual, así que todas las particiones salen con la misma
    proporción de negativos.
    """
    if config.folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0 < config.test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    rng = random.Random(config.seed)
    strata: dict[str, list[Path]] = {}
    for defect, image, _mask in iter_defect_samples(category_dir):
        strata.setdefault(defect, []).append(image)

    defective = sum(len(images) for images in strata.values())
    negatives = round(config.negative_ratio * defective)
    pool = good_images(category_dir)
    if negatives > len(pool):
        raise ValueError(f"negative_ratio needs {negatives} good images, only {len(pool)} exist")
    strata[GOOD] = sorted(rng.sample(pool, negatives))

    partitions: dict[str, list[str]] = {TEST: []}
    partitions.update({fold_name(i): [] for i in range(config.folds)})
    # El contador de fold sigue de un estrato al siguiente para que los folds queden del
    # mismo tamaño en total, no solo dentro de cada clase.
    next_fold = 0
    for name in sorted(strata):
        images = strata[name]
        rng.shuffle(images)
        test_count = round(len(images) * config.test_fraction)
        for index, image in enumerate(images):
            relative = image.relative_to(category_dir).as_posix()
            if index < test_count:
                partitions[TEST].append(relative)
            else:
                partitions[fold_name(next_fold)].append(relative)
                next_fold = (next_fold + 1) % config.folds

    split = {key: getattr(config, key) for key in _PARAM_KEYS}
    split["partitions"] = {name: sorted(images) for name, images in partitions.items()}
    return split


def load_or_create_split(
    path: Path, category_dir: Path, config: DataConfig, force: bool = False
) -> dict:
    """Devuelve el split guardado en ``path`` o lo genera y lo guarda si no existe.

    Si el fichero existe pero se generó con otros parámetros, falla en vez de usar un
    split que ya no corresponde a la configuración; ``force`` lo regenera.
    """
    if path.exists() and not force:
        split = json.loads(path.read_text(encoding="utf-8"))
        stale = [key for key in _PARAM_KEYS if split.get(key) != getattr(config, key)]
        if stale:
            raise ValueError(
                f"{path} was generated with different {', '.join(stale)}; "
                "regenerate it explicitly to change the split"
            )
        logger.info("Using existing split %s", path)
        return split

    split = build_split(category_dir, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote split %s", path)
    return split


def class_counts(split: dict) -> dict[str, dict[str, int]]:
    """Número de imágenes por clase en cada partición, para logs y comprobaciones."""
    counts: dict[str, dict[str, int]] = {}
    for name, images in split["partitions"].items():
        counts[name] = {}
        for image in images:
            cls = class_of(Path(image))
            counts[name][cls] = counts[name].get(cls, 0) + 1
    return counts
