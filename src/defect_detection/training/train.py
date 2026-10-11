"""Entrenamiento de YOLO con Ultralytics a partir de una ``TrainConfig``.

La lógica vive aquí y no en el notebook de Colab para que el mismo código corra en local
(prueba rápida en CPU) y en Colab (entrenamiento real en GPU).
"""

import logging
from pathlib import Path

from ultralytics import YOLO

from defect_detection.training.config import TrainConfig

logger = logging.getLogger(__name__)


def train_arguments(config: TrainConfig, runs_dir: Path) -> dict:
    """Argumentos de ``YOLO.train`` que se derivan de la configuración."""
    return {
        # Rutas absolutas: Ultralytics resuelve las relativas contra sus propios
        # directorios por defecto, no contra el directorio de trabajo.
        "data": str(config.data_yaml.resolve()),
        "project": str(runs_dir.resolve()),
        "name": config.name,
        "imgsz": config.imgsz,
        "single_cls": config.single_cls,
        "epochs": config.epochs,
        "patience": config.patience,
        "batch": config.batch,
        "seed": config.seed,
        "deterministic": True,
        **config.hyperparameters,
    }


def train(config: TrainConfig, runs_dir: Path = Path("runs"), **overrides) -> Path:
    """Entrena y devuelve la carpeta del run (pesos en ``weights/best.pt``).

    ``overrides`` pisa argumentos de ``YOLO.train`` para una ejecución concreta sin tocar
    la configuración, p. ej. ``epochs=1, device="cpu"`` en una prueba local.
    """
    if not config.data_yaml.is_file():
        raise FileNotFoundError(
            f"{config.data_yaml} not found; run scripts/prepare_data.py or check 'dataset'"
        )
    arguments = {**train_arguments(config, runs_dir), **overrides}
    logger.info("Training %s with %s", config.model, arguments)
    results = YOLO(config.model).train(**arguments)
    save_dir = Path(results.save_dir)
    logger.info("Run saved to %s", save_dir)
    return save_dir
