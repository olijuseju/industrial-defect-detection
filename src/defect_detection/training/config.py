"""Parámetros de un entrenamiento, leídos de ``configs/train_<nombre>.yaml``."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    name: str
    dataset: str
    fold: int
    model: str
    imgsz: int
    single_cls: bool
    epochs: int
    patience: int
    batch: int
    seed: int
    # Umbral de confianza a partir del cual una detección cuenta al evaluar por imagen.
    # No interviene en el entrenamiento.
    confidence: float
    # Argumentos de ``YOLO.train`` que solo fija algún experimento (optimizador, lr0...).
    # Lo que no aparece aquí se queda con el valor por defecto de Ultralytics.
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def data_yaml(self) -> Path:
        """YAML de Ultralytics del fold que hace de validación."""
        return Path(self.dataset) / f"fold{self.fold}.yaml"
