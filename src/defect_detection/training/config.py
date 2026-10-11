"""Parámetros de un entrenamiento, leídos de ``configs/train_<nombre>.yaml``."""

from dataclasses import dataclass
from pathlib import Path

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

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def data_yaml(self) -> Path:
        """YAML de Ultralytics del fold que hace de validación."""
        return Path(self.dataset) / f"fold{self.fold}.yaml"
