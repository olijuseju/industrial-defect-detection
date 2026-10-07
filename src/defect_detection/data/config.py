"""Parámetros de la preparación de datos, leídos de ``configs/data_<categoría>.yaml``."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    category: str
    merge_gap_px: int
    min_area_px: int
    test_fraction: float
    folds: int
    seed: int
    negative_ratio: float

    @classmethod
    def from_yaml(cls, path: Path) -> "DataConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))
