"""Configuración del servicio, leída de variables de entorno.

Variables de entorno y no un fichero: es lo que mejor encaja con un contenedor, donde la
misma imagen se despliega con valores distintos sin reconstruirla.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from defect_detection.inference.predictor import DEFAULT_CONFIDENCE


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("models/lr5e-4.pt")
    confidence: float = DEFAULT_CONFIDENCE
    max_upload_mb: float = 10.0

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls) -> "Settings":
        """Lee ``MODEL_PATH``, ``CONFIDENCE`` y ``MAX_UPLOAD_MB``; lo que falte usa su defecto."""
        defaults = cls()
        return cls(
            model_path=Path(os.environ.get("MODEL_PATH", defaults.model_path)),
            confidence=float(os.environ.get("CONFIDENCE", defaults.confidence)),
            max_upload_mb=float(os.environ.get("MAX_UPLOAD_MB", defaults.max_upload_mb)),
        )
