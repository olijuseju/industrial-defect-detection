"""Predicción sobre una imagen con un modelo entrenado.

Esta capa no sabe nada de HTTP: recibe una imagen como array y devuelve datos propios
(`Prediction`), no objetos de Ultralytics. Así la API, un script por lotes o un test usan
la misma lógica y el resto del código no depende de la librería de detección.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Umbral validado en el paso 4 (ver configs/train_lr5e-4.yaml y reports/).
DEFAULT_CONFIDENCE = 0.25

# Nombres de clase del modelo -> nombres que ve el cliente. `single_cls=True` de Ultralytics
# deja la clase única con el nombre `item`; cualquier otra clase conserva el suyo.
LABELS = {"item": "defect"}

_BOX_COLOR = (0, 0, 255)  # BGR
_TEXT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # Esquinas en píxeles de la imagen original: x1, y1, x2, y2.
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": self.confidence, "box": list(self.box)}


@dataclass(frozen=True)
class Prediction:
    width: int
    height: int
    detections: tuple[Detection, ...]
    inference_ms: float

    @property
    def defective(self) -> bool:
        """La pieza se da por defectuosa con una sola detección por encima del umbral."""
        return bool(self.detections)

    def to_dict(self) -> dict:
        return {
            "defective": self.defective,
            "detections": [detection.to_dict() for detection in self.detections],
            "width": self.width,
            "height": self.height,
            "inference_ms": self.inference_ms,
        }


class Predictor:
    """Modelo cargado en memoria. Se crea una vez y se reutiliza en cada predicción."""

    def __init__(
        self,
        weights: Path,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str | None = None,
    ):
        if not weights.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        self.weights = weights
        self.confidence = confidence
        self.device = device
        self._model = YOLO(str(weights))
        logger.info("Loaded %s (confidence threshold %.2f)", weights, confidence)

    def predict(self, image: np.ndarray) -> Prediction:
        """Detecta defectos en una imagen BGR (la que devuelve ``cv2.imread``)."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected a BGR image of shape (H, W, 3), got {image.shape}")
        result = self._model.predict(
            image, conf=self.confidence, device=self.device, verbose=False
        )[0]
        boxes = result.boxes
        detections = [
            Detection(
                label=LABELS.get(result.names[int(cls)], result.names[int(cls)]),
                confidence=round(float(conf), 4),
                box=tuple(round(value) for value in xyxy),
            )
            for xyxy, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
            )
        ]
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        height, width = image.shape[:2]
        return Prediction(
            width=width,
            height=height,
            detections=tuple(detections),
            inference_ms=round(float(result.speed["inference"]), 1),
        )


def annotate(image: np.ndarray, prediction: Prediction) -> np.ndarray:
    """Devuelve una copia de la imagen con las cajas y su confianza pintadas."""
    out = image.copy()
    # Grosor y texto proporcionales al tamaño: las imágenes de MVTec son de 1024 px.
    scale = max(out.shape[:2]) / 1024
    thickness = max(1, round(3 * scale))
    font_scale = max(0.4, 0.8 * scale)
    for detection in prediction.detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, thickness)
        caption = f"{detection.label} {detection.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # La etiqueta va encima de la caja, o dentro si no cabe por arriba.
        top = y1 - text_h - baseline if y1 - text_h - baseline >= 0 else y1
        cv2.rectangle(out, (x1, top), (x1 + text_w, top + text_h + baseline), _BOX_COLOR, -1)
        cv2.putText(
            out,
            caption,
            (x1, top + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _TEXT_COLOR,
            thickness,
        )
    return out
