"""Formato de las respuestas JSON; FastAPI genera con ellos la documentación de la API."""

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    label: str = Field(examples=["defect"])
    confidence: float = Field(ge=0, le=1, examples=[0.85])
    box: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Esquinas en píxeles de la imagen enviada: x1, y1, x2, y2",
        examples=[[264, 267, 822, 535]],
    )


class PredictionOut(BaseModel):
    defective: bool = Field(description="True si hay al menos una detección sobre el umbral")
    detections: list[DetectionOut]
    width: int
    height: int
    inference_ms: float


class HealthOut(BaseModel):
    status: str = Field(examples=["ok"])
    model: str = Field(description="Fichero de pesos cargado", examples=["lr5e-4.pt"])
    confidence: float = Field(description="Umbral de confianza del servicio")
