"""Evaluación de un modelo entrenado sobre una partición del dataset.

Dos niveles de métricas:

- Por caja (precision, recall, mAP): las calcula Ultralytics con ``model.val``.
- Por imagen: en cuántas piezas buenas salta una falsa alarma y en cuántas defectuosas se
  detecta algo. Es lo que decide si una pieza se aparta de la línea.

Las métricas por imagen dependen del umbral de confianza. Viene de la configuración y se
decide mirando solo validación, donde además se tabulan varios umbrales; en test se aplica
sin cambios, para no ajustar nada con los datos de test.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO

from defect_detection.data.download import sha256_file
from defect_detection.data.split import TEST

logger = logging.getLogger(__name__)

VAL = "val"
# Umbrales de confianza con los que se tabulan las métricas por imagen en validación.
SWEEP_THRESHOLDS = (0.1, 0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ImageLevelMetrics:
    good_images: int
    false_alarms: int
    defective_images: int
    detected: int

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.good_images if self.good_images else 0.0

    @property
    def detection_rate(self) -> float:
        return self.detected / self.defective_images if self.defective_images else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "false_alarm_rate": self.false_alarm_rate,
            "detection_rate": self.detection_rate,
        }


def partition_images(data_yaml: Path, split: str) -> list[Path]:
    """Imágenes de una partición (``val`` o ``test``) según el YAML del dataset."""
    dataset = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    listing = data_yaml.parent / dataset[split]
    lines = listing.read_text(encoding="utf-8").splitlines()
    return [(listing.parent / line).resolve() for line in lines if line.strip()]


def count_boxes(image: Path) -> int:
    """Cajas etiquetadas de una imagen; 0 significa pieza buena."""
    label = image.parents[1] / "labels" / f"{image.stem}.txt"
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


def image_level_metrics(samples: list[tuple[bool, float]], confidence: float) -> ImageLevelMetrics:
    """Resume pares ``(es_defectuosa, confianza máxima detectada)``, uno por imagen.

    Una imagen cuenta como marcada si tiene alguna detección con confianza >= ``confidence``,
    esté donde esté: no se comprueba que la caja caiga sobre el defecto (de eso se ocupan
    las métricas por caja).
    """
    flagged_good = sum(1 for defective, top in samples if not defective and top >= confidence)
    flagged_defective = sum(1 for defective, top in samples if defective and top >= confidence)
    defective_total = sum(1 for defective, _ in samples if defective)
    return ImageLevelMetrics(
        good_images=len(samples) - defective_total,
        false_alarms=flagged_good,
        defective_images=defective_total,
        detected=flagged_defective,
    )


def top_confidences(model: YOLO, images: list[Path], device: str | None = None) -> list[float]:
    """Confianza de la detección más segura de cada imagen (0.0 si no hay ninguna).

    Con ella se evalúa cualquier umbral sin volver a predecir: una imagen queda marcada
    con umbral ``t`` si y solo si su confianza máxima es >= ``t``.
    """
    results = model.predict(
        [str(image) for image in images], conf=SWEEP_THRESHOLDS[0], device=device, verbose=False
    )
    return [float(result.boxes.conf.max()) if len(result.boxes) else 0.0 for result in results]


def evaluate_split(
    model: YOLO,
    data_yaml: Path,
    split: str,
    single_cls: bool,
    runs_dir: Path,
    confidence: float,
    sweep: bool = False,
    device: str | None = None,
) -> dict:
    """Métricas de una partición; ``sweep`` añade las de imagen con varios umbrales."""
    metrics = model.val(
        data=str(data_yaml.resolve()),
        split=split,
        single_cls=single_cls,
        project=str(runs_dir.resolve()),
        name=f"eval_{split}",
        exist_ok=True,
        plots=False,
        device=device,
    )
    box = metrics.box
    images = partition_images(data_yaml, split)
    labelled = [count_boxes(image) for image in images]
    samples = [
        (boxes > 0, top)
        for boxes, top in zip(labelled, top_confidences(model, images, device), strict=True)
    ]
    report = {
        "images": len(images),
        "boxes": sum(labelled),
        "box_level": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
    }
    report["image_level"] = image_level_metrics(samples, confidence).to_dict()
    if sweep:
        report["image_level_by_confidence"] = {
            f"{threshold:.2f}": image_level_metrics(samples, threshold).to_dict()
            for threshold in SWEEP_THRESHOLDS
        }
    return report


def evaluate(
    weights: Path,
    data_yaml: Path,
    single_cls: bool,
    confidence: float,
    include_test: bool = False,
    runs_dir: Path = Path("runs"),
    device: str | None = None,
) -> dict:
    """Evalúa en validación y, solo si se pide, en test.

    Los umbrales alternativos se tabulan solo en validación: en test no, para no tener la
    tentación de elegir el umbral mirándolo.
    """
    model = YOLO(str(weights))
    report: dict = {
        "weights": {"file": weights.name, "sha256": sha256_file(weights)},
        "data": data_yaml.as_posix(),
        "single_cls": single_cls,
        "confidence": confidence,
    }
    report[VAL] = evaluate_split(
        model, data_yaml, VAL, single_cls, runs_dir, confidence, sweep=True, device=device
    )
    if include_test:
        report[TEST] = evaluate_split(
            model, data_yaml, TEST, single_cls, runs_dir, confidence, device=device
        )
    return report
