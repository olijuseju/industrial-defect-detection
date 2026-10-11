from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from defect_detection.evaluation import evaluate as evaluation
from defect_detection.evaluation.evaluate import (
    ImageLevelMetrics,
    best_f1_confidence,
    count_boxes,
    image_level_metrics,
    partition_images,
)

# Confianza máxima que el modelo falso "detecta" en cada imagen.
TOP_CONFIDENCE = {"bad_hit": 0.9, "bad_weak": 0.3, "bad_miss": 0.0, "good_ok": 0.0, "good_fa": 0.6}


def make_dataset(root: Path) -> Path:
    """Dataset mínimo: las mismas 5 imágenes en val y en test."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for name in TOP_CONFIDENCE:
        (root / "images" / f"{name}.png").touch()
        label = "0 0.5 0.5 0.1 0.1\n" if name.startswith("bad") else ""
        (root / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")
    listing = "".join(f"./images/{name}.png\n" for name in TOP_CONFIDENCE)
    (root / "fold0_val.txt").write_text(listing, encoding="utf-8")
    (root / "test.txt").write_text(listing, encoding="utf-8")
    data_yaml = root / "fold0.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"train": "x.txt", "val": "fold0_val.txt", "test": "test.txt"}),
        encoding="utf-8",
    )
    return data_yaml


def fake_box_metrics(best_confidence: float) -> SimpleNamespace:
    """Métricas con una curva F1 cuyo máximo está en ``best_confidence``."""
    px = np.linspace(0, 1, 1000)
    f1_curve = (1 - np.abs(px - best_confidence))[None, :]
    return SimpleNamespace(px=px, f1_curve=f1_curve, mp=0.9, mr=0.8, map50=0.7, map=0.6)


class FakeBoxes:
    def __init__(self, confidences: list[float]):
        self.conf = np.array(confidences)

    def __len__(self) -> int:
        return len(self.conf)


class FakeYOLO:
    """Sustituye a ``ultralytics.YOLO``: devuelve métricas y detecciones fijas."""

    val_calls: list[dict] = []

    def __init__(self, weights: str):
        self.weights = weights

    def val(self, **arguments):
        FakeYOLO.val_calls.append(arguments)
        return SimpleNamespace(box=fake_box_metrics(0.5))

    def predict(self, images: list[str], conf: float, **_):
        tops = [TOP_CONFIDENCE[Path(image).stem] for image in images]
        return [SimpleNamespace(boxes=FakeBoxes([top] if top >= conf else [])) for top in tops]


@pytest.fixture
def fake_yolo(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    FakeYOLO.val_calls = []
    monkeypatch.setattr(evaluation, "YOLO", FakeYOLO)
    return FakeYOLO


def test_partition_images_resolves_paths_relative_to_the_listing(tmp_path: Path):
    data_yaml = make_dataset(tmp_path)

    images = partition_images(data_yaml, "val")

    assert images[0] == (tmp_path / "images/bad_hit.png").resolve()
    assert len(images) == len(TOP_CONFIDENCE)


def test_count_boxes_is_zero_for_good_images(tmp_path: Path):
    make_dataset(tmp_path)

    assert count_boxes(tmp_path / "images/bad_hit.png") == 1
    assert count_boxes(tmp_path / "images/good_ok.png") == 0


def test_image_level_metrics_counts_flagged_images_at_the_threshold():
    samples = [(True, 0.9), (True, 0.3), (True, 0.0), (False, 0.0), (False, 0.6)]

    strict = image_level_metrics(samples, 0.5)
    loose = image_level_metrics(samples, 0.2)

    assert strict == ImageLevelMetrics(
        good_images=2, false_alarms=1, defective_images=3, detected=1
    )
    assert strict.false_alarm_rate == 0.5
    assert loose.detected == 2 and loose.detection_rate == pytest.approx(2 / 3)


def test_rates_are_zero_without_images():
    empty = image_level_metrics([], 0.5)

    assert empty.false_alarm_rate == 0.0 and empty.detection_rate == 0.0


def test_best_f1_confidence_finds_the_peak_of_the_curve():
    assert best_f1_confidence(fake_box_metrics(0.4)) == pytest.approx(0.4, abs=0.01)


def test_evaluate_does_not_touch_test_unless_asked(tmp_path: Path, fake_yolo: type[FakeYOLO]):
    data_yaml = make_dataset(tmp_path)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")

    report = evaluation.evaluate(weights, data_yaml, single_cls=True, runs_dir=tmp_path / "runs")

    assert [call["split"] for call in fake_yolo.val_calls] == ["val"]
    assert fake_yolo.val_calls[0]["single_cls"] is True
    assert "test" not in report
    assert report["confidence"] == pytest.approx(0.5, abs=0.01)
    assert report["val"]["images"] == 5 and report["val"]["boxes"] == 3
    assert report["val"]["image_level"]["detected"] == 1
    assert report["val"]["image_level"]["false_alarms"] == 1
    assert report["val"]["image_level_by_confidence"]["0.25"]["detected"] == 2


def test_test_split_uses_the_threshold_chosen_on_val(tmp_path: Path, fake_yolo: type[FakeYOLO]):
    data_yaml = make_dataset(tmp_path)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")

    report = evaluation.evaluate(
        weights, data_yaml, single_cls=True, include_test=True, runs_dir=tmp_path / "runs"
    )

    assert [call["split"] for call in fake_yolo.val_calls] == ["val", "test"]
    assert report["test"]["image_level"] == report["val"]["image_level"]
    assert "image_level_by_confidence" not in report["test"]
