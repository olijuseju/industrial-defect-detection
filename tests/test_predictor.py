from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defect_detection.inference import predictor as predictor_module
from defect_detection.inference.predictor import Detection, Prediction, Predictor, annotate


class FakeYOLO:
    """Sustituye a ``ultralytics.YOLO``: devuelve dos cajas fijas, la más débil primero."""

    calls: list[dict] = []
    names = {0: "item"}

    def __init__(self, weights: str):
        self.weights = weights

    def predict(self, image: np.ndarray, **arguments):
        FakeYOLO.calls.append(arguments)
        boxes = SimpleNamespace(
            xyxy=np.array([[10.4, 20.6, 110.2, 220.9], [300.0, 310.0, 400.0, 410.0]]),
            conf=np.array([0.31234, 0.91239]),
            cls=np.array([0.0, 0.0]),
        )
        return [SimpleNamespace(boxes=boxes, names=FakeYOLO.names, speed={"inference": 12.34})]


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    FakeYOLO.calls = []
    FakeYOLO.names = {0: "item"}
    monkeypatch.setattr(predictor_module, "YOLO", FakeYOLO)
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    return path


def test_predict_returns_plain_data_sorted_by_confidence(weights: Path):
    prediction = Predictor(weights, confidence=0.3).predict(np.zeros((480, 640, 3), np.uint8))

    assert prediction.defective is True
    assert (prediction.width, prediction.height) == (640, 480)
    assert prediction.inference_ms == 12.3
    assert prediction.detections == (
        Detection(label="defect", confidence=0.9124, box=(300, 310, 400, 410)),
        Detection(label="defect", confidence=0.3123, box=(10, 21, 110, 221)),
    )
    assert FakeYOLO.calls[0]["conf"] == 0.3


def test_class_names_other_than_item_are_kept(weights: Path):
    FakeYOLO.names = {0: "crack"}

    prediction = Predictor(weights).predict(np.zeros((64, 64, 3), np.uint8))

    assert {detection.label for detection in prediction.detections} == {"crack"}


def test_prediction_without_detections_is_not_defective():
    prediction = Prediction(width=10, height=10, detections=(), inference_ms=1.0)

    assert prediction.defective is False
    assert prediction.to_dict()["detections"] == []


def test_to_dict_is_json_ready():
    detection = Detection(label="defect", confidence=0.5, box=(1, 2, 3, 4))
    prediction = Prediction(width=8, height=6, detections=(detection,), inference_ms=2.0)

    assert prediction.to_dict() == {
        "defective": True,
        "detections": [{"label": "defect", "confidence": 0.5, "box": [1, 2, 3, 4]}],
        "width": 8,
        "height": 6,
        "inference_ms": 2.0,
    }


def test_missing_weights_fail_with_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="weights"):
        Predictor(tmp_path / "nope.pt")


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_confidence_out_of_range_is_rejected(weights: Path, confidence: float):
    with pytest.raises(ValueError, match="confidence"):
        Predictor(weights, confidence=confidence)


def test_grayscale_images_are_rejected(weights: Path):
    with pytest.raises(ValueError, match="BGR"):
        Predictor(weights).predict(np.zeros((64, 64), np.uint8))


def test_annotate_draws_on_a_copy():
    image = np.zeros((200, 200, 3), np.uint8)
    detection = Detection(label="defect", confidence=0.9, box=(50, 60, 150, 160))
    prediction = Prediction(width=200, height=200, detections=(detection,), inference_ms=1.0)

    annotated = annotate(image, prediction)

    assert not image.any()
    assert annotated.shape == image.shape
    assert annotated[60, 100].tolist() == [0, 0, 255]


def test_annotate_keeps_the_label_inside_the_image_at_the_top_edge():
    image = np.zeros((100, 100, 3), np.uint8)
    detection = Detection(label="defect", confidence=0.9, box=(0, 0, 50, 50))
    prediction = Prediction(width=100, height=100, detections=(detection,), inference_ms=1.0)

    assert annotate(image, prediction).any()
