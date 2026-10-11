from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="needs the optional `api` extra: uv sync --extra api")

from fastapi.testclient import TestClient  # noqa: E402

from defect_detection.api.app import create_app  # noqa: E402
from defect_detection.api.settings import Settings  # noqa: E402
from defect_detection.inference.predictor import Detection, Prediction  # noqa: E402


class FakePredictor:
    """Sustituye al ``Predictor``: una detección fija, sin cargar ningún modelo."""

    weights = Path("models/fake.pt")
    confidence = 0.25

    def __init__(self, detections: tuple[Detection, ...] | None = None):
        self.detections = (
            (Detection(label="defect", confidence=0.9, box=(10, 20, 60, 80)),)
            if detections is None
            else detections
        )
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, image: np.ndarray) -> Prediction:
        self.shapes.append(image.shape)
        height, width = image.shape[:2]
        return Prediction(width=width, height=height, detections=self.detections, inference_ms=5.0)


def png_bytes(width: int = 120, height: int = 100) -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(predictor: FakePredictor):
    app = create_app(Settings(max_upload_mb=0.01), predictor=predictor)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "fake.pt", "confidence": 0.25}


def test_startup_warms_the_model_up(client: TestClient, predictor: FakePredictor):
    assert predictor.shapes == [(640, 640, 3)]


def test_predict_returns_the_detections_as_json(client: TestClient, predictor: FakePredictor):
    response = client.post("/predict", files={"file": ("part.png", png_bytes(), "image/png")})

    assert response.status_code == 200
    assert response.json() == {
        "defective": True,
        "detections": [{"label": "defect", "confidence": 0.9, "box": [10, 20, 60, 80]}],
        "width": 120,
        "height": 100,
        "inference_ms": 5.0,
    }
    assert predictor.shapes[-1] == (100, 120, 3)


def test_good_part_is_not_defective():
    app = create_app(Settings(), predictor=FakePredictor(detections=()))

    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("p.png", png_bytes(), "image/png")})

    assert response.json()["defective"] is False
    assert response.json()["detections"] == []


def test_annotated_endpoint_returns_a_png_with_the_boxes(client: TestClient):
    response = client.post(
        "/predict/annotated", files={"file": ("part.png", png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-defective"] == "true"
    assert response.headers["x-detections"] == "1"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (100, 120, 3)
    assert image.any()


def test_unsupported_content_type_is_rejected(client: TestClient):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_bytes_that_are_not_an_image_are_rejected(client: TestClient):
    response = client.post("/predict", files={"file": ("part.png", b"not a png", "image/png")})

    assert response.status_code == 400
    assert response.json()["detail"] == "File is not a valid image"


def test_too_large_upload_is_rejected(client: TestClient):
    too_big = b"0" * (Settings(max_upload_mb=0.01).max_upload_bytes + 1)

    response = client.post("/predict", files={"file": ("part.png", too_big, "image/png")})

    assert response.status_code == 413


def test_missing_file_is_a_validation_error(client: TestClient):
    assert client.post("/predict").status_code == 422


def test_service_does_not_start_without_weights(tmp_path: Path):
    app = create_app(Settings(model_path=tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError, match="weights"), TestClient(app):
        pass


def test_settings_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_PATH", "/models/custom.pt")
    monkeypatch.setenv("CONFIDENCE", "0.4")
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)

    settings = Settings.from_env()

    assert settings.model_path == Path("/models/custom.pt")
    assert settings.confidence == 0.4
    assert settings.max_upload_mb == 10.0
