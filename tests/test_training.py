from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from defect_detection.training import train as train_module
from defect_detection.training.config import TrainConfig

CONFIGS = Path(__file__).parents[1] / "configs"
BASELINE = CONFIGS / "train_baseline.yaml"


def make_config(dataset: Path) -> TrainConfig:
    return TrainConfig(
        name="unit",
        dataset=str(dataset),
        fold=2,
        model="fake.pt",
        imgsz=320,
        single_cls=True,
        epochs=5,
        patience=2,
        batch=4,
        seed=1,
        confidence=0.25,
    )


class FakeYOLO:
    """Sustituye a ``ultralytics.YOLO``: registra la llamada en vez de entrenar."""

    calls: list[tuple[str, dict]] = []

    def __init__(self, model: str):
        self.model = model

    def train(self, **arguments):
        FakeYOLO.calls.append((self.model, arguments))
        return SimpleNamespace(save_dir=str(Path(arguments["project"]) / arguments["name"]))


@pytest.fixture
def fake_yolo(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    FakeYOLO.calls = []
    monkeypatch.setattr(train_module, "YOLO", FakeYOLO)
    return FakeYOLO


def test_baseline_config_is_valid():
    config = TrainConfig.from_yaml(BASELINE)

    assert config.single_cls is True
    assert config.data_yaml == Path("data/processed/hazelnut/fold0.yaml")


def test_train_passes_config_to_ultralytics(tmp_path: Path, fake_yolo: type[FakeYOLO]):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "fold2.yaml").touch()

    save_dir = train_module.train(make_config(dataset), tmp_path / "runs")

    model, arguments = fake_yolo.calls[0]
    assert model == "fake.pt"
    assert arguments == {
        "data": str(dataset / "fold2.yaml"),
        "project": str(tmp_path / "runs"),
        "name": "unit",
        "imgsz": 320,
        "single_cls": True,
        "epochs": 5,
        "patience": 2,
        "batch": 4,
        "seed": 1,
        "deterministic": True,
    }
    assert save_dir == tmp_path / "runs/unit"


def test_baseline_leaves_hyperparameters_to_ultralytics():
    assert TrainConfig.from_yaml(BASELINE).hyperparameters == {}


def test_lr_experiment_only_differs_from_baseline_in_name_and_hyperparameters():
    baseline = TrainConfig.from_yaml(BASELINE)
    experiment = TrainConfig.from_yaml(CONFIGS / "train_lr5e-4.yaml")

    assert experiment.hyperparameters["lr0"] == 0.0005
    assert replace(experiment, name=baseline.name, hyperparameters={}) == baseline


def test_hyperparameters_are_passed_to_ultralytics(tmp_path: Path, fake_yolo: type[FakeYOLO]):
    (tmp_path / "fold2.yaml").touch()
    config = replace(make_config(tmp_path), hyperparameters={"optimizer": "AdamW", "lr0": 0.001})

    train_module.train(config, tmp_path / "runs")

    arguments = fake_yolo.calls[0][1]
    assert arguments["optimizer"] == "AdamW" and arguments["lr0"] == 0.001


def test_overrides_replace_config_values(tmp_path: Path, fake_yolo: type[FakeYOLO]):
    (tmp_path / "fold2.yaml").touch()

    train_module.train(make_config(tmp_path), tmp_path / "runs", epochs=1, device="cpu")

    arguments = fake_yolo.calls[0][1]
    assert arguments["epochs"] == 1 and arguments["device"] == "cpu"


def test_missing_dataset_fails_before_loading_the_model(tmp_path: Path, fake_yolo: type[FakeYOLO]):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        train_module.train(make_config(tmp_path / "nope"), tmp_path / "runs")

    assert fake_yolo.calls == []
