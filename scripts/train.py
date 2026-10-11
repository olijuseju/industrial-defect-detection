"""CLI: entrena un modelo a partir de un YAML de configs/."""

import argparse
import logging
from pathlib import Path

from defect_detection.training.config import TrainConfig
from defect_detection.training.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_baseline.yaml"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, help="Pisa las épocas de la config (pruebas)")
    parser.add_argument("--device", help="P. ej. cpu o 0; por defecto lo elige Ultralytics")
    parser.add_argument("--name", help="Pisa el nombre del run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    overrides = {
        key: value
        for key, value in (("epochs", args.epochs), ("device", args.device), ("name", args.name))
        if value is not None
    }
    train(TrainConfig.from_yaml(args.config), args.runs_dir, **overrides)


if __name__ == "__main__":
    main()
