"""CLI: mide las falsas alarmas sobre las imágenes buenas que quedaron fuera del split."""

import argparse
import json
import logging
from pathlib import Path

from defect_detection.data.config import DataConfig
from defect_detection.evaluation.evaluate import false_alarms_on_good, unused_good_images
from defect_detection.logging_config import configure_logging
from defect_detection.training.config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="P. ej. models/lr5e-4.pt")
    parser.add_argument(
        "--config", type=Path, required=True, help="Config de entrenamiento: fija el umbral"
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_hazelnut.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", type=Path, help="Por defecto configs/splits/<categoría>.json")
    parser.add_argument("--out", type=Path, help="Por defecto reports/false_alarms_<name>.json")
    parser.add_argument("--device", help="P. ej. cpu o 0; por defecto lo elige Ultralytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = TrainConfig.from_yaml(args.config)
    category = DataConfig.from_yaml(args.data_config).category
    split_path = args.split or Path("configs/splits") / f"{category}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    images = unused_good_images(args.raw_dir / category, split)
    report = false_alarms_on_good(args.weights, images, config.confidence, args.device)
    out = args.out or Path("reports") / f"false_alarms_{config.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("Report written to %s", out)


if __name__ == "__main__":
    main()
