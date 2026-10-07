"""CLI: genera el dataset YOLO de una categoría (split + imágenes + etiquetas + YAML)."""

import argparse
import logging
from pathlib import Path

from defect_detection.data.config import DataConfig
from defect_detection.data.prepare import prepare_dataset
from defect_detection.data.split import class_counts, load_or_create_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_hazelnut.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=Path, help="Por defecto configs/splits/<categoría>.json")
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Regenera el split aunque ya exista (cambia qué imágenes van a test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = DataConfig.from_yaml(args.config)
    category_dir = args.raw_dir / config.category
    split_path = args.split or Path("configs/splits") / f"{config.category}.json"
    split = load_or_create_split(split_path, category_dir, config, force=args.force_split)
    for partition, counts in class_counts(split).items():
        logging.getLogger(__name__).info("%s: %s", partition, dict(sorted(counts.items())))
    prepare_dataset(category_dir, args.processed_dir / config.category, split, config)


if __name__ == "__main__":
    main()
