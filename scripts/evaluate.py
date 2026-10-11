"""CLI: evalúa unos pesos entrenados y guarda las métricas en reports/."""

import argparse
import json
import logging
from pathlib import Path

from defect_detection.evaluation.evaluate import evaluate
from defect_detection.training.config import TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="P. ej. models/lr5e-4.pt")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config con la que se entrenó: fija dataset, fold, single_cls y umbral",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evalúa también en test. Solo para el modelo final: test se usa una vez",
    )
    parser.add_argument("--out", type=Path, help="Por defecto reports/metrics_<name>.json")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", help="P. ej. cpu o 0; por defecto lo elige Ultralytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = TrainConfig.from_yaml(args.config)
    report = evaluate(
        args.weights,
        config.data_yaml,
        config.single_cls,
        config.confidence,
        include_test=args.test,
        runs_dir=args.runs_dir,
        device=args.device,
    )
    out = args.out or Path("reports") / f"metrics_{config.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("Metrics written to %s", out)


if __name__ == "__main__":
    main()
