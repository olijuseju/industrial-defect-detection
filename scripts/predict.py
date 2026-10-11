"""CLI: predice sobre una imagen, imprime el resultado en JSON y guarda la imagen anotada."""

import argparse
import json
import logging
from pathlib import Path

import cv2

from defect_detection.inference.predictor import DEFAULT_CONFIDENCE, Predictor, annotate
from defect_detection.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/lr5e-4.pt"))
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--out", type=Path, help="Dónde guardar la imagen anotada (opcional)")
    parser.add_argument("--device", help="P. ej. cpu o 0; por defecto lo elige Ultralytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Los logs van a stderr para que stdout sea solo el JSON y se pueda encadenar con jq.
    configure_logging()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image {args.image}")
    predictor = Predictor(args.weights, args.confidence, args.device)
    prediction = predictor.predict(image)
    print(json.dumps(prediction.to_dict(), indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), annotate(image, prediction))
        logging.getLogger(__name__).info("Annotated image written to %s", args.out)


if __name__ == "__main__":
    main()
