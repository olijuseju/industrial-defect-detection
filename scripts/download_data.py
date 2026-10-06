"""CLI para descargar y extraer MVTec AD en data/raw. La lógica vive en el paquete."""

import argparse
import logging
import os
from pathlib import Path

from defect_detection.data.download import MVTEC_AD_CATEGORIES, ensure_dataset

URL_ENV_VAR = "MVTEC_AD_URL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url", help=f"URL del archivo (por defecto, la variable {URL_ENV_VAR})")
    source.add_argument("--archive", type=Path, help="Archivo .tar ya descargado a mano")
    parser.add_argument(
        "--category",
        action="append",
        choices=MVTEC_AD_CATEGORIES,
        help="Categoría a extraer; se puede repetir. Por defecto, todas",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checksums", type=Path, default=Path("configs/dataset_checksums.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    url = args.url or (None if args.archive else os.environ.get(URL_ENV_VAR))
    ensure_dataset(
        raw_dir=args.raw_dir,
        checksums_path=args.checksums,
        categories=args.category,
        url=url,
        archive=args.archive,
    )


if __name__ == "__main__":
    main()
