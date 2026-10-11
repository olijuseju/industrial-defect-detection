"""CLI: empaqueta el dataset procesado en un zip, o lo desempaqueta verificando su SHA256."""

import argparse
from pathlib import Path

from defect_detection.data.package import package_dataset, unpack_dataset
from defect_detection.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checksums", type=Path, default=Path("configs/dataset_checksums.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="Crea <processed-dir>/<categoría>.zip")
    pack.add_argument("--category", default="hazelnut")
    unpack = commands.add_parser("unpack", help="Verifica y extrae un zip en <processed-dir>")
    unpack.add_argument("zip", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.command == "pack":
        package_dataset(
            args.processed_dir / args.category,
            args.processed_dir / f"{args.category}.zip",
            args.checksums,
        )
    else:
        unpack_dataset(args.zip, args.processed_dir, args.checksums)


if __name__ == "__main__":
    main()
