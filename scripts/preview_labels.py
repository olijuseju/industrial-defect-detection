"""CLI: hojas de contacto con las cajas derivadas de las máscaras, para elegir umbrales."""

import argparse
from pathlib import Path

from defect_detection.data.preview import write_contact_sheets
from defect_detection.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="hazelnut")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, help="Por defecto data/processed/<cat>/preview")
    parser.add_argument("--merge-gap-px", type=int, default=0)
    parser.add_argument("--min-area-px", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    out_dir = args.out_dir or Path("data/processed") / args.category / "preview"
    write_contact_sheets(args.raw_dir / args.category, out_dir, args.merge_gap_px, args.min_area_px)


if __name__ == "__main__":
    main()
