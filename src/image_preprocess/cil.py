import argparse
import logging
from pathlib import Path

from .batch import process_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch remove backgrounds, center, and resize clothing images to 224x224 PNGs."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--area_fraction", type=float, default=0.65)
    parser.add_argument("--max_occupancy", type=float, default=0.9)
    parser.add_argument("--bbox_margin_ratio", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logging.info(f"Input:  {input_dir}")
    logging.info(f"Output: {output_dir}")

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=args.size,
        target_area_fraction=args.area_fraction,
        max_occupancy=args.max_occupancy,
        bbox_margin_ratio=args.bbox_margin_ratio,
    )

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
