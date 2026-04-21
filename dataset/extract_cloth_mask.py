"""Extract the upper-body garment region from image-parse-v3 parse maps.

The parse maps are P-mode indexed PNGs following the LIP / VITON-HD palette.
The paired product images in `cloth/` are always upper-body garments, which
may be labeled as upper-clothes (5), dress (6), or coat (7) in the parse map,
so this script unions those three classes into a single binary mask
(cloth=255, else=0) and writes it into a sibling `image_cloth_mask/`
directory.

Usage:
    python dataset/extract_cloth_mask.py [parse_dir]

Example:
    python dataset/extract_cloth_mask.py dataset/data/train/image-parse-v3
    # -> dataset/data/train/image_cloth_mask/*.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# LIP palette: 5 upper-clothes, 6 dress, 7 coat — all appear as upper-body
# products in VITON-HD's `cloth/` directory.
UPPER_GARMENT_LABELS = (5, 6, 7)
OUTPUT_DIRNAME = "image_cloth_mask"
DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "train" / "image-parse-v3"


def extract(parse_path: Path, out_path: Path, labels) -> None:
    parse = Image.open(parse_path)
    if parse.mode != "P":
        parse = parse.convert("P")
    arr = np.array(parse)
    mask = np.isin(arr, labels).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT,
        type=Path,
        help=f"Directory of parse-map PNGs (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=list(UPPER_GARMENT_LABELS),
        help="Parse label ids to include in the mask (default: 5 6 7)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.input_dir}")

    output_dir = args.input_dir.parent / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(args.input_dir.glob("*.png"))
    for src in pngs:
        extract(src, output_dir / src.name, args.labels)
    print(f"Wrote {len(pngs)} masks to {output_dir} (labels={args.labels})")


if __name__ == "__main__":
    main()
