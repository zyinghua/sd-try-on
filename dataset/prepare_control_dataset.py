#!/usr/bin/env python
"""
Prepare a HuggingFace `datasets`-compatible `metadata.jsonl` for ControlNet
training (`train.py`), written in-place inside the raw VITON-HD split dir.

Source layout (VITON-HD):
    <data_dir>/<split>/
        image/           XXXXX_00.jpg            target images
        cloth/           XXXXX_00.jpg            cloth items
        openpose_img/    XXXXX_00_rendered.png   pose renders (default)
        image-densepose/ XXXXX_00.jpg            densepose (alternative, --pose_source)

This script writes `metadata.jsonl` directly into `<data_dir>/<split>/` with
per-row fields (paths relative to that directory):
    image              e.g. "image/00000_00.jpg"
    text               caption
    conditioning_image e.g. "openpose_img/00000_00_rendered.png"
    cloth_image        e.g. "cloth/00000_00.jpg"

After running this script, use:
    accelerate launch train.py --train_data_dir <data_dir>/<split> ...
"""

import argparse
import json
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Raw VITON-HD dataset root. Default: {DEFAULT_DATA_DIR}")
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="Which split of the VITON-HD layout to prepare. Default: train")
    p.add_argument("--pose_source", choices=["openpose_img", "image-densepose"], default="openpose_img",
                   help="Which pose representation to use. Default: openpose_img")
    p.add_argument("--caption", type=str,
                   default="a person wearing a garment",
                   help="Caption used for every row (VITON-HD ships no per-image captions).")
    p.add_argument("--max_samples", type=int, default=None,
                   help="If set, only emit the first N rows (quick-debug aid).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing metadata.jsonl.")
    return p.parse_args()


def main():
    args = parse_args()
    split_dir = args.data_dir / args.split
    image_src = split_dir / "image"
    cloth_src = split_dir / "cloth"
    pose_src = split_dir / args.pose_source

    for d, label in [(image_src, "image"), (cloth_src, "cloth"), (pose_src, args.pose_source)]:
        if not d.is_dir():
            raise FileNotFoundError(f"Missing source directory for '{label}': {d}")

    metadata_path = split_dir / "metadata.jsonl"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} exists; pass --overwrite to replace.")

    image_files = sorted(p.name for p in image_src.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_samples is not None:
        image_files = image_files[: args.max_samples]

    written = skipped = 0
    with metadata_path.open("w") as f:
        for fname in image_files:
            stem = Path(fname).stem
            cloth_name = fname
            if args.pose_source == "openpose_img":
                pose_name = f"{stem}_rendered.png"
            else:
                pose_name = f"{stem}.jpg"

            if not (cloth_src / cloth_name).exists():
                skipped += 1
                continue
            if not (pose_src / pose_name).exists():
                skipped += 1
                continue

            row = {
                "image": f"image/{fname}",
                "text": args.caption,
                "conditioning_image": f"{args.pose_source}/{pose_name}",
                "cloth_image": f"cloth/{cloth_name}",
            }
            f.write(json.dumps(row) + "\n")
            written += 1

    print(f"Wrote {written} rows to {metadata_path} (skipped {skipped} with missing pair/pose).")
    print(f"Use with: accelerate launch train.py --train_data_dir {split_dir}")


if __name__ == "__main__":
    main()
