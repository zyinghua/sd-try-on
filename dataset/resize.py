"""Resize all dataset images from 1024x768 to 512x384, in place.

Uses LANCZOS for RGB images and NEAREST for masks / parse maps so discrete
label values are preserved. Also scales OpenPose JSON keypoints by the same
factor so coordinates stay aligned with the resized images.
"""

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent / "data"
TARGET = (384, 512)  # PIL uses (width, height); original is 768x1024
SCALE = 0.5  # 1024 -> 512, 768 -> 384
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NEAREST_HINTS = ("mask", "parse", "label", "agnostic-mask", "pose", "densepose")
KEYPOINT_FIELD_SUFFIX = "_keypoints_2d"


def pick_resample(path: Path) -> int:
    lowered = str(path).lower()
    if any(h in lowered for h in NEAREST_HINTS):
        return Image.NEAREST
    return Image.LANCZOS


def resize_file(path: Path) -> bool:
    with Image.open(path) as img:
        if img.size == TARGET:
            return False
        mode = img.mode
        resized = img.resize(TARGET, pick_resample(path))
        if resized.mode != mode:
            resized = resized.convert(mode)
        resized.save(path)
    return True


def scale_keypoints(values):
    """Scale x, y of each (x, y, confidence) triplet; leave confidence alone."""
    scaled = list(values)
    for i in range(0, len(scaled) - 2, 3):
        scaled[i] = scaled[i] * SCALE
        scaled[i + 1] = scaled[i + 1] * SCALE
    return scaled


def rescale_pose_json(path: Path) -> bool:
    with open(path) as f:
        data = json.load(f)
    people = data.get("people")
    if not isinstance(people, list):
        return False
    changed = False
    for person in people:
        for key, val in list(person.items()):
            if key.endswith(KEYPOINT_FIELD_SUFFIX) and isinstance(val, list):
                person[key] = scale_keypoints(val)
                changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(data, f)
    return changed


def main():
    if not ROOT.exists():
        sys.exit(f"Dataset folder not found: {ROOT}. Run download.py first.")

    total_imgs = 0
    resized = 0
    total_jsons = 0
    rescaled = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            total_imgs += 1
            try:
                if resize_file(path):
                    resized += 1
            except Exception as e:
                print(f"Skipping {path}: {e}")
            if total_imgs % 500 == 0:
                print(f"Processed {total_imgs} images...")
        elif suffix == ".json" and "pose" in str(path).lower():
            total_jsons += 1
            try:
                if rescale_pose_json(path):
                    rescaled += 1
            except Exception as e:
                print(f"Skipping {path}: {e}")

    print(f"Done. Resized {resized}/{total_imgs} images; "
          f"rescaled {rescaled}/{total_jsons} pose JSONs under {ROOT}.")


if __name__ == "__main__":
    main()
