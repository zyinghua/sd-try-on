"""Predict an upper-body garment mask for a real photo of a person.

By default, masks are written to `dataset/garment_masks/<stem>.png` (0/255,
L-mode PNGs at the input image's original resolution).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
UPPER_CLOTHES = 4   # shirts, t-shirts, coats, jackets, sweaters
DRESS = 7           # dresses (full-length upper garments)
DEFAULT_LABELS = (UPPER_CLOTHES, DRESS)

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATASET_DIR / "garment_masks"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise SystemExit(f"Input not found: {path}")


@torch.inference_mode()
def predict_mask(
    image: Image.Image,
    model: AutoModelForSemanticSegmentation,
    processor: SegformerImageProcessor,
    device: torch.device,
    labels: tuple[int, ...],
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    logits = model(**inputs).logits  # (1, C, h, w) at model resolution
    
    upsampled = F.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W) int64
    label_arr = np.asarray(labels, dtype=pred.dtype)
    return (np.isin(pred, label_arr).astype(np.uint8) * 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Image file, or a directory of images (recursed).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for masks (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="ATR class ids to include in the mask (default: 4 7 — upper-clothes, dress)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda > mps > cpu)",
    )
    args = parser.parse_args()

    inputs = collect_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No images found under {args.input}")

    device = pick_device(args.device)
    print(f"Loading {MODEL_ID} on {device} ...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = tuple(args.labels)

    for src in tqdm(inputs, desc="masking"):
        image = Image.open(src).convert("RGB")
        mask = predict_mask(image, model, processor, device, labels)
        Image.fromarray(mask, mode="L").save(args.out / f"{src.stem}.png")

    print(f"Wrote {len(inputs)} masks to {args.out} (labels={list(labels)})")


if __name__ == "__main__":
    main()
