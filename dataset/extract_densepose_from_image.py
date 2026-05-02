import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from densepose import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import DensePoseResultExtractor


CONFIG_NAME = "densepose_rcnn_R_50_FPN_s1x.yaml"
BASE_CONFIG_NAME = "Base-DensePose-RCNN-FPN.yaml"
CONFIG_URL_BASE = (
    "https://raw.githubusercontent.com/facebookresearch/detectron2/main/"
    "projects/DensePose/configs/"
)
DEFAULT_WEIGHTS = (
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/"
    "165712039/model_final_162be9.pkl"
)

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATASET_DIR / "densepose_outputs"
CONFIG_CACHE = DATASET_DIR / ".densepose_configs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise SystemExit(f"Input not found: {path}")


def ensure_config(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in (CONFIG_NAME, BASE_CONFIG_NAME):
        target = cache_dir / fname
        if not target.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(CONFIG_URL_BASE + fname, target)
    return cache_dir / CONFIG_NAME


def build_predictor(
    cfg_path: Path, weights: str, device: str, score_thresh: float
) -> DefaultPredictor:
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(str(cfg_path))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return DefaultPredictor(cfg)


@torch.inference_mode()
def render_densepose(
    image_bgr: np.ndarray,
    predictor: DefaultPredictor,
    extractor: DensePoseResultExtractor,
    visualizer: DensePoseResultsFineSegmentationVisualizer,
) -> np.ndarray:
    """Run DensePose on `image_bgr` and render onto a black canvas of equal size."""
    H, W = image_bgr.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    instances = predictor(image_bgr)["instances"]
    if len(instances) == 0:
        return canvas

    # Keep the highest-scoring detection. VITON-HD images contain a single subject.
    best_idx = int(instances.scores.argmax().item())
    instances = instances[[best_idx]]

    extracted = extractor(instances)
    if extracted is None or extracted[0] is None:
        return canvas
    return visualizer.visualize(canvas, extracted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path,
        help="Image file, or a directory of images (recursed).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a densepose .yaml config "
             "(auto-downloads densepose_rcnn_R_50_FPN_s1x by default).",
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help="DensePose model weights (URL or local path).",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.7,
        help="Detection score threshold (default: 0.7).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: cuda > mps > cpu).",
    )
    parser.add_argument(
        "--ext", type=str, default="jpg", choices=("jpg", "png"),
        help="Output file extension (default: jpg, matching VITON-HD).",
    )
    args = parser.parse_args()

    inputs = collect_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No images found under {args.input}")

    cfg_path = args.config if args.config else ensure_config(CONFIG_CACHE)
    device = pick_device(args.device)
    print(f"Loading DensePose-RCNN on {device} ...")
    predictor = build_predictor(cfg_path, args.weights, device, args.score_thresh)
    extractor = DensePoseResultExtractor()
    # alpha=1.0 -> the colormap is fully opaque on top of the black canvas.
    visualizer = DensePoseResultsFineSegmentationVisualizer(alpha=1.0)

    args.out.mkdir(parents=True, exist_ok=True)

    for src in tqdm(inputs, desc="densepose"):
        bgr = cv2.imread(str(src))
        if bgr is None:
            print(f"Skipping unreadable image: {src}")
            continue
        rendered_bgr = render_densepose(bgr, predictor, extractor, visualizer)
        rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        out_path = args.out / f"{src.stem}.{args.ext}"
        if args.ext == "jpg":
            Image.fromarray(rgb).save(out_path, quality=95)
        else:
            Image.fromarray(rgb).save(out_path)

    print(f"Wrote {len(inputs)} densepose images to {args.out}")


if __name__ == "__main__":
    main()
