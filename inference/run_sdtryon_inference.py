import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from pipelines.pipeline_sdtryon import StableDiffusionIDControlPipeline

# ----------------------------- Configuration -----------------------------
# Which cloth-encoder head was trained:
#   "clip_resampler": Resampler (train_clip_resampler.py), CLIP penultimate patch tokens
#   "control":        ImageProjModel (train.py), CLIP projected image_embeds
MODE = "clip_resampler"

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
IMAGE_ENCODER_PATH = "patrickjohncyh/fashion-clip"
CONTROLNET_PATH = "/root/autodl-tmp/sdtryon-model/checkpoint-30000/controlnet"                  # ends with /controlnet
IP_ADAPTER_PATH = "/root/autodl-tmp/sdtryon-model/checkpoint-30000/ip_adapter/ip_adapter.bin"   # ends with /ip_adapter/ip_adapter.bin

POSE_IMAGE = "/root/autodl-tmp/data/sdtryon/train/image-densepose/00000_00.jpg"
CLOTH_IMAGE = "/root/autodl-tmp/data/sdtryon/train/cloth/00000_00.jpg"
PROMPT = "a person wearing a garment"
NEGATIVE_PROMPT = "noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, ugly, disfigured"

OUTPUT_PATH = "/root/autodl-tmp/sdtryon-eval"
SAMPLE_NUM = 4
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
CONTROLNET_CONDITIONING_SCALE = 1.0
IP_ADAPTER_SCALE = 1.0
SEED = None
DTYPE = torch.float16

# Number of IP tokens (must match training).
# Defaults: clip_resampler=16 (Resampler num_queries), control=4 (ImageProjModel extra tokens).
NUM_IP_TOKENS = 16 if MODE == "clip_resampler" else 4

ENABLE_POSE_OVERLAY = True
POSE_OVERLAY_ALPHA = 0.4


def encode_cloth_image(cloth_pil, image_encoder, clip_image_processor, mode, device, dtype):
    pixel_values = clip_image_processor(images=cloth_pil, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    with torch.no_grad():
        if mode == "clip_resampler":
            return image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]  # (1, N, D)
        return image_encoder(pixel_values).image_embeds  # (1, D)


def overlay_pose_on_image(generated, pose, alpha=0.4):
    if generated.size != pose.size:
        pose = pose.resize(generated.size, Image.Resampling.LANCZOS)
    g = generated.convert("RGBA")
    p = pose.convert("RGBA")
    return Image.blend(g, p, alpha).convert("RGB")


# ----------------------------- Main --------------------------------------
def main():
    if MODE not in {"clip_resampler", "control"}:
        raise ValueError(f"MODE must be 'clip_resampler' or 'control', got {MODE!r}")
    for label, p in [("CONTROLNET_PATH", CONTROLNET_PATH),
                     ("IP_ADAPTER_PATH", IP_ADAPTER_PATH),
                     ("POSE_IMAGE", POSE_IMAGE),
                     ("CLOTH_IMAGE", CLOTH_IMAGE)]:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    print(f"MODE: {MODE}")
    print(f"Loading ControlNet from: {CONTROLNET_PATH}")
    print(f"Loading IP-Adapter from: {IP_ADAPTER_PATH}")
    print(f"Pose image:  {POSE_IMAGE}")
    print(f"Cloth image: {CLOTH_IMAGE}")
    print(f"Prompt:      {PROMPT}")

    # 1. ControlNet + base pipeline
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=DTYPE)
    pipe = StableDiffusionIDControlPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # 2. CLIP image encoder for cloth
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(IMAGE_ENCODER_PATH).to(dtype=DTYPE)
    image_encoder.requires_grad_(False)
    clip_image_processor = CLIPImageProcessor.from_pretrained(IMAGE_ENCODER_PATH)

    # 3. Load IP-Adapter
    if MODE == "clip_resampler":
        pipe.load_ip_adapter_clip_resampler(
            IP_ADAPTER_PATH,
            clip_embeddings_dim=image_encoder.config.hidden_size,
            num_tokens=NUM_IP_TOKENS,
            scale=IP_ADAPTER_SCALE,
        )
    else:
        pipe.load_ip_adapter(
            IP_ADAPTER_PATH,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            num_tokens=NUM_IP_TOKENS,
            scale=IP_ADAPTER_SCALE,
        )

    pipe.enable_model_cpu_offload()

    # 4. Inputs
    pose_image = load_image(POSE_IMAGE)
    cloth_image = load_image(CLOTH_IMAGE)
    device = pipe._execution_device

    image_encoder = image_encoder.to(device)
    cloth_embeds = encode_cloth_image(cloth_image, image_encoder, clip_image_processor, MODE, device, DTYPE)

    # Tile to SAMPLE_NUM so pipeline's prepare_image_embeds uses the 3D-safe branch.
    if cloth_embeds.dim() == 2:
        cloth_embeds = cloth_embeds.expand(SAMPLE_NUM, -1).contiguous()
    else:
        cloth_embeds = cloth_embeds.expand(SAMPLE_NUM, -1, -1).contiguous()

    prompts = [PROMPT] * SAMPLE_NUM
    negatives = [NEGATIVE_PROMPT] * SAMPLE_NUM

    generator = torch.Generator(device="cpu").manual_seed(SEED) if SEED is not None else None

    print(f"Generating {SAMPLE_NUM} sample(s)...")
    images = pipe(
        prompt=prompts,
        negative_prompt=negatives,
        control_image=pose_image,
        image_embeds=cloth_embeds,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        generator=generator,
    ).images

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for i, img in enumerate(images):
        out = os.path.join(OUTPUT_PATH, f"sdtryon_{MODE}_{i}.png")
        img.save(out)
        print(f"Saved: {out}")
        if ENABLE_POSE_OVERLAY:
            ov_path = os.path.join(OUTPUT_PATH, f"sdtryon_{MODE}_{i}_pose_overlay.png")
            overlay_pose_on_image(img, pose_image, alpha=POSE_OVERLAY_ALPHA).save(ov_path)
            print(f"Saved: {ov_path}")

    print("Inference complete!")


if __name__ == "__main__":
    main()
