import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from pipelines.pipeline_sdtryon_control import (
    CrossAttnZeroConvBlock,
    IPAdapter,
    PoseEncoder,
    StableDiffusionSDTryOnControlPipeline,
)
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor


# ----------------------------- Defaults ----------------------------------
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
IMAGE_ENCODER_PATH = "patrickjohncyh/fashion-clip"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/sdtryon-model/checkpoint-40000"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/data/sdtryon"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/data/sdtryon"

PROMPT = "a person wearing a garment"
NEGATIVE_PROMPT = (
    "noisy, blurry, low contrast, watermark, painting, drawing, illustration, "
    "glitch, deformed, mutated, ugly, disfigured"
)

NUM_IP_TOKENS = 4
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------- Architecture helpers ----------------------------

def _patch_unet_conv_in(unet: UNet2DConditionModel, new_in_channels: int) -> None:
    orig = unet.conv_in
    new = nn.Conv2d(
        new_in_channels,
        orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : orig.in_channels, :, :].copy_(orig.weight)
        if orig.bias is not None:
            new.bias.copy_(orig.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=new_in_channels)


def _build_cloth_inject_blocks(unet: UNet2DConditionModel) -> nn.ModuleList:
    boc = list(unet.config.block_out_channels)
    L = len(boc)
    up_hidden = list(reversed(boc))
    layers_per_block = unet.config.layers_per_block
    resnets_per_up_block = layers_per_block + 1

    blocks = nn.ModuleList()
    for up_idx in range(1, L):
        hidden = up_hidden[up_idx]
        for r_idx in range(resnets_per_up_block):
            if r_idx < resnets_per_up_block - 1:
                ctx_level = L - 1 - up_idx
            else:
                ctx_level = max(L - 2 - up_idx, 0)
            blocks.append(CrossAttnZeroConvBlock(hidden_size=hidden, context_channels=boc[ctx_level]))
    return blocks


def _install_ip_adapter_attn_processors(unet: UNet2DConditionModel) -> nn.ModuleList:
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise RuntimeError(f"Unexpected attn processor name: {name}")

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
            attn_procs[name].load_state_dict(
                {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
            )
    unet.set_attn_processor(attn_procs)
    return nn.ModuleList(unet.attn_processors.values())


def _split_and_load_wrapped_unet(
    path: str, unet: UNet2DConditionModel, cloth_inject_blocks: nn.ModuleList
) -> None:
    sd = load_file(path)
    unet_sd, cloth_sd = {}, {}
    for k, v in sd.items():
        if k.startswith("unet."):
            unet_sd[k[len("unet."):]] = v
        elif k.startswith("cloth_inject_blocks."):
            cloth_sd[k[len("cloth_inject_blocks."):]] = v
        else:
            raise KeyError(f"Unexpected key in {path}: {k}")
    missing, unexpected = unet.load_state_dict(unet_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in unet state: {unexpected[:5]} ...")
    if missing:
        print(f"[warn] {len(missing)} missing keys in unet load (first: {missing[:3]})")
    cloth_inject_blocks.load_state_dict(cloth_sd, strict=True)


# --------------------------- Pipeline build ------------------------------


def build_pipeline(checkpoint_path: str, device: str, dtype: torch.dtype) -> StableDiffusionSDTryOnControlPipeline:
    shard = {
        "controlnet":  os.path.join(checkpoint_path, "model.safetensors"),
        "unet_wrap":   os.path.join(checkpoint_path, "model_1.safetensors"),
        "pose":        os.path.join(checkpoint_path, "model_2.safetensors"),
        "ip_adapter":  os.path.join(checkpoint_path, "model_3.safetensors"),
    }
    for name, p in shard.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint shard '{name}': {p}")

    print(f"Loading from checkpoint: {checkpoint_path}")

    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3)

    pose_latent_channels = unet.config.in_channels
    _patch_unet_conv_in(unet, new_in_channels=unet.config.in_channels + pose_latent_channels)

    pose_encoder = PoseEncoder(in_channels=3, out_channels=pose_latent_channels)
    cloth_inject_blocks = _build_cloth_inject_blocks(unet)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(IMAGE_ENCODER_PATH)
    image_encoder.requires_grad_(False)
    clip_image_processor = CLIPImageProcessor.from_pretrained(IMAGE_ENCODER_PATH)

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=NUM_IP_TOKENS,
    )
    ip_adapter_modules = _install_ip_adapter_attn_processors(unet)
    ip_adapter = IPAdapter(image_proj_model=image_proj_model, adapter_modules=ip_adapter_modules)

    controlnet.load_state_dict(load_file(shard["controlnet"]), strict=True)
    _split_and_load_wrapped_unet(shard["unet_wrap"], unet, cloth_inject_blocks)
    pose_encoder.load_state_dict(load_file(shard["pose"]), strict=True)

    ip_sd = load_file(shard["ip_adapter"])
    ip_proj_sd = {k[len("image_proj_model."):]: v for k, v in ip_sd.items() if k.startswith("image_proj_model.")}
    ip_adap_sd = {k[len("adapter_modules."):]: v for k, v in ip_sd.items() if k.startswith("adapter_modules.")}
    ip_adapter.image_proj_model.load_state_dict(ip_proj_sd, strict=True)
    ip_adapter.adapter_modules.load_state_dict(ip_adap_sd, strict=True)

    vae.to(device, dtype=dtype).eval()
    text_encoder.to(device, dtype=dtype).eval()
    unet.to(device, dtype=dtype).eval()
    controlnet.to(device, dtype=dtype).eval()
    pose_encoder.to(device, dtype=dtype).eval()
    cloth_inject_blocks.to(device, dtype=dtype).eval()
    image_encoder.to(device, dtype=dtype).eval()
    ip_adapter.to(device, dtype=dtype).eval()

    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    pipe = StableDiffusionSDTryOnControlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        pose_encoder=pose_encoder,
        cloth_inject_blocks=cloth_inject_blocks,
        image_encoder=image_encoder,
        clip_image_processor=clip_image_processor,
        ip_adapter=ip_adapter,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# --------------------------- Sampling helpers ----------------------------


def list_images(folder: str) -> list:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Expected folder not found: {folder}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    return sorted(files)


def sample_pairs(pose_files: list, cloth_files: list, n: int, rng: random.Random) -> list:
    """Sample n (pose, cloth) pairs with replacement (independent random draws)."""
    return [
        (rng.choice(pose_files), rng.choice(cloth_files))
        for _ in range(n)
    ]


# ------------------------------ Main -------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", "-n", type=int, required=True, help="Number of images to generate.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint-XXXXX/.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="Dataset root (containing train/ and test/).")
    parser.add_argument("--output-root", default=None,
                        help="Where to put generated_<N>_images/ (defaults to --data-root).")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Which split to draw densepose/cloth from.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for sampling pose/cloth pairs and the diffusion generator.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    pose_dir = os.path.join(args.data_root, args.split, "image-densepose")
    cloth_dir = os.path.join(args.data_root, args.split, "cloth")
    pose_files = list_images(pose_dir)
    cloth_files = list_images(cloth_dir)
    print(f"Pose pool:  {len(pose_files)} images @ {pose_dir}")
    print(f"Cloth pool: {len(cloth_files)} images @ {cloth_dir}")

    rng = random.Random(args.seed)
    pairs = sample_pairs(pose_files, cloth_files, args.num, rng)

    output_root = args.output_root or args.data_root
    out_dir = os.path.join(output_root, f"generated_{args.num}_images")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing samples to: {out_dir}")

    pipe = build_pipeline(args.checkpoint, device=device, dtype=dtype)

    # Width of the index in filenames so a directory listing sorts correctly.
    width = max(5, len(str(args.num - 1)))

    generated = 0
    for batch_start in range(0, args.num, args.batch_size):
        batch = pairs[batch_start: batch_start + args.batch_size]
        bsz = len(batch)

        pose_imgs = [Image.open(os.path.join(pose_dir, p)).convert("RGB") for p, _ in batch]
        cloth_imgs = [Image.open(os.path.join(cloth_dir, c)).convert("RGB") for _, c in batch]

        generator = torch.Generator(device=device).manual_seed(args.seed + batch_start)

        out = pipe(
            prompt=[PROMPT] * bsz,
            negative_prompt=[NEGATIVE_PROMPT] * bsz,
            pose_image=pose_imgs,
            cloth_image=cloth_imgs,
            cloth_clip_image=cloth_imgs,
            image=None,
            mask_image=None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            generator=generator,
        )

        for j, img in enumerate(out.images):
            idx = batch_start + j
            fname = f"{idx:0{width}d}.png"
            img.save(os.path.join(out_dir, fname))
            generated += 1
        print(f"  [{generated}/{args.num}] saved")

    print(f"Done. {generated} images written to {out_dir}")


if __name__ == "__main__":
    main()
