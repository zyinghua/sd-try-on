import os
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


# ----------------------------- Configuration -----------------------------
IMAGE_ID = "00025"

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
IMAGE_ENCODER_PATH = "patrickjohncyh/fashion-clip"
# Directory written by accelerator.save_state(...) (contains model*.safetensors)
CHECKPOINT_PATH = "/root/autodl-tmp/sdtryon-model/checkpoint-40000"

POSE_IMAGE = f"/root/autodl-tmp/data/sdtryon/train/image-densepose/{IMAGE_ID}_00.jpg"
CLOTH_IMAGE = f"/root/autodl-tmp/data/sdtryon/train/cloth/00014_00.jpg"
SOURCE_IMAGE = f"/root/autodl-tmp/data/sdtryon/train/image/{IMAGE_ID}_00.jpg"
MASK_IMAGE = f"/root/autodl-tmp/data/sdtryon/train/image_cloth_mask/{IMAGE_ID}_00.png"


PROMPT = "a person wearing a garment"
NEGATIVE_PROMPT = (
    "noisy, blurry, low contrast, watermark, painting, drawing, illustration, "
    "glitch, deformed, mutated, ugly, disfigured"
)

OUTPUT_PATH = "/root/autodl-tmp/sdtryon-eval"
SAMPLE_NUM = 4
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
CONTROLNET_CONDITIONING_SCALE = 1.0
SEED = None
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inpainting: if SOURCE_IMAGE and MASK_IMAGE both exist, the pipeline runs
# blended-diffusion latent inpainting (white mask = repaint region).
# Set STRENGTH < 1.0 to preserve more of the source image globally.
ENABLE_INPAINT = True
STRENGTH = 1.0

ENABLE_POSE_OVERLAY = True
POSE_OVERLAY_ALPHA = 0.4

# Number of IP tokens -- must match training (ImageProjModel.clip_extra_context_tokens=4).
NUM_IP_TOKENS = 4


# ----------------------- Architecture helpers ----------------------------
# These must stay in sync with train_control.py.


def _patch_unet_conv_in(unet: UNet2DConditionModel, new_in_channels: int) -> None:
    """Replicates `patch_unet_conv_in` from train_control.py."""
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
    """Mirrors UNetWithClothInjection.__init__ from train_control.py."""
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
    """Rebuild the IP-Adapter attention-processor layout from train_control.py."""
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
            # Warm-start (matches training). Values from model_3 will override these.
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
    """Load `model_1.safetensors` into (unet, cloth_inject_blocks)."""
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
        # Tolerate (e.g. removed attn processor temporaries); log for sanity.
        print(f"[warn] {len(missing)} missing keys in unet load (first: {missing[:3]})")
    cloth_inject_blocks.load_state_dict(cloth_sd, strict=True)


def overlay_pose_on_image(generated, pose, alpha=0.4):
    if generated.size != pose.size:
        pose = pose.resize(generated.size, Image.Resampling.LANCZOS)
    return Image.blend(generated.convert("RGBA"), pose.convert("RGBA"), alpha).convert("RGB")


# ------------------------------ Main -------------------------------------
def main():
    for label, p in [
        ("CHECKPOINT_PATH", CHECKPOINT_PATH),
        ("POSE_IMAGE", POSE_IMAGE),
        ("CLOTH_IMAGE", CLOTH_IMAGE),
    ]:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    shard = {
        "controlnet":  os.path.join(CHECKPOINT_PATH, "model.safetensors"),
        "unet_wrap":   os.path.join(CHECKPOINT_PATH, "model_1.safetensors"),
        "pose":        os.path.join(CHECKPOINT_PATH, "model_2.safetensors"),
        "ip_adapter":  os.path.join(CHECKPOINT_PATH, "model_3.safetensors"),
    }
    for name, p in shard.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint shard '{name}': {p}")

    print(f"Loading from checkpoint: {CHECKPOINT_PATH}")

    # 1. Base components (architecture-only; we overwrite weights below).
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    # ControlNet architecture is `from_unet` at train time; match it here.
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3)

    # 2. Reproduce train_control.py's UNet modifications.
    pose_latent_channels = unet.config.in_channels  # 4 for SD 2.1
    _patch_unet_conv_in(unet, new_in_channels=unet.config.in_channels + pose_latent_channels)

    pose_encoder = PoseEncoder(in_channels=3, out_channels=pose_latent_channels)
    cloth_inject_blocks = _build_cloth_inject_blocks(unet)

    # IP-Adapter
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

    # 3. Load checkpoint shards.
    controlnet.load_state_dict(load_file(shard["controlnet"]), strict=True)
    _split_and_load_wrapped_unet(shard["unet_wrap"], unet, cloth_inject_blocks)
    pose_encoder.load_state_dict(load_file(shard["pose"]), strict=True)

    ip_sd = load_file(shard["ip_adapter"])
    ip_proj_sd = {k[len("image_proj_model."):]: v for k, v in ip_sd.items() if k.startswith("image_proj_model.")}
    ip_adap_sd = {k[len("adapter_modules."):]: v for k, v in ip_sd.items() if k.startswith("adapter_modules.")}
    ip_adapter.image_proj_model.load_state_dict(ip_proj_sd, strict=True)
    ip_adapter.adapter_modules.load_state_dict(ip_adap_sd, strict=True)

    # 4. Cast + move.
    vae.to(DEVICE, dtype=DTYPE).eval()
    text_encoder.to(DEVICE, dtype=DTYPE).eval()
    unet.to(DEVICE, dtype=DTYPE).eval()
    controlnet.to(DEVICE, dtype=DTYPE).eval()
    pose_encoder.to(DEVICE, dtype=DTYPE).eval()
    cloth_inject_blocks.to(DEVICE, dtype=DTYPE).eval()
    image_encoder.to(DEVICE, dtype=DTYPE).eval()
    ip_adapter.to(DEVICE, dtype=DTYPE).eval()

    # 5. Assemble pipeline.
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
    pipe.set_progress_bar_config(disable=False)

    # 6. Inputs.
    pose_image = Image.open(POSE_IMAGE).convert("RGB")
    cloth_image = Image.open(CLOTH_IMAGE).convert("RGB")

    # Optional inpainting inputs.
    source_image = mask_image = None
    if ENABLE_INPAINT and SOURCE_IMAGE and MASK_IMAGE and os.path.exists(SOURCE_IMAGE) and os.path.exists(MASK_IMAGE):
        source_image = Image.open(SOURCE_IMAGE).convert("RGB")
        mask_image = Image.open(MASK_IMAGE).convert("L")
        print(f"Inpainting enabled (strength={STRENGTH}):")
        print(f"  source: {SOURCE_IMAGE}")
        print(f"  mask:   {MASK_IMAGE}")
    else:
        print("Inpainting disabled (running full text-to-image denoise).")

    generator = (
        torch.Generator(device=DEVICE).manual_seed(SEED) if SEED is not None else None
    )

    print(f"Generating {SAMPLE_NUM} sample(s)...")
    out = pipe(
        prompt=[PROMPT] * SAMPLE_NUM,
        negative_prompt=[NEGATIVE_PROMPT] * SAMPLE_NUM,
        pose_image=pose_image,
        cloth_image=cloth_image,
        cloth_clip_image=cloth_image,
        image=source_image,
        mask_image=mask_image,
        strength=STRENGTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        generator=generator,
    )
    images = out.images

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(OUTPUT_PATH, f"sdtryon_control_{i}.png")
        img.save(out_path)
        print(f"Saved: {out_path}")
        if ENABLE_POSE_OVERLAY:
            ov = os.path.join(OUTPUT_PATH, f"sdtryon_control_{i}_pose_overlay.png")
            overlay_pose_on_image(img, pose_image, alpha=POSE_OVERLAY_ALPHA).save(ov)
            print(f"Saved: {ov}")

    print("Inference complete!")


if __name__ == "__main__":
    main()
