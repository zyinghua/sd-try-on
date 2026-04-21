import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import torch.nn as nn
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from pipelines.pipeline_sdtryon import StableDiffusionIDControlPipeline, IPAdapter
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


# ----------------------------- Configuration -----------------------------
# Which cloth-encoder head was trained:
#   "clip_resampler": Resampler (train_clip_resampler.py), CLIP penultimate patch tokens
#   "control":        ImageProjModel (train.py), CLIP projected image_embeds
MODE = "clip_resampler"

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
IMAGE_ENCODER_PATH = "patrickjohncyh/fashion-clip"
CONTROLNET_PATH = "/root/autodl-tmp/sdtryon-model/controlnet"                  # ends with /controlnet
IP_ADAPTER_PATH = "/root/autodl-tmp/sdtryon-model/ip_adapter/ip_adapter.bin"   # ends with /ip_adapter/ip_adapter.bin

POSE_IMAGE = "/root/autodl-tmp/viton-hd-dataset/train/openpose_img/00000_00_rendered.png"
CLOTH_IMAGE = "/root/autodl-tmp/viton-hd-dataset/train/cloth/00000_00.jpg"
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


# ---------------------- Resampler (matches train_clip_resampler.py) -------
def _ff(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def _reshape_heads(x, heads):
    b, n, _ = x.shape
    return x.view(b, n, heads, -1).transpose(1, 2)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = _reshape_heads(q, self.heads)
        k = _reshape_heads(k, self.heads)
        v = _reshape_heads(v, self.heads)

        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(self, dim=1024, depth=4, dim_head=64, heads=16, num_queries=16,
                 embedding_dim=1280, output_dim=1024, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                _ff(dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))


# ----------------------------- Helpers -----------------------------------
def build_image_proj_model(mode: str, unet, image_encoder, num_ip_tokens: int) -> nn.Module:
    cross_attention_dim = unet.config.cross_attention_dim
    if mode == "clip_resampler":
        return Resampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=16,
            num_queries=num_ip_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
    if mode == "control":
        return ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_ip_tokens,
        )
    raise ValueError(f"Unknown MODE: {mode!r}. Expected 'clip_resampler' or 'control'.")


def build_ip_attn_processors(unet):
    """Mirror the training-time IPAttnProcessor setup; weights are overwritten when
    we later load `ip_adapter.bin` into the IPAdapter wrapper."""
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_tokens=NUM_IP_TOKENS,
            )
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    return torch.nn.ModuleList(unet.attn_processors.values())


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

    # 3. Build projection model + IP attention processors, wrap in IPAdapter (loads ckpt)
    image_proj_model = build_image_proj_model(MODE, pipe.unet, image_encoder, NUM_IP_TOKENS)
    image_proj_model = image_proj_model.to(dtype=DTYPE)
    ip_adapter_modules = build_ip_attn_processors(pipe.unet)
    ip_adapter_modules = ip_adapter_modules.to(dtype=DTYPE)

    ip_adapter = IPAdapter(
        image_proj_model=image_proj_model,
        adapter_modules=ip_adapter_modules,
        ckpt_path=IP_ADAPTER_PATH,
    )
    ip_adapter = ip_adapter.to(dtype=DTYPE)

    # Attach to pipeline (mirrors validation code path in training scripts)
    pipe.image_proj_model = ip_adapter.image_proj_model
    pipe.ip_adapter = ip_adapter
    if IP_ADAPTER_SCALE is not None:
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

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
