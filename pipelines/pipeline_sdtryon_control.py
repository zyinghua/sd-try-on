# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
StableDiffusionSDTryOnControlPipeline

Virtual try-on pipeline for the architecture trained in `train.py`:

  pose image  --PoseEncoder-->  pose_latents (4 ch) ---+
                                                       |-- concat --> UNet.conv_in (8 ch)
  noise latent (4 ch) ---------------------------------+

  cloth image -- ControlNet (empty-text conditioned) --> multi-scale features
                                                           |
                                                           v
                                         one CrossAttnZeroConvBlock per UNet up_block,
                                         injecting cloth features via cross-attention.

  (optional) cloth image --CLIP vision--> image_embeds --ImageProjModel-->
                                                           4 extra context tokens
                                                           appended to text tokens.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Architecture blocks (mirrored from train.py; duplicated to avoid a circular
# import between train.py <-> pipeline module).
# ---------------------------------------------------------------------------

class PoseEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.body(x)


class _FFN_GEGLU(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner * 2)
        self.proj_out = nn.Linear(inner, dim)

    def forward(self, x):
        a, b = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(F.gelu(a) * b)


class _Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context=None):
        ctx = x if context is None else context
        B, Nq, _ = x.shape
        Nk = ctx.shape[1]
        q = self.to_q(x).view(B, Nq, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(ctx).view(B, Nk, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(ctx).view(B, Nk, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, self.heads * self.dim_head)
        return self.to_out(out)


class _BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _Attention(dim, context_dim=None, heads=heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _Attention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FFN_GEGLU(dim)

    def forward(self, x, context):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x


def _choose_gn_groups(channels, prefer=32):
    g = min(prefer, channels)
    while channels % g != 0:
        g -= 1
    return max(g, 1)


class CrossAttnZeroConvBlock(nn.Module):
    def __init__(self, hidden_size, context_channels, heads=8, groups=32):
        super().__init__()
        assert hidden_size % heads == 0
        dim_head = hidden_size // heads
        inner_dim = heads * dim_head
        self.norm = nn.GroupNorm(_choose_gn_groups(hidden_size, groups), hidden_size, eps=1e-6)
        self.proj_in = nn.Conv2d(hidden_size, inner_dim, kernel_size=1)
        self.transformer = _BasicTransformerBlock(
            dim=inner_dim, context_dim=context_channels, heads=heads, dim_head=dim_head
        )
        self.proj_out = nn.Conv2d(inner_dim, hidden_size, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.zero_conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, unet_feat, cloth_feat):
        B, _, H, W = unet_feat.shape
        residual = unet_feat
        x = self.norm(unet_feat)
        x = self.proj_in(x)
        x = x.flatten(2).transpose(1, 2)
        ctx = cloth_feat.flatten(2).transpose(1, 2) if cloth_feat.dim() == 4 else cloth_feat
        x = self.transformer(x, ctx)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.proj_out(x)
        x = x + residual
        x = self.zero_conv(x)
        return residual + x


def select_cloth_feats_for_up_blocks(down_block_res_samples, layers_per_block: int = 2):
    """Cloth-context feats in UNet pop order, one per injection site."""
    skip = layers_per_block + 1
    return list(reversed(down_block_res_samples))[skip:]


class IPAdapter(nn.Module):
    """Wraps the cloth-image projection (4 tokens) + IP-Adapter attention modules."""

    def __init__(self, image_proj_model, adapter_modules, ckpt_path: Optional[str] = None):
        super().__init__()
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    def forward(self, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        return torch.cat([encoder_hidden_states, ip_tokens], dim=1), ip_tokens


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class StableDiffusionSDTryOnControlPipeline(DiffusionPipeline):
    """Inference pipeline for the try-on architecture trained in `train.py`."""

    _optional_components = ["image_encoder", "clip_image_processor", "ip_adapter"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        pose_encoder: PoseEncoder,
        cloth_inject_blocks: nn.ModuleList,
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        clip_image_processor: Optional[CLIPImageProcessor] = None,
        ip_adapter: Optional[IPAdapter] = None,
    ):
        super().__init__()

        self.register_modules(
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

        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self._cloth_feats: Optional[List[torch.Tensor]] = None
        self._hook_handles: List[Any] = []
        self._install_cloth_hooks()

    # --- hook management ---------------------------------------------------

    def _install_cloth_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        L = len(self.unet.config.block_out_channels)
        layers_per_block = self.unet.config.layers_per_block
        resnets_per_up_block = layers_per_block + 1
        expected = (L - 1) * resnets_per_up_block
        assert len(self.cloth_inject_blocks) == expected, (
            f"cloth_inject_blocks ({len(self.cloth_inject_blocks)}) must equal "
            f"(L-1) * (layers_per_block+1) = {expected}"
        )

        pipeline = self
        flat = 0
        for up_idx in range(1, L):
            up_block = self.unet.up_blocks[up_idx]
            for r_idx in range(resnets_per_up_block):
                inj = self.cloth_inject_blocks[flat]

                def make_hook(i, inj_ref):
                    def hook(_module, _inputs, output):
                        feats = pipeline._cloth_feats
                        if feats is None or feats[i] is None:
                            return output
                        return inj_ref(output, feats[i])
                    return hook

                self._hook_handles.append(
                    up_block.resnets[r_idx].register_forward_hook(make_hook(flat, inj))
                )
                flat += 1

    # --- prompt encoding ---------------------------------------------------

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            else:
                uncond_tokens = list(negative_prompt)

            uncond_inputs = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_inputs.input_ids.to(device), return_dict=False)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, negative_prompt_embeds

    def _empty_text_embed(self, device, dtype, batch_size):
        tok = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = self.text_encoder(tok.input_ids.to(device), return_dict=False)[0]
        return emb.to(dtype=dtype)

    # --- image preparation -------------------------------------------------

    def _prepare_condition_image(self, image, height, width, batch_size, device, dtype, do_cfg):
        """Normalize to [0, 1] 3-channel RGB, then replicate for CFG."""
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        if image.shape[0] == 1:
            image = image.repeat(batch_size, 1, 1, 1)
        elif image.shape[0] != batch_size:
            raise ValueError(f"image batch ({image.shape[0]}) does not match batch_size ({batch_size})")
        image = image.to(device=device, dtype=dtype)
        if do_cfg:
            image = torch.cat([image] * 2, dim=0)
        return image

    def _prepare_pose_image(self, image, height, width, batch_size, device, dtype, do_cfg):
        """Pose input goes to PoseEncoder; keep as [0, 1] 3-channel RGB."""
        return self._prepare_condition_image(image, height, width, batch_size, device, dtype, do_cfg)

    def _prepare_cloth_control_image(self, image, height, width, batch_size, device, dtype, do_cfg):
        """Cloth input goes to ControlNet; keep as [0, 1] 3-channel RGB."""
        return self._prepare_condition_image(image, height, width, batch_size, device, dtype, do_cfg)

    # --- IP-Adapter helper -------------------------------------------------

    def _get_ip_tokens(self, cloth_image, batch_size, device, dtype, do_cfg):
        """Return (cond_ip_tokens, uncond_ip_tokens) each of shape (B, 4, D)."""
        assert self.ip_adapter is not None and self.image_encoder is not None and self.clip_image_processor is not None
        if isinstance(cloth_image, list):
            clip_px = self.clip_image_processor(images=cloth_image, return_tensors="pt").pixel_values
        else:
            clip_px = self.clip_image_processor(images=cloth_image, return_tensors="pt").pixel_values
            if clip_px.shape[0] == 1:
                clip_px = clip_px.expand(batch_size, -1, -1, -1)
        clip_px = clip_px.to(device=device, dtype=dtype)
        with torch.no_grad():
            image_embeds = self.image_encoder(clip_px).image_embeds
            cond_tokens = self.ip_adapter.image_proj_model(image_embeds)
            uncond_tokens = self.ip_adapter.image_proj_model(torch.zeros_like(image_embeds)) if do_cfg else None
        return cond_tokens, uncond_tokens

    # --- latent prep -------------------------------------------------------

    def _prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # --- main entry --------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        pose_image: Union[PIL.Image.Image, torch.Tensor, List[PIL.Image.Image]],
        cloth_image: Union[PIL.Image.Image, torch.Tensor, List[PIL.Image.Image]],
        cloth_clip_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        controlnet_conditioning_scale: float = 1.0,
    ):
        device = self._execution_device
        # Use the UNet body's dtype as the pipeline dtype; conv_in may be fp32
        # (trainable) so reading from conv_in would not be representative.
        try:
            dtype = next(self.unet.down_blocks.parameters()).dtype
        except StopIteration:
            dtype = self.unet.dtype

        # Resolve image size
        if height is None or width is None:
            if isinstance(pose_image, PIL.Image.Image):
                width, height = pose_image.size
            elif isinstance(pose_image, list) and isinstance(pose_image[0], PIL.Image.Image):
                width, height = pose_image[0].size
            else:
                raise ValueError("height/width must be given when pose_image is a tensor.")
        height = (height // self.vae_scale_factor) * self.vae_scale_factor
        width = (width // self.vae_scale_factor) * self.vae_scale_factor

        # Encode prompt
        do_cfg = guidance_scale > 1.0
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self._encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompt,
            )

        batch_size = prompt_embeds.shape[0]

        # IP-Adapter: append 4 cloth CLIP tokens to text embeds (both branches)
        if self.ip_adapter is not None:
            assert cloth_clip_image is not None or cloth_image is not None, (
                "IP-Adapter requires a cloth image (pass cloth_clip_image or cloth_image)."
            )
            cloth_for_clip = cloth_clip_image if cloth_clip_image is not None else cloth_image
            cond_ip_tokens, uncond_ip_tokens = self._get_ip_tokens(
                cloth_for_clip, batch_size, device, dtype, do_cfg
            )
            prompt_embeds = torch.cat([prompt_embeds, cond_ip_tokens], dim=1)
            if do_cfg:
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_ip_tokens], dim=1)

        # Assemble the full encoder_hidden_states for the UNet (cond first or uncond first?
        # Diffusers convention: [uncond; cond] when doing CFG).
        if do_cfg:
            unet_encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            unet_encoder_hidden_states = prompt_embeds
        unet_encoder_hidden_states = unet_encoder_hidden_states.to(dtype=dtype, device=device)

        # Cloth-ControlNet uses EMPTY text for both branches (encoder-only conditioning).
        controlnet_text_ctx = self._empty_text_embed(device, dtype, batch_size=batch_size * (2 if do_cfg else 1))

        # Prepare pose/cloth conditioning images
        pose_pixels = self._prepare_pose_image(
            pose_image, height, width, batch_size, device, dtype, do_cfg
        )
        cloth_pixels = self._prepare_cloth_control_image(
            cloth_image, height, width, batch_size, device, dtype, do_cfg
        )

        # Init latents. `batch_size` already includes num_images_per_prompt
        # because _encode_prompt repeated prompt embeds.
        num_channels_latents = self.vae.config.latent_channels
        latents = self._prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # Scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Extra step kwargs (eta only relevant to DDIM-style schedulers)
        accepts_eta = "eta" in inspect.signature(self.scheduler.step).parameters
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        layers_per_block = self.unet.config.layers_per_block

        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Pose: encode to 4-ch latent and concat
            pose_latents = self.pose_encoder(pose_pixels.to(dtype=dtype))
            unet_input = torch.cat([latent_model_input, pose_latents], dim=1)

            # ControlNet on cloth with empty text
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=controlnet_text_ctx,
                controlnet_cond=cloth_pixels,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )
            cloth_feats = select_cloth_feats_for_up_blocks(
                down_block_res_samples, layers_per_block=layers_per_block,
            )
            cloth_feats = [f.to(dtype=dtype) if f is not None else None for f in cloth_feats]

            # Set cloth features for the hooks, run UNet, clear.
            self._cloth_feats = cloth_feats
            noise_pred = self.unet(
                unet_input,
                t,
                encoder_hidden_states=unet_encoder_hidden_states,
                return_dict=False,
            )[0]
            self._cloth_feats = None

            # CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # Decode
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image, None)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
