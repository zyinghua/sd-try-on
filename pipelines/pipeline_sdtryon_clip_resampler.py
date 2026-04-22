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
StableDiffusionTryOnClipResamplerPipeline

IP-Adapter-Plus variant of the sd-try-on pipeline. Consumes CLIP patch tokens
(penultimate hidden_states[-2]) through a Perceiver-style Resampler that
produces `num_queries` IP tokens, which are concatenated to the text tokens
via decoupled IPAttnProcessor cross-attention. ControlNet (pose) + scheduler
+ CFG machinery is forked verbatim from pipeline_sdtryon.py so this module
can be edited independently.
"""

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import ImageProjection, MultiControlNetModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import empty_device_cache, is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# Resampler (IP-Adapter-Plus projection head)
# ---------------------------------------------------------------------------
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
    x = x.view(b, n, heads, -1).transpose(1, 2)
    return x  # (b, heads, n, dim_head)


class PerceiverAttention(nn.Module):
    """Perceiver-style cross-attention: learned latents attend to concat(image tokens, latents)."""

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
    """IP-Adapter-Plus Resampler: CLIP patch tokens -> num_queries IP tokens at cross_attention_dim.

    Input:  (B, N_patches, embedding_dim)  -- typically CLIP hidden_states[-2]
    Output: (B, num_queries, output_dim)   -- IP tokens consumed by IPAttnProcessor
    """

    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=16,
        embedding_dim=1280,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    _ff(dim, mult=ff_mult),
                ])
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ...     "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""


class IPAdapter(torch.nn.Module):
    def __init__(self, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        logger.info(f"Successfully loaded pretrained IP-Adapter weights from checkpoint {ckpt_path}")

    def forward(self, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return encoder_hidden_states, ip_tokens


class StableDiffusionTryOnClipResamplerPipeline(StableDiffusionControlNetPipeline):

    def load_ip_adapter(self, model_ckpt: str, clip_embeddings_dim: int = 1280, num_tokens: int = 16, scale: float = 1.0):
        self.set_image_proj_model(clip_embeddings_dim, num_tokens)
        ip_adapter_modules = self.set_ip_adapter(num_tokens, scale)

        self.ip_adapter = IPAdapter(
            image_proj_model=self.image_proj_model,
            adapter_modules=ip_adapter_modules,
            ckpt_path=model_ckpt
        )

    def set_image_proj_model(self, clip_embeddings_dim: int = 1280, num_tokens: int = 16):
        cross_attention_dim = self.unet.config.cross_attention_dim
        image_proj_model = Resampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=16,
            num_queries=num_tokens,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

        image_proj_model.eval()
        self.image_proj_model = image_proj_model.to(self.unet.device).to(self.unet.dtype)

        self._clip_embeddings_dim = clip_embeddings_dim

    def set_ip_adapter(self, num_tokens: int = 16, scale: float = 1.0):
        # Check if UNet already has IP-Adapter processors (from validation)
        has_ip_adapter_processors = all(
            isinstance(proc, IPAttnProcessor) for name, proc in self.unet.attn_processors.items()
            if not name.endswith("attn1.processor")
        )

        if not has_ip_adapter_processors:
            # Create new IP-Adapter processors only if UNet doesn't have them
            unet = self.unet
            attn_procs = {}
            unet_sd = unet.state_dict()
            for name in self.unet.attn_processors.keys():
                # attn1 = self-attention (no cross_attention_dim), attn2 = cross-attention (has cross_attention_dim)
                # IP-Adapter only modifies cross-attention layers
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
                    attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
                else:
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name] = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=scale,
                        num_tokens=num_tokens
                    ).to(unet.device, dtype=unet.dtype)
                    attn_procs[name].load_state_dict(weights)
            unet.set_attn_processor(attn_procs)

        ip_adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        return ip_adapter_modules

    def set_ip_adapter_scale(self, scale: float):
        for attn_processor in self.unet.attn_processors.values():
            if hasattr(attn_processor, 'scale'):
                attn_processor.scale = scale

    def check_inputs(
        self,
        prompt,
        control_image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
        image_embeds=None,
        mask_image=None,
        image=None,
    ):
        # Call parent class check_inputs
        super().check_inputs(
            prompt,
            control_image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        # Validate cloth CLIP patch embeddings
        if image_embeds is None:
            raise ValueError(
                "`image_embeds` must be provided for StableDiffusionTryOnClipResamplerPipeline. "
                "Please encode the cloth image with a CLIP vision encoder and pass the penultimate-layer patch tokens "
                "(e.g. `image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]`)."
            )

        if not isinstance(image_embeds, torch.Tensor):
            raise ValueError(f"`image_embeds` must be a torch.Tensor, but got {type(image_embeds)}")

        if mask_image is not None:
            if self.unet.config.in_channels != 4:
                raise ValueError(f"Inpainting is only supported for UNet with 4 input channels, but got {self.unet.config.in_channels}. ")
            if image is None:
                raise ValueError("`image` must be provided when `mask_image` is provided for inpainting.")

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_image_embeds(
        self,
        image_embeds: torch.Tensor,
        device: torch.device,
        batch_size: int,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        """
        Expand cloth CLIP patch embeddings (B, N, D) to match the batch size and number of images per prompt.
        Handle CFG if enabled.
        """
        if image_embeds.shape[0] == 1:
            # Resampler input is 3D (B, N, D) -- repeat along batch only.
            repeat_sizes = [batch_size * num_images_per_prompt] + [1] * (image_embeds.dim() - 1)
            image_embeds = image_embeds.repeat(*repeat_sizes)
        elif image_embeds.shape[0] != batch_size:
            raise ValueError(
                f"Image embeds batch size ({image_embeds.shape[0]}) must match prompt batch size ({batch_size})"
            )
        else:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        image_embeds = image_embeds.to(device=device)

        if do_classifier_free_guidance:
            image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds], dim=0)

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline.prepare_mask_latents
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.StableDiffusionInpaintPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        control_image: PipelineImageInput = None,
        image_embeds: Optional[torch.Tensor] = None,
        mask_image: Optional[PipelineImageInput] = None,
        image: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        strength: float = 1.0,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        ip_adapter_scale: Optional[float] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation.
            image_embeds (`torch.Tensor`):
                Cloth image CLIP patch embeddings (penultimate `hidden_states[-2]`, shape `(B, N_patches, D)`) used
                as input to the Resampler IP-Adapter-Plus projection head.
            mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, ...):
                Optional inpainting mask. White pixels are repainted; black pixels are preserved.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, ..., *optional*):
                Source image for inpainting. Required when `mask_image` is provided.
            height/width (`int`, *optional*):
                Output image size; defaults to `self.unet.config.sample_size * self.vae_scale_factor`.
            num_inference_steps (`int`, *optional*, defaults to 50): Number of denoising steps.
            timesteps/sigmas (`List`, *optional*): Custom timesteps/sigmas for the scheduler.
            strength (`float`): Inpainting strength (1.0 = pure noise init).
            guidance_scale (`float`, *optional*, defaults to 7.5): Classifier-free guidance scale.
            negative_prompt (`str` or `List[str]`, *optional*): Negative prompt.
            num_images_per_prompt (`int`, *optional*, defaults to 1): Images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0): DDIM eta.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*): RNG.
            latents (`torch.Tensor`, *optional*): Pre-generated noise latents.
            prompt_embeds/negative_prompt_embeds (`torch.Tensor`, *optional*): Pre-computed text embeddings.
            output_type (`str`): `"pil"` or `"np.array"`.
            return_dict (`bool`, *optional*, defaults to `True`): Return pipeline output dataclass if True.
            cross_attention_kwargs (`dict`, *optional*): Passed to `AttentionProcessor`.
            controlnet_conditioning_scale (`float` or `List[float]`, defaults to 1.0): ControlNet scale.
            guess_mode (`bool`, defaults to `False`): ControlNet guess mode.
            control_guidance_start/end (`float` or `List[float]`): Step range where ControlNet applies.
            clip_skip (`int`, *optional*): Skip last N CLIP text layers.
            callback_on_step_end: Per-step callback.
            callback_on_step_end_tensor_inputs: Tensor names to pass to the callback.
            ip_adapter_scale (`float`, *optional*): Override the current IPAttnProcessor scale.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if ip_adapter_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale)

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        if not hasattr(self, 'mask_processor') or self.mask_processor is None:
            if not hasattr(self, 'vae_scale_factor'):
                self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=False, do_convert_grayscale=True
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            None,
            None,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
            image_embeds,
            mask_image,
            image,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        image_embeds_prepared = self.prepare_image_embeds(
            image_embeds,
            device,
            batch_size,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

        # Check if IP adapter is loaded
        if not hasattr(self, 'ip_adapter') or self.ip_adapter is None:
            raise ValueError(
                "IP-Adapter is not loaded. Please call `load_ip_adapter()` before using the pipeline."
            )

        prompt_embeds, ip_tokens = self.ip_adapter(prompt_embeds.to(self.unet.device), image_embeds_prepared.to(self.unet.device))
        prompt_embeds = prompt_embeds.to(device=device)
        ip_tokens = ip_tokens.to(device=device)

        # 4. Preprocess mask and image for inpainting if mask is provided
        if mask_image is not None:
            original_image = image
            init_image = self.image_processor.preprocess(
                image, height=height, width=width
            )
            init_image = init_image.to(dtype=torch.float32)

            mask = self.mask_processor.preprocess(
                mask_image, height=height, width=width
            )

            masked_image = init_image * (mask < 0.5)
            _, _, height, width = init_image.shape

        # 4.1 Prepare control image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            if mask_image is None:
                height, width = control_image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            control_images_list = []

            # Nested lists as ControlNet condition
            if isinstance(control_image[0], list):
                # Transpose the nested image list
                control_image = [list(t) for t in zip(*control_image)]

            for control_image_ in control_image:
                control_image_ = self.prepare_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images_list.append(control_image_)

            control_image = control_images_list
            if mask_image is None:
                height, width = control_image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps

        if mask_image is not None:
            # Copied from diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps=num_inference_steps, strength=strength, device=device
            )
            # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
            is_strength_max = strength == 1.0
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas
            )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        num_channels_unet = self.unet.config.in_channels

        if mask_image is not None:
            return_image_latents = num_channels_unet == 4
            latents_outputs = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                return_noise=True,
                return_image_latents=return_image_latents,
            )

            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs

            mask, masked_image_latents = self.prepare_mask_latents(
                mask,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                self.do_classifier_free_guidance,
            )
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )[0]

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = ip_tokens.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = ip_tokens

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if mask_image is not None and num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    control_image = callback_outputs.pop("image", control_image)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            empty_device_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
