import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
idx = 2
model = ["stabilityai/sd-turbo", "stabilityai/sdxl-turbo","Manojb/stable-diffusion-2-1-base"][idx]
num_inference_steps = [4, 4, 20][idx]
pipeline = [StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline][idx]

pipe = pipeline.from_pretrained(
    model, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

my_prompts = [
    "full head portrait of a person"
] * 1

images = pipe(
    prompt=my_prompts, 
    num_inference_steps=num_inference_steps, 
    guidance_scale=0.0
).images

for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")