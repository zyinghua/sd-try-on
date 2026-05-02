# Codebase for SD-Try-On

## Environment setup:
Create a conda environment:

``` bash
conda create -n sdtryon python=3.10 -y
conda activate sdtryon
```

or

``` bash
conda create -p /path/to/conda-env python=3.10 -y
conda activate /path/to/conda-env
```

Then, cd to the repo and install packages:

``` bash
cd sd-try-on/
pip install -r requirements.txt
```

## Download Dataset
https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset/data?select=train

## Training
You will need to set up training environment config first, please run `accelerate config`, and follow the below setup as default (change as needed based on your specific environment).
```bash
- This machine
- multi-GPU
- How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
- Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no
- Do you wish to optimize your script with torch dynamo?[yes/NO]: no
- Do you want to use FullyShardedDataParallel? [yes/NO]: no
- Do you want to use Megatron-LM ? [yes/NO]: no
- How many GPU(s) should be used for distributed training? [1]: <input your gpu num setting>
- Please select a choice using the arrow or number keys, and selecting with enter: fp16
```


Please use `train_clip_resampler.py` to train our version of try-on model with the cloth injected via IP-Adapter only. An exemplar usage is at `run_scripts/train_sdtryon_clip_resampler.sh`.
Please use `train_control.py` to train our version of try-on model with the cloth injected into the ControlNet simultaneously (with pose injected channel-wise). An exemplar usage is at `run_scripts/train_sdtryon_control.sh`.

## Inference
Please head to `inference/run_sdtryon_inference.py`(IP-Adapter only version)  or `inference/run_sdtryon_control_inference.py`, and adjust the paths and hyperparameters defined at the top accordingly based on the comments. To support the generation of control conditions, we offer a script `dataset/extract_cloth_mask_from_image.py` for extracting the upper-body cloth mask from a given image, and another script `dataset/extract_densepose_from_image.py` for extracting the corresponding dense pose image.

## Compute
Our model is trained and experimented based on two Nvidia RTX 4090 GPUs.
