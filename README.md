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


Please use `train.py` to train our face swap model. An exemplar usage is at `/run_scripts/train_faceswap.sh`.

We also provide two additional training pipelines for ControlNet-only and ControlNet + IP-Adapter training that we used throughout our experiments, corresponding files are in the `legacy` directory.


## Inference

## Compute
Our model is trained and experimented based on two Nvidia RTX 4090 GPUs.
