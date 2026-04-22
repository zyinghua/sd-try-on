#!/bin/bash

# source /etc/network_turbo
# export HF_HUB_DISABLE_XET=1

accelerate launch train_control.py \
    --pretrained_model_name_or_path "Manojb/stable-diffusion-2-1-base" \
    --image_encoder_path "patrickjohncyh/fashion-clip" \
    --conditioning_channels 3 \
    --train_data_dir "/root/autodl-tmp/data/sdtryon/train" \
    --output_dir "/root/autodl-tmp/sdtryon-model" \
    --num_train_epochs 1 \
    --max_train_steps 50000 \
    --resolution 512 384 \
    --learning_rate 1e-5 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 20 \
    --validation_steps 500 \
    --validation_prompt "A female wearing a white T-shirt with a red LEVI's logo." "A female wearing a black T-shirt with white logo and text." \
    --validation_image "/root/autodl-tmp/data/sdtryon/train/image-densepose/00000_00.jpg" "/root/autodl-tmp/data/sdtryon/train/image-densepose/00019_00.jpg" \
    --validation_cloth_image "/root/autodl-tmp/data/sdtryon/train/cloth/00000_00.jpg" "/root/autodl-tmp/data/sdtryon/train/cloth/00019_00.jpg" \
    --ip_adapter_image_drop_rate 0.05 \
    #--pretrained_ip_adapter_path "/path/to/pretrained_ip_adapter.bin" \
