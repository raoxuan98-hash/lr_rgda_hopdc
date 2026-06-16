#!/bin/bash

# GPU 1: nsp_weight=0.05, nsp_eps=0.10
export CUDA_VISIBLE_DEVICES=1

python main.py \
    --cross_domain \
    --vit_type vit-b-p16-clip \
    --lora_type nsp_lora \
    --nsp_weight 0.05 \
    --nsp_eps 0.10 \
    --num_shots 64 \
    --lrate 0.0001 \
    --batch_size 16 \
    --iterations 2000 \
    --seed_list 1993 \
    --optimizer adamw \
    --weight_decay 3e-5 \
    --feature_combination_type combined \
    --auxiliary_data_size 2048 \