#!/bin/bash

# GPU 2: nsp_weight=0.00, nsp_eps=0.05
export CUDA_VISIBLE_DEVICES=5

python main.py \
    --cross_domain \
    --vit_type vit-b-p16-clip \
    --lora_type basic_lora \
    --num_shots 64 \
    --lrate 0.0001 \
    --batch_size 16 \
    --iterations 1500 \
    --seed_list 1993 \
    --optimizer adamw \
    --weight_decay 3e-5 \
    --feature_combination_type aux_only \
    --auxiliary_data_size 2048 \
    --enable_incremental_split \
    --num_incremental_splits 3