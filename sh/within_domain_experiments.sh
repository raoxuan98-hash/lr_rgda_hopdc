#!/bin/bash

# Configuration
DATASETS=("imagenet-r" "cifar100_224" "cub200_224" "cars196_224")
GPUS=(0 1 2 4)
GAMMA_KDS=(0.0 1.0)
VIT_TYPES=("vit-b-p16-mocov3")
LORA_TYPES=("full" "full_nsp")

# Ensure script is executable: chmod +x run_within_domain.sh

run_dataset() {
    local DATASET=$1
    local GPU=$2
    local GAMMA_KD=$3
    local VIT_TYPE=$4
    local LORA_TYPE=$5
    echo "============================================"
    echo "Starting dataset: $DATASET on GPU $GPU | Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE | Seeds: $SEEDS"
    echo "============================================"
    echo "[$(date)] Running $DATASET | GPU: $GPU | Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE"
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset "$DATASET" \
        --vit_type "$VIT_TYPE" \
        --lora_type "$LORA_TYPE" \
        --smart_defaults \
        --seed_list 1990 \
        --gamma_kd "$GAMMA_KD" \
        --classifier_types lr_rgda
        # --enable_weight_interpolation \

    echo "[$(date)] Completed: $DATASET (Gamma_KD: $GAMMA_KD | Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE)"
}

# 按顺序运行不同的lora_type、vit_type和gamma_kd值
for LORA_TYPE in "${LORA_TYPES[@]}"; do
    for VIT_TYPE in "${VIT_TYPES[@]}"; do
        for GAMMA_KD in "${GAMMA_KDS[@]}"; do
            echo "################################################################"
            echo "Starting experiments with Lora_Type = $LORA_TYPE | Vit_Type = $VIT_TYPE | Gamma_KD = $GAMMA_KD"
            echo "################################################################"
            
            # 为当前lora_type、vit_type和gamma_kd值启动所有数据集的并行运行
            for i in "${!DATASETS[@]}"; do
                DATASET=${DATASETS[$i]}
                GPU=${GPUS[$i]}
                run_dataset "$DATASET" "$GPU" "$GAMMA_KD" "$VIT_TYPE" "$LORA_TYPE" &
            done

            # 等待当前lora_type、vit_type和gamma_kd的所有作业完成
            wait
            
            echo "################################################################"
            echo "Completed all experiments with Lora_Type = $LORA_TYPE | Vit_Type = $VIT_TYPE | Gamma_KD = $GAMMA_KD"
            echo "################################################################"
            echo ""
        done
    done
done

echo "All within-domain evaluations completed for all Lora_Type, Vit_Type and Gamma_KD values."
