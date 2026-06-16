#!/bin/bash

# Configuration
DATASETS=("imagenet-r" "cifar100_224" "cub200_224" "cars196_224")
GPUS=(5 1 2 4)
RANDOM_PROJECTION_DIMS=(6000)
VIT_TYPES=("vit-b-p16-mocov3")

# Ensure script is executable: chmod +x within_domain_random_experiments.sh

run_dataset() {
    local DATASET=$1
    local GPU=$2
    local RANDOM_PROJECTION_DIM=$3
    local VIT_TYPE=$4
    echo "============================================"
    echo "Starting dataset: $DATASET on GPU $GPU | Random_Projection_Dim: $RANDOM_PROJECTION_DIM | Vit_Type: $VIT_TYPE | Seeds: $SEEDS"
    echo "============================================"

    echo "[$(date)] Running $DATASET | GPU: $GPU | Random_Projection_Dim: $RANDOM_PROJECTION_DIM | Vit_Type: $VIT_TYPE"
    CUDA_VISIBLE_DEVICES=$GPU python main_random.py \
        --dataset "$DATASET" \
        --vit_type "$VIT_TYPE" \
        --lora_type "basic_lora" \
        --model_name "random_projector" \
        --smart_defaults \
        --seed_list 1990 \
        --random_projection_dim "$RANDOM_PROJECTION_DIM" \
        --lora_rank 4 \
        # --enable_weight_interpolation \


    echo "[$(date)] Completed: $DATASET (Random_Projection_Dim: $RANDOM_PROJECTION_DIM | Vit_Type: $VIT_TYPE)"
}

# 按顺序运行不同的vit_type和random_projection_dim值
for VIT_TYPE in "${VIT_TYPES[@]}"; do
    for RANDOM_PROJECTION_DIM in "${RANDOM_PROJECTION_DIMS[@]}"; do
        echo "################################################################"
        echo "Starting experiments with Vit_Type = $VIT_TYPE | Random_Projection_Dim = $RANDOM_PROJECTION_DIM"
        echo "################################################################"
        
        # 为当前vit_type和random_projection_dim值启动所有数据集的并行运行
        for i in "${!DATASETS[@]}"; do
            DATASET=${DATASETS[$i]}
            GPU=${GPUS[$i]}
            run_dataset "$DATASET" "$GPU" "$RANDOM_PROJECTION_DIM" "$VIT_TYPE" "$SEEDS" &
        done

        # 等待当前vit_type和random_projection_dim的所有作业完成
        wait
        
        echo "################################################################"
        echo "Completed all experiments with Vit_Type = $VIT_TYPE | Random_Projection_Dim = $RANDOM_PROJECTION_DIM"
        echo "################################################################"
        echo ""
    done
done

echo "All within-domain random projection experiments completed for all Vit_Type and Random_Projection_Dim values."