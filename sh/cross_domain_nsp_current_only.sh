#!/usr/bin/env bash
set -euo pipefail

echo "Starting LoRA-NSP Cross-Domain Experiments with current_only feature combination..."

# 创建总日志目录
MASTER_LOG_DIR="logs/cross_domain_nsp_current_only_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)

# GPU分配 - 使用GPU 2
GPU_ID=2

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    local gpu_id="$3"
    shift 3
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Cross-Domain Experiments on GPU $gpu_id"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 串行运行所有种子
    for SEED in "${SEEDS[@]}"; do
        echo "Starting $experiment_name cross-domain experiment for seed $SEED on GPU $gpu_id"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$lora_type" \
            --vit_type "vit-b-p16" \
            --cross_domain \
            --num_shots "64" \
            --iterations "1500" \
            "${additional_params[@]}" \
            --seed_list "$SEED" \
            2>&1 | tee "$LOG_DIR/seed${SEED}.log"
        
        echo "$experiment_name cross-domain experiment completed for seed $SEED on GPU $gpu_id"
    done
    
    echo "$experiment_name cross-domain experiments completed. Logs saved to $LOG_DIR"
}

# 运行LoRA-NSP跨域实验 (nsp_weight=0.05) 使用current_only特征组合
run_experiment_type "nsp_lora_0.05_current_only" "basic_lora" "$GPU_ID" \
    --gamma_kd "0.0" \
    --nsp_weight "0.05" \
    --nsp_eps "0.05" \
    --feature_combination_type "current_only"

echo "=========================================="
echo "LoRA-NSP Cross-Domain experiments with current_only feature combination completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=${#SEEDS[@]}
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Feature combination type: current_only"
echo "GPU used: $GPU_ID"