#!/usr/bin/env bash
set -euo pipefail

echo "Starting HopDC TopK Experiments (fixed temp=0.1)..."

# 创建总日志目录
MASTER_LOG_DIR="logs/hopfield_topk_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)
FIXED_TEMP=0.1
TOPK_VALUES=(500 1000 2000)

# GPU分配 - 使用GPU 1
GPU_ID=1

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    local gpu_id="$3"
    local topk_value="$4"
    shift 4
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Cross-Domain Experiments on GPU $gpu_id"
    echo "Temperature: $FIXED_TEMP, TopK: $topk_value"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_topk${topk_value}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 串行运行所有种子
    for SEED in "${SEEDS[@]}"; do
        echo "Starting $experiment_name cross-domain experiment for seed $SEED on GPU $gpu_id"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "$lora_type" \
            --vit_type "vit-b-p16-dino" \
            --cross_domain \
            --num_shots "64" \
            --iterations "1500" \
            --hopfield_temp "$FIXED_TEMP" \
            --hopfield_topk "$topk_value" \
            "${additional_params[@]}" \
            --seed_list "$SEED" \
            2>&1 | tee "$LOG_DIR/seed${SEED}.log"
        
        echo "$experiment_name cross-domain experiment completed for seed $SEED on GPU $gpu_id"
    done
    
    echo "$experiment_name cross-domain experiments completed. Logs saved to $LOG_DIR"
}

# 运行不同topk的实验
for TOPK in "${TOPK_VALUES[@]}"; do
    run_experiment_type "sgp_lora_topk${TOPK}" "basic_lora" "$GPU_ID" "$TOPK" \
        --gamma_kd "0.0" \
        --weight_temp "2.0" \
        --weight_kind "log1p" \
        --weight_p "1.0" \
        --compensator_types "SeqFT" "SeqFT + linear" "SeqFT + HopDC"
done

echo "=========================================="
echo "HopDC TopK experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#TOPK_VALUES[@]} * ${#SEEDS[@]}))
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "TopK values tested: ${TOPK_VALUES[*]}"
echo "Fixed Temperature: $FIXED_TEMP"
echo "GPU used: $GPU_ID"