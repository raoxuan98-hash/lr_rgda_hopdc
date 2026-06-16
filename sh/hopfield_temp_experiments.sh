#!/usr/bin/env bash
set -euo pipefail

echo "Starting HopDC Temperature Experiments (fixed topk=1000)..."

# 创建总日志目录
MASTER_LOG_DIR="logs/hopfield_temp_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)
FIXED_TOPK=1000
TEMP_VALUES=(0.1 0.5 1.0)

# GPU分配 - 使用GPU 1
GPU_ID=0

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    local gpu_id="$3"
    local temp_value="$4"
    shift 4
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Cross-Domain Experiments on GPU $gpu_id"
    echo "Temperature: $temp_value, TopK: $FIXED_TOPK"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_temp${temp_value}_$(date +%Y%m%d_%H%M%S)"
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
            --hopfield_temp "$temp_value" \
            --hopfield_topk "$FIXED_TOPK" \
            "${additional_params[@]}" \
            --seed_list "$SEED" \
            2>&1 | tee "$LOG_DIR/seed${SEED}.log"
        
        echo "$experiment_name cross-domain experiment completed for seed $SEED on GPU $gpu_id"
    done
    
    echo "$experiment_name cross-domain experiments completed. Logs saved to $LOG_DIR"
}

# 运行不同温度的实验
for TEMP in "${TEMP_VALUES[@]}"; do
    run_experiment_type "sgp_lora_temp${TEMP}" "basic_lora" "$GPU_ID" "$TEMP" \
        --gamma_kd "0.0" \
        --weight_temp "2.0" \
        --weight_kind "log1p" \
        --weight_p "1.0" \
        --compensator_types "SeqFT" "SeqFT + linear" "SeqFT + HopDC"
done

echo "=========================================="
echo "HopDC Temperature experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#TEMP_VALUES[@]} * ${#SEEDS[@]}))
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Temperature values tested: ${TEMP_VALUES[*]}"
echo "Fixed TopK: $FIXED_TOPK"
echo "GPU used: $GPU_ID"