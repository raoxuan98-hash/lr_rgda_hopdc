#!/usr/bin/env bash
set -euo pipefail

echo "Starting HopDC Auxiliary Data Size Experiments (fixed temp=0.05, topk=100)..."

# 创建总日志目录
MASTER_LOG_DIR="logs/hopfield_auxiliary_data_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)
FIXED_TEMP=0.05
FIXED_TOPK=100
AUXILIARY_DATA_SIZES=(500 1000 2000 4000)

# 可用GPU列表
GPUS=(0 1 2 4 5)

# 运行单个实验类型的函数
run_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    local gpu_id="$3"
    local auxiliary_data_size="$4"
    shift 4
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Running $experiment_name Cross-Domain Experiments on GPU $gpu_id"
    echo "Temperature: $FIXED_TEMP, TopK: $FIXED_TOPK, Auxiliary Data Size: $auxiliary_data_size"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_aux${auxiliary_data_size}_$(date +%Y%m%d_%H%M%S)"
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
            --hopfield_topk "$FIXED_TOPK" \
            --auxiliary_data_size "$auxiliary_data_size" \
            --feature_combination_type "aux_only" \
            "${additional_params[@]}" \
            --seed_list "$SEED" \
            2>&1 | tee "$LOG_DIR/seed${SEED}.log"
        
        echo "$experiment_name cross-domain experiment completed for seed $SEED on GPU $gpu_id"
    done
    
    echo "$experiment_name cross-domain experiments completed. Logs saved to $LOG_DIR"
}

# 并行运行不同辅助数据大小的实验
pids=()
gpu_index=0

for AUX_SIZE in "${AUXILIARY_DATA_SIZES[@]}"; do
    # 循环使用GPU
    GPU_ID=${GPUS[$gpu_index]}
    
    # 在后台运行实验
    run_experiment_type "sgp_lora_aux${AUX_SIZE}" "basic_lora" "$GPU_ID" "$AUX_SIZE" \
        --gamma_kd "0.0" \
        --weight_temp "2.0" \
        --weight_kind "log1p" \
        --weight_p "1.0" \
        --compensator_types "SeqFT" "SeqFT + linear" "SeqFT + HopDC" &
    
    # 保存进程ID
    pids+=($!)
    
    # 更新GPU索引，循环使用
    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))
done

# 等待所有后台进程完成
echo "等待所有实验完成..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "=========================================="
echo "HopDC Auxiliary Data Size experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#AUXILIARY_DATA_SIZES[@]} * ${#SEEDS[@]}))
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Auxiliary data sizes tested: ${AUXILIARY_DATA_SIZES[*]}"
echo "Fixed Temperature: $FIXED_TEMP"
echo "Fixed TopK: $FIXED_TOPK"
echo "Feature Combination Type: aux_only"
echo "GPUs used: ${GPUS[*]}"