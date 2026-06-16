#!/usr/bin/env bash
set -euo pipefail

echo "Starting HopDC Parameter Experiments..."

# 创建总日志目录
MASTER_LOG_DIR="logs/hopfield_parameter_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)

# GPU分配 - 使用GPU 0,1,2,4,5
GPU_IDS=(0 1 2 4 5)

# 实验配置数组
# 实验1: 固定 hopfield_temp=0.1，变动 hopfield_topk 为 50, 200
# 实验2: 固定 hopfield_topk=500，变动 hopfield_temp 为 0.05, 0.01
declare -a EXPERIMENT_CONFIGS=(
    "0.1:50:sgp_lora_temp0.1_topk50"
    "0.1:200:sgp_lora_temp0.1_topk200"
    "0.05:500:sgp_lora_temp0.05_topk500"
    "0.01:500:sgp_lora_temp0.01_topk500"
)

# 运行单个实验的函数
run_experiment() {
    local experiment_config="$1"
    local gpu_id="$2"
    
    # 解析实验配置
    IFS=':' read -r temp_value topk_value experiment_name <<< "$experiment_config"
    
    echo "=========================================="
    echo "Running $experiment_name on GPU $gpu_id"
    echo "Temperature: $temp_value, TopK: $topk_value"
    echo "=========================================="
    
    # 创建实验特定的日志目录
    LOG_DIR="$MASTER_LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 串行运行所有种子
    for SEED in "${SEEDS[@]}"; do
        echo "Starting $experiment_name experiment for seed $SEED on GPU $gpu_id"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python -u main.py \
            --dataset "$DATASET" \
            --smart_defaults \
            --lora_type "basic_lora" \
            --vit_type "vit-b-p16-dino" \
            --cross_domain \
            --num_shots "64" \
            --iterations "1500" \
            --hopfield_temp "$temp_value" \
            --hopfield_topk "$topk_value" \
            --gamma_kd "0.0" \
            --weight_temp "2.0" \
            --weight_kind "log1p" \
            --weight_p "1.0" \
            --compensator_types "SeqFT" "SeqFT + linear" "SeqFT + HopDC" \
            --seed_list "$SEED" \
            2>&1 | tee "$LOG_DIR/seed${SEED}.log"
        
        echo "$experiment_name experiment completed for seed $SEED on GPU $gpu_id"
    done
    
    echo "$experiment_name experiments completed. Logs saved to $LOG_DIR"
}

# 启动并行实验
echo "Starting parallel experiments on GPUs: ${GPU_IDS[*]}"
echo "Total experiments: ${#EXPERIMENT_CONFIGS[@]}"

# 使用后台进程并行运行实验
for i in "${!EXPERIMENT_CONFIGS[@]}"; do
    config="${EXPERIMENT_CONFIGS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    
    # 在后台启动实验
    run_experiment "$config" "$gpu_id" &
    
    echo "Started experiment $i ($config) on GPU $gpu_id in background"
done

# 等待所有后台实验完成
echo "Waiting for all experiments to complete..."
wait

echo "=========================================="
echo "All HopDC Parameter experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#EXPERIMENT_CONFIGS[@]} * ${#SEEDS[@]}))
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Experiment Groups:"
echo "  Group 1 - Fixed hopfield_temp=0.1, varying hopfield_topk: 50, 200"
echo "  Group 2 - Fixed hopfield_topk=500, varying hopfield_temp: 0.05, 0.01"
echo "GPUs used: ${GPU_IDS[*]}"