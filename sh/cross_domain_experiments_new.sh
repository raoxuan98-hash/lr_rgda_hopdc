#!/usr/bin/env bash
set -euo pipefail

echo "Starting Cross-Domain Experiments..."

# 创建总日志目录
MASTER_LOG_DIR="logs/cross_domain_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993)

# GPU分配 - 每个方法使用一个GPU，并行运行
GPUS=(0 1 2 4 5)

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
            --vit_type "vit-b-p16-clip" \
            --hopfield_temp "0.05" \
            --hopfield_topk "100" \
            --feature_combination_type "aux_only" \
            --auxiliary_data_size "2048" \
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

echo "=========================================="
echo "PHASE 1: Running LoRA and LoRA + KD Experiments"
echo "=========================================="

# 并行运行所有实验类型，每个在一个GPU上
PIDS=()

# 3. 运行基础LoRA跨域实验 (GPU 0)
{
    run_experiment_type "basic_lora" "basic_lora" "0" "--gamma_kd" "0.0" "--hopfield_temp" "0.05" "--hopfield_topk" "100" "--feature_combination_type" "combined" "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 4. 运行LoRA + 蒸馏跨域实验 (gamma_kd=1.0, iterations=1500) (GPU 1)
{
    run_experiment_type "lora_kd_1.0" "basic_lora" "1" \
        "--gamma_kd" "1.0" \
        "--update_teacher_each_task" \
        "--distillation_transform" "identity" \
        "--kd_type" "cos" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 4b. 运行LoRA + 蒸馏跨域实验 (gamma_kd=0.5, iterations=1500) (GPU 2)
{
    run_experiment_type "lora_kd_0.5" "basic_lora" "2" \
        "--gamma_kd" "0.5" \
        "--update_teacher_each_task" \
        "--distillation_transform" "identity" \
        "--kd_type" "cos" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 4c. 运行LoRA + 蒸馏跨域实验 (gamma_kd=1.0, iterations=2500) (GPU 4)
{
    run_experiment_type "lora_kd_1.0_iter2500" "basic_lora" "4" \
        "--gamma_kd" "1.0" \
        "--update_teacher_each_task" \
        "--distillation_transform" "identity" \
        "--kd_type" "cos" \
        "--iterations" "2500" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 等待第一阶段实验完成
echo "Waiting for Phase 1 experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "=========================================="
echo "PHASE 2: Running LoRA-NSP and LoRA-SGP Experiments"
echo "=========================================="

# 重置PID数组用于第二批实验
PIDS=()

# 1. 运行LoRA-NSP跨域实验 (nsp_weight=0.05) (GPU 4)
{
    run_experiment_type "nsp_lora_0.05" "nsp_lora" "4" \
        "--gamma_kd" "0.0" \
        "--nsp_weight" "0.05" \
        "--nsp_eps" "0.05" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 1b. 运行LoRA-NSP跨域实验 (nsp_weight=0.00) (GPU 5)
{
    run_experiment_type "nsp_lora_0.00" "nsp_lora" "5" \
        "--gamma_kd" "0.0" \
        "--nsp_weight" "0.00" \
        "--nsp_eps" "0.05" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 2. 运行LoRA-SGP跨域实验 (weight_temp=1.0, weight_p=1.0) (GPU 0)
{
    run_experiment_type "sgp_lora_t1.0_p1.0" "sgp_lora" "0" \
        "--gamma_kd" "0.0" \
        "--weight_temp" "1.0" \
        "--weight_p" "1.0" \
        "--weight_kind" "log1p" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 2b. 运行LoRA-SGP跨域实验 (weight_temp=2.0, weight_p=2.0) (GPU 1)
{
    run_experiment_type "sgp_lora_t2.0_p2.0" "sgp_lora" "1" \
        "--gamma_kd" "0.0" \
        "--weight_temp" "2.0" \
        "--weight_p" "2.0" \
        "--weight_kind" "log1p" \
        "--hopfield_temp" "0.05" \
        "--hopfield_topk" "100" \
        "--feature_combination_type" "combined" \
        "--auxiliary_data_size" "2048"
} &
PIDS+=($!)

# 等待第二批实验完成
echo "Waiting for Phase 2 experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "=========================================="
echo "All Cross-Domain experiments completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#SEEDS[@]} * 8))  # 8种实验变体
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Experiments per type: ${#SEEDS[@]} seeds (run sequentially)"
echo "Experiment variants: 8 (basic_lora, lora_kd_1.0, lora_kd_0.5, lora_kd_1.0_iter2500, nsp_lora_0.05, nsp_lora_0.00, sgp_lora_t1.0_p1.0, sgp_lora_t2.0_p2.0)"
echo "GPUs used: ${GPUS[@]} (5 GPUs available)"
echo "Execution strategy: Methods run in parallel, seeds run sequentially"