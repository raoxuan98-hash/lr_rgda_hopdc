#!/usr/bin/env bash
set -euo pipefail

echo "Starting Cross-Domain Experiments with Incremental Split..."

# 创建总日志目录
MASTER_LOG_DIR="logs/cross_domain_experiments_incremental_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# 跨域实验数据集
DATASET="cross_domain_elevater"
SEEDS=(1993 1996)

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
            --hopfield_temp "0.1" \
            --hopfield_topk "200" \
            --feature_combination_type "aux_only" \
            --auxiliary_data_size "4096" \
            --cross_domain \
            --num_shots "64" \
            --iterations "1500" \
            --enable_incremental_split \
            --num_incremental_splits "2" \
            --evaluate_final_only \
            "${additional_params[@]}" \
            --seed_list "$SEED" \
            2>&1 | tee "$LOG_DIR/seed${SEED}.log"
        
        echo "$experiment_name cross-domain experiment completed for seed $SEED on GPU $gpu_id"
    done
    
    echo "$experiment_name cross-domain experiments completed. Logs saved to $LOG_DIR"
}

echo "=========================================="
echo "PARALLEL EXECUTION: Running All Experiments Simultaneously"
echo "=========================================="

# 并行运行所有实验类型，充分利用5个GPU
PIDS=()

# 1. 运行基础LoRA跨域实验 (GPU 0)
{
    run_experiment_type "basic_lora" "basic_lora" "0" "--gamma_kd" "0.0" "--hopfield_temp" "0.1" "--hopfield_topk" "200" "--feature_combination_type" "aux_only" "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 2. 运行LoRA + 蒸馏跨域实验 (gamma_kd=1.0) (GPU 1)
{
    run_experiment_type "lora_kd_1.0" "basic_lora" "1" \
        "--gamma_kd" "1.0" \
        "--update_teacher_each_task" \
        "--distillation_transform" "identity" \
        "--kd_type" "cos" \
        "--hopfield_temp" "0.1" \
        "--hopfield_topk" "200" \
        "--feature_combination_type" "aux_only" \
        "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 3. 运行LoRA + 蒸馏跨域实验 (gamma_kd=0.5) (GPU 2)
{
    run_experiment_type "lora_kd_0.5" "basic_lora" "2" \
        "--gamma_kd" "0.5" \
        "--update_teacher_each_task" \
        "--distillation_transform" "identity" \
        "--kd_type" "cos" \
        "--hopfield_temp" "0.1" \
        "--hopfield_topk" "200" \
        "--feature_combination_type" "aux_only" \
        "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 4. 运行LoRA-NSP跨域实验 (nsp_weight=0.02, nsp_eps=0.05) (GPU 4)
{
    run_experiment_type "nsp_lora_0.05" "nsp_lora" "4" \
        "--gamma_kd" "0.0" \
        "--nsp_weight" "0.02" \
        "--nsp_eps" "0.05" \
        "--hopfield_temp" "0.1" \
        "--hopfield_topk" "200" \
        "--feature_combination_type" "aux_only" \
        "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 5. 运行LoRA-NSP跨域实验 (nsp_weight=0.02, nsp_eps=0.10) (GPU 5)
{
    run_experiment_type "nsp_lora_0.10" "nsp_lora" "5" \
        "--gamma_kd" "0.0" \
        "--nsp_weight" "0.02" \
        "--nsp_eps" "0.10" \
        "--hopfield_temp" "0.1" \
        "--hopfield_topk" "200" \
        "--feature_combination_type" "aux_only" \
        "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 等待所有实验完成
echo "Waiting for all experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "=========================================="
echo "PHASE 2: Running SGP LoRA Experiments"
echo "=========================================="

# 第二阶段：运行SGP LoRA实验，分布在GPU 0,1,2上
PIDS=()

# 6. 运行LoRA-SGP跨域实验 (weight_temp=1.0, weight_p=1.0) (GPU 0)
{
    run_experiment_type "sgp_lora_t1.0_p1.0" "sgp_lora" "0" \
        "--gamma_kd" "0.0" \
        "--weight_temp" "1.0" \
        "--weight_p" "1.0" \
        "--weight_kind" "log1p" \
        "--hopfield_temp" "0.1" \
        "--hopfield_topk" "200" \
        "--feature_combination_type" "aux_only" \
        "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 7. 运行LoRA-SGP跨域实验 (weight_temp=2.0, weight_p=2.0) (GPU 1)
{
    run_experiment_type "sgp_lora_t2.0_p2.0" "sgp_lora" "1" \
        "--gamma_kd" "0.0" \
        "--weight_temp" "2.0" \
        "--weight_p" "2.0" \
        "--weight_kind" "log1p" \
        "--hopfield_temp" "0.1" \
        "--hopfield_topk" "200" \
        "--feature_combination_type" "aux_only" \
        "--auxiliary_data_size" "4096"
} &
PIDS+=($!)

# 等待第二阶段实验完成
echo "Waiting for Phase 2 experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "=========================================="
echo "All Cross-Domain experiments with incremental split completed!"
echo "Logs saved to: $MASTER_LOG_DIR"
echo "=========================================="

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#SEEDS[@]} * 7))  # 7种实验变体 × 2个种子
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Experiments per type: ${#SEEDS[@]} seeds (run sequentially)"
echo "Seeds used: ${SEEDS[@]}"
echo "Experiment variants: 7 (basic_lora, lora_kd_1.0, lora_kd_0.5, nsp_lora_0.05, nsp_lora_0.10, sgp_lora_t1.0_p1.0, sgp_lora_t2.0_p2.0)"
echo "GPUs used: ${GPUS[@]} (5 GPUs available)"
echo "Execution strategy: Phased parallel execution"
echo "  - Phase 1: basic_lora, lora_kd_1.0, lora_kd_0.5, nsp_lora_0.05, nsp_lora_0.10 (5 GPUs)"
echo "  - Phase 2: sgp_lora_t1.0_p1.0, sgp_lora_t2.0_p2.0 (3 GPUs)"
echo "Configuration: vit-b-p16-clip, enable_incremental_split=True, num_incremental_splits=2, evaluate_final_only=True"
echo "GPU allocation strategy: Optimized parallelization with efficient GPU utilization"