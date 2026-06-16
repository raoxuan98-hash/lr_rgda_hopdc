#!/bin/bash

# Configuration
DATASET="cub200_224"  # 1. 固定为同一个数据集
SEED=1990  # 3. 固定随机种子为1990
ITERATIONS=(400 600)  # 2. iterations配置
GPUS=(0 1)  # 4. 在GPUs 0,1,2,4,5上运行

# Ensure script is executable: chmod +x run_within_domain.sh

run_experiment() {
    local ITERATION=$1
    local GPU=$2

    echo "============================================"
    echo "Starting experiment: $DATASET | Iteration: $ITERATION on GPU $GPU"
    echo "============================================"

    echo "[$(date)] Running $DATASET | Seed: $SEED | GPU: $GPU | Iterations: $ITERATION"

    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset "$DATASET" \
        --vit_type "vit-b-p16" \
        --lora_type "basic_lora" \
        --seed_list "$SEED" \
        --gamma_kd 0.0 \
        --iterations "$ITERATION" \
        --kd_type "feat"

    echo "[$(date)] Completed: $DATASET seed $SEED iteration $ITERATION"
}

# 检查参数是否匹配
if [ ${#ITERATIONS[@]} -ne ${#GPUS[@]} ]; then
    echo "Error: Number of iterations (${#ITERATIONS[@]}) must match number of GPUs (${#GPUS[@]})"
    echo "Iterations: ${ITERATIONS[*]}"
    echo "GPUs: ${GPUS[*]}"
    exit 1
fi

# Launch each iteration on its assigned GPU in parallel
for i in "${!ITERATIONS[@]}"; do
    ITERATION=${ITERATIONS[$i]}
    GPU=${GPUS[$i]}
    run_experiment "$ITERATION" "$GPU" &
done

# Wait for all background jobs to finish
wait

echo "All experiments completed."