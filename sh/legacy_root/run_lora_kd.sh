#!/bin/bash

# ==============================================================================
# run_lora_kd.sh - LoRA + 知识蒸馏方案
# ==============================================================================
# 建议运行方式: nohup bash run_lora_kd.sh > logs/lora_kd_run.log 2>&1 &
# ==============================================================================

export CUDA_VISIBLE_DEVICES="0,1,2,4,5"
GPUS=(0 1 2 4 5)
MAX_JOBS=${#GPUS[@]}

DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
# LoRA + 知识蒸馏
METHODS=("basic_lora:1.0")

mkdir -p logs/lora_kd_parallel

current_jobs=0
gpu_idx=0

echo "📅 启动时间: $(date)"
echo "🚀 开始并行训练任务池 - LoRA + 知识蒸馏 (GPU: ${GPUS[*]})"

for ds in "${DATASETS[@]}"; do
    for method_info in "${METHODS[@]}"; do
        lora_type=$(echo $method_info | cut -d':' -f1)
        gamma_kd=$(echo $method_info | cut -d':' -f2)
        
        # 超参数对齐逻辑
        iters=1000
        if [[ "$ds" == "cub200_224" ]]; then
            iters=1000
        else
            iters=1500
        fi

        gpu=${GPUS[$gpu_idx]}
        method_name="LoRA_KD"

        log_file="logs/lora_kd_parallel/${ds}_${method_name}_$(date +%m%d_%H%M).log"

        echo "  [$(date +%H:%M:%S)] 启动: $ds | $method_name -> GPU $gpu"

        # 核心启动指令
        CUDA_VISIBLE_DEVICES=$gpu python main.py \
            --dataset "$ds" \
            --lora_type "$lora_type" \
            --gamma_kd "$gamma_kd" \
            --iterations "$iters" \
            --smart_defaults > "$log_file" 2>&1 &
        
        ((current_jobs++))
        ((gpu_idx=(gpu_idx+1)%MAX_JOBS))

        if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
            echo "  [Wait] 当前批次 5 个任务正在运行，等待中..."
            wait
            current_jobs=0
            gpu_idx=0
        fi
    done
done

wait
echo "✅ 所有任务于 $(date) 完成！"
