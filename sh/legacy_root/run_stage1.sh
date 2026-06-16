#!/bin/bash

# ==============================================================================
# run_stage1.sh - 后台稳健并行版本
# ==============================================================================
# 建议运行方式: nohup bash run_stage1.sh > logs/main_run.log 2>&1 &
# ==============================================================================

export CUDA_VISIBLE_DEVICES="0,1,2,4,5"
GPUS=(0 1 2 4 5)
MAX_JOBS=${#GPUS[@]}

DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
METHODS=("full:0.0" "full:1.0" "basic_lora:0.0")

mkdir -p logs/stage1_parallel

current_jobs=0
gpu_idx=0

echo "📅 启动时间: $(date)"
echo "🚀 开始并行训练任务池 (GPU: ${GPUS[*]})"

for ds in "${DATASETS[@]}"; do
    for method_info in "${METHODS[@]}"; do
        lora_type=$(echo $method_info | cut -d':' -f1)
        gamma_kd=$(echo $method_info | cut -d':' -f2)
        
        # 超参数对齐逻辑 (同前)
        iters=1000
        if [[ "$ds" == "cub200_224" ]]; then
            [[ "$gamma_kd" != "0.0" ]] && iters=1000 || iters=500
        else
            [[ "$gamma_kd" != "0.0" ]] && iters=1500 || iters=1000
        fi

        gpu=${GPUS[$gpu_idx]}
        method_name="SeqFT"
        [[ "$gamma_kd" != "0.0" ]] && method_name="SeqKD"
        [[ "$lora_type" == "basic_lora" ]] && method_name="LoRA"

        log_file="logs/stage1_parallel/${ds}_${method_name}_$(date +%m%d_%H%M).log"

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


