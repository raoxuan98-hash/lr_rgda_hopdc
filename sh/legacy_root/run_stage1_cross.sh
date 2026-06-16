#!/bin/bash

# ==============================================================================
# run_stage1_cross.sh - 后台稳健并行版本
# ==============================================================================
export CUDA_VISIBLE_DEVICES="0,1,2,4,5"
GPUS=(0 1 2 4 5)
MAX_JOBS=${#GPUS[@]}

DATASETS=("cross_domain_elevater")
METHODS=("full:0.0" "full:1.0" "basic_lora:0.0" "basic_lora:1.0")

mkdir -p logs/stage1_cross_parallel

current_jobs=0
gpu_idx=0

echo "📅 启动时间: $(date)"
echo "🚀 开始并行训练任务池 (GPU: ${GPUS[*]})"

for ds in "${DATASETS[@]}"; do
    for method_info in "${METHODS[@]}"; do
        lora_type=$(echo $method_info | cut -d':' -f1)
        gamma_kd=$(echo $method_info | cut -d':' -f2)
        
        # 对于跨域实验，迭代次数统一设置，或根据有无KD调整
        iters=1500
        [[ "$gamma_kd" == "0.0" ]] && iters=1000

        gpu=${GPUS[$gpu_idx]}
        method_name="SeqFT"
        [[ "$gamma_kd" != "0.0" ]] && method_name="SeqKD"
        if [[ "$lora_type" == "basic_lora" ]]; then
            if [[ "$gamma_kd" != "0.0" ]]; then
                method_name="LoRA_KD"
            else
                method_name="LoRA"
            fi
        fi

        log_file="logs/stage1_cross_parallel/${ds}_${method_name}_$(date +%m%d_%H%M).log"

        echo "  [$(date +%H:%M:%S)] 启动: $ds | $method_name -> GPU $gpu"

        # 核心启动指令
        CUDA_VISIBLE_DEVICES=$gpu python main.py \
            --dataset "$ds" \
            --lora_type "$lora_type" \
            --gamma_kd "$gamma_kd" \
            --iterations "$iters" \
            --cross_domain \
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
