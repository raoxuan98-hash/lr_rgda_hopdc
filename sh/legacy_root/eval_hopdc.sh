#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,4,5"
GPUS=(0 1 2 4 5)
MAX_JOBS=${#GPUS[@]}

DATASETS=("cross_domain_elevater")
METHODS=("full:0.0" "full:1.0" "basic_lora:0.0" "basic_lora:1.0")
HOPDC_TYPES=("current_only" "mnist")

mkdir -p logs/eval_hopdc_parallel

current_jobs=0
gpu_idx=0

echo "📅 启动时间: $(date)"
echo "🚀 开始并行评估 HopDC 任务池 (GPU: ${GPUS[*]})"

for ds in "${DATASETS[@]}"; do
    for method_info in "${METHODS[@]}"; do
        lora_type=$(echo $method_info | cut -d':' -f1)
        gamma_kd=$(echo $method_info | cut -d':' -f2)
        
        method_name="SeqFT"
        [[ "$gamma_kd" != "0.0" ]] && method_name="SeqKD"
        if [[ "$lora_type" == "basic_lora" ]]; then
            if [[ "$gamma_kd" != "0.0" ]]; then
                method_name="LoRA_KD"
            else
                method_name="LoRA"
            fi
        fi

        # The log path where the checkpoints were saved
        # Note: main.py generates log_path like:
        # RGDA_WD_2025-12-19-within/DS_cross_domain_elevater/VB16/I...
        # We need to find the exact log_path or just let main.py find it using the same parameters.
        
        for hopdc_type in "${HOPDC_TYPES[@]}"; do
            gpu=${GPUS[$gpu_idx]}
            log_file="logs/eval_hopdc_parallel/${ds}_${method_name}_${hopdc_type}_$(date +%m%d_%H%M).log"
            
            echo "  [$(date +%H:%M:%S)] 启动评估: $ds | $method_name | $hopdc_type -> GPU $gpu"

            if [[ "$hopdc_type" == "current_only" ]]; then
                CUDA_VISIBLE_DEVICES=$gpu python main.py \
                    --dataset "$ds" \
                    --lora_type "$lora_type" \
                    --gamma_kd "$gamma_kd" \
                    --cross_domain \
                    --smart_defaults \
                    --eval_only \
                    --feature_combination_type "current_only" \
                    > "$log_file" 2>&1 &
            elif [[ "$hopdc_type" == "mnist" ]]; then
                CUDA_VISIBLE_DEVICES=$gpu python main.py \
                    --dataset "$ds" \
                    --lora_type "$lora_type" \
                    --gamma_kd "$gamma_kd" \
                    --cross_domain \
                    --smart_defaults \
                    --eval_only \
                    --feature_combination_type "aux_only" \
                    --aux_dataset "mnist" \
                    > "$log_file" 2>&1 &
            fi
            
            ((current_jobs++))
            ((gpu_idx=(gpu_idx+1)%MAX_JOBS))

            if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
                echo "  [Wait] 当前批次 $MAX_JOBS 个评估任务正在运行，等待中..."
                wait
                current_jobs=0
                gpu_idx=0
            fi
        done
    done
done

wait
echo "✅ 所有评估任务于 $(date) 完成！"
