#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# extract_seed_1993_parallel.sh - Parallel Feature Extraction
# ==============================================================================
# Uses GPUs 0, 1, 2, 4, 5 to speed up feature extraction for seed 1993 models.
# ==============================================================================

# Available GPUs (Skip GPU 3)
GPUS=(0 1 2 4 5)
MAX_JOBS=${#GPUS[@]}

# Find all seed_1993 directories in RGDA_WD_2025-12-19-within
directories=$(find RGDA_WD_2025-12-19-within -name "seed_1993" -type d)

mkdir -p logs/extract_parallel

current_jobs=0
gpu_idx=0

echo "🚀 Starting parallel feature extraction for seed 1993 (Max concurrent: $MAX_JOBS, GPUs: ${GPUS[*]})"

for log_path in $directories; do
    # Skip test runs (IT100)
    if [[ "$log_path" == *"IT100/"* ]]; then
        echo "  [Skip] Test run: $log_path"
        continue
    fi

    # Extract dataset name
    if [[ "$log_path" == *"DS_cifar100_224"* ]]; then
        dataset="cifar100_224"
    elif [[ "$log_path" == *"DS_imagenet-r"* ]]; then
        dataset="imagenet-r"
    elif [[ "$log_path" == *"DS_cub200_224"* ]]; then
        dataset="cub200_224"
    elif [[ "$log_path" == *"DS_cars196_224"* ]]; then
        dataset="cars196_224"
    else
        echo "  [Skip] Unknown dataset for $log_path"
        continue
    fi

    # Extract lora_type
    if [[ "$log_path" == *"r4_Basic"* ]]; then
        lora_type="basic_lora"
    elif [[ "$log_path" == *"r4_Full"* ]]; then
        lora_type="full"
    else
        echo "  [Skip] Unknown lora_type for $log_path"
        continue
    fi

    # Extract init_cls and increment
    if [[ "$log_path" == *"I10_C10"* ]]; then
        init_cls=10
        increment=10
    elif [[ "$log_path" == *"I20_C20"* ]]; then
        init_cls=20
        increment=20
    else
        echo "  [Warn] Unknown class settings for $log_path, defaulting to 20/20."
        init_cls=20
        increment=20
    fi

    # Assign current GPU
    gpu=${GPUS[$gpu_idx]}
    
    # Generate log name
    log_name=$(echo "$log_path" | sed 's/\//_/g')
    log_file="logs/extract_parallel/extract_${log_name}.log"

    echo "  [Run] $dataset | $lora_type -> GPU $gpu"

    # Run extraction in background
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT_DIR/extract_features.py" \
        --dataset "$dataset" \
        --log_path "$log_path" \
        --lora_type "$lora_type" \
        --init_cls "$init_cls" \
        --increment "$increment" \
        --seed 1993 > "$log_file" 2>&1 &
    
    ((current_jobs++))
    ((gpu_idx=(gpu_idx+1)%MAX_JOBS))

    # Wait if pool is full
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        echo "  [Wait] Process pool full, waiting for current batch..."
        wait
        current_jobs=0
        gpu_idx=0
        echo "  [Resume] Continuing with next batch..."
    fi
done

wait
echo "✅ All seed 1993 extractions completed!"
