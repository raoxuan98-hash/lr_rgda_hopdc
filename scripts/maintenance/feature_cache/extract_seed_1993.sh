#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU 3 MUST NOT be used
export CUDA_VISIBLE_DEVICES="0,1,2,4,5"

# Find all seed_1993 directories in RGDA_WD_2025-12-19-within
directories=$(find RGDA_WD_2025-12-19-within -name "seed_1993" -type d)

for log_path in $directories; do
    # Skip test runs (IT100)
    if [[ "$log_path" == *"IT100/"* ]]; then
        echo "Skipping test run: $log_path"
        continue
    fi

    echo "Processing: $log_path"

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
        echo "Unknown dataset for $log_path, skipping."
        continue
    fi

    # Extract lora_type
    if [[ "$log_path" == *"r4_Basic"* ]]; then
        lora_type="basic_lora"
    elif [[ "$log_path" == *"r4_Full"* ]]; then
        lora_type="full"
    else
        echo "Unknown lora_type for $log_path, skipping."
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
        echo "Unknown class settings for $log_path, defaulting to 20/20."
        init_cls=20
        increment=20
    fi

    # Run extraction
    python "$SCRIPT_DIR/extract_features.py" \
        --dataset "$dataset" \
        --log_path "$log_path" \
        --lora_type "$lora_type" \
        --init_cls "$init_cls" \
        --increment "$increment" \
        --seed 1993

done

echo "All extractions for seed 1993 completed."
