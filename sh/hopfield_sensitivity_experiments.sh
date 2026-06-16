#!/bin/bash

# Configuration
DATASETS=("imagenet-r" "cars196_224")
GPUS=(4 5)
VIT_TYPES=("vit-b-p16-mocov3")
LORA_TYPES=("full")


# Default parameters
DEFAULT_TEMP=0.05
DEFAULT_TOPK=400
DEFAULT_AUX_SIZE=2048

# Ensure script is executable: chmod +x hopfield_sensitivity_experiments.sh

run_experiment() {
    local DATASET=$1
    local GPU=$2
    local VIT_TYPE=$3
    local LORA_TYPE=$4
    local TEMP=$5
    local TOPK=$6
    local AUX_SIZE=$7
    local EXPERIMENT_NAME=$8
    
    echo "============================================"
    echo "Starting experiment: $EXPERIMENT_NAME"
    echo "Dataset: $DATASET on GPU $GPU | Vit_Type: $VIT_TYPE | Lora_Type: $LORA_TYPE"
    echo "Parameters: temp=$TEMP, topk=$TOPK, aux_size=$AUX_SIZE"
    echo "============================================"
    echo "[$(date)] Running $EXPERIMENT_NAME"
    
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        --dataset "$DATASET" \
        --vit_type "$VIT_TYPE" \
        --lora_type "$LORA_TYPE" \
        --smart_defaults \
        --seed_list 1990 \
        --gamma_kd 1.0 \
        --hopfield_temp "$TEMP" \
        --hopfield_topk "$TOPK" \
        --auxiliary_data_size "$AUX_SIZE" \
        --compensator_types "SeqFT" "SeqFT + HopDC" "SeqFT + linear"\
        # --enable_weight_interpolation \

    echo "[$(date)] Completed: $EXPERIMENT_NAME"
}

# # Experiment 1: 固定temp为0.05，变动hopfield_topk为=100， 200， 400， 800
# echo "################################################################"
# echo "Experiment 1: Fixed temp=$DEFAULT_TEMP, varying hopfield_topk"
# echo "################################################################"

# TOPK_VALUES=(100 200 400 800)
# for TOPK in "${TOPK_VALUES[@]}"; do
#     echo "Starting experiments with hopfield_topk = $TOPK"
    
#     # 为当前topk值启动所有数据集的并行运行
#     for i in "${!DATASETS[@]}"; do
#         DATASET=${DATASETS[$i]}
#         GPU=${GPUS[$i]}
#         EXPERIMENT_NAME="exp1_temp${DEFAULT_TEMP}_topk${TOPK}_${DATASET}"
#         run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" "$DEFAULT_TEMP" "$TOPK" "$DEFAULT_AUX_SIZE" "$EXPERIMENT_NAME" &
#     done

#     # 等待当前topk值的所有作业完成
#     wait
    
#     echo "Completed all experiments with hopfield_topk = $TOPK"
# done

# echo "Experiment 1 completed!"

# # Experiment 2: 固定hopfield_topk为400，变动temp为0.01， 0.05， 0.25， 1.0
# echo "################################################################"
# echo "Experiment 2: Fixed hopfield_topk=$DEFAULT_TOPK, varying temp"
# echo "################################################################"

TEMP_VALUES=(1.0)
for TEMP in "${TEMP_VALUES[@]}"; do
    echo "Starting experiments with temp = $TEMP"
    
    # 为当前temp值启动所有数据集的并行运行
    for i in "${!DATASETS[@]}"; do
        DATASET=${DATASETS[$i]}
        GPU=${GPUS[$i]}
        EXPERIMENT_NAME="exp2_temp${TEMP}_topk${DEFAULT_TOPK}_${DATASET}"
        run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" "$TEMP" "$DEFAULT_TOPK" "$DEFAULT_AUX_SIZE" "$EXPERIMENT_NAME" &
    done

    # 等待当前temp值的所有作业完成
    wait
    
    echo "Completed all experiments with temp = $TEMP"
done

echo "Experiment 2 completed!"

# Experiment 3: 固定temp为0.05, hopfield_topk=400, 变动auxiliary_data_size为512, 1024, 2048, 4096
echo "################################################################"
echo "Experiment 3: Fixed temp=$DEFAULT_TEMP, hopfield_topk=$DEFAULT_TOPK, varying auxiliary_data_size"
echo "################################################################"

AUX_SIZE_VALUES=(256 512 1024 2048)
for AUX_SIZE in "${AUX_SIZE_VALUES[@]}"; do
    echo "Starting experiments with auxiliary_data_size = $AUX_SIZE"
    
    # 为当前aux_size值启动所有数据集的并行运行
    for i in "${!DATASETS[@]}"; do
        DATASET=${DATASETS[$i]}
        GPU=${GPUS[$i]}
        EXPERIMENT_NAME="exp3_temp${DEFAULT_TEMP}_topk${DEFAULT_TOPK}_aux${AUX_SIZE}_${DATASET}"
        run_experiment "$DATASET" "$GPU" "${VIT_TYPES[0]}" "${LORA_TYPES[0]}" "$DEFAULT_TEMP" "$DEFAULT_TOPK" "$AUX_SIZE" "$EXPERIMENT_NAME" &
    done

    # 等待当前aux_size值的所有作业完成
    wait
    
    echo "Completed all experiments with auxiliary_data_size = $AUX_SIZE"
done

echo "Experiment 3 completed!"

echo "All HopDC sensitivity experiments completed!"