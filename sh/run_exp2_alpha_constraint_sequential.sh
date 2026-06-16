#!/bin/bash

# 设置GPU和模型架构
GPUS=(0 1 2 4)
MODELS=("vit-b-p16" "vit-b-p16-clip" "vit-b-p16-dino" "vit-b-p16-mocov3")

# 设置实验参数
ITERATIONS=0
NUM_SHOTS=128
BASE_OUTPUT_DIR="实验结果保存/分类器消融实验"

# 创建日志目录
LOG_DIR="logs/exp2_alpha_constraint"
mkdir -p $LOG_DIR

# 打印实验信息
echo "=========================================="
echo "运行Alpha约束实验 - 串行执行"
echo "=========================================="
echo "GPU列表: ${GPUS[*]}"
echo "模型列表: ${MODELS[*]}"
echo "迭代次数: $ITERATIONS"
echo "每类样本数: $NUM_SHOTS"
echo "输出目录: $BASE_OUTPUT_DIR"
echo "=========================================="

# 串行运行实验
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    MODEL=${MODELS[$i]}
    
    echo "=========================================="
    echo "在GPU $GPU上运行模型 $MODEL 的实验..."
    echo "=========================================="
    
    # 运行实验
    python classifier_ablation/experiments/exp2_alpha_constraint.py \
        --model_name $MODEL \
        --gpu $GPU \
        --iterations $ITERATIONS \
        --num_shots $NUM_SHOTS \
        --base_output_dir $BASE_OUTPUT_DIR \
        2>&1 | tee $LOG_DIR/${MODEL}_gpu${GPU}.log
    
    echo "模型 $MODEL 在GPU $GPU 上的实验完成"
    echo ""
done

echo "=========================================="
echo "所有实验已完成"
echo "=========================================="