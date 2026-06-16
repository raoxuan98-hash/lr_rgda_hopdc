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
echo "运行Alpha约束实验 - 并行执行"
echo "=========================================="
echo "GPU列表: ${GPUS[*]}"
echo "模型列表: ${MODELS[*]}"
echo "迭代次数: $ITERATIONS"
echo "每类样本数: $NUM_SHOTS"
echo "输出目录: $BASE_OUTPUT_DIR"
echo "=========================================="

# 并行运行实验
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    MODEL=${MODELS[$i]}
    
    echo "在GPU $GPU上启动模型 $MODEL 的实验..."
    
    # 使用nohup在后台运行，并将输出重定向到日志文件
    nohup python3 classifier_ablation/experiments/exp2_alpha_constraint.py \
        --model_name $MODEL \
        --gpu $GPU \
        --iterations $ITERATIONS \
        --num_shots $NUM_SHOTS \
        --base_output_dir $BASE_OUTPUT_DIR \
        > $LOG_DIR/${MODEL}_gpu${GPU}.log 2>&1 &
    
    echo "已启动进程: $!"
done

echo "=========================================="
echo "所有实验已在后台启动"
echo "使用以下命令查看日志:"
echo "  tail -f $LOG_DIR/<model>_gpu<gpu>.log"
echo "使用以下命令检查进程状态:"
echo "  ps aux | grep exp2_alpha_constraint"
echo "=========================================="