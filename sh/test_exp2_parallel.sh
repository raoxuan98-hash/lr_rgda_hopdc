#!/bin/bash

# 测试并行运行exp2_alpha_constraint实验，只运行一个模型进行测试
# GPU 0: vit-b-p16

# 设置工作目录
cd /home/raoxuan/projects/low_rank_rda

# 创建日志目录
LOG_DIR="实验结果保存/分类器消融实验/logs"
mkdir -p $LOG_DIR

# 定义测试模型和GPU
MODEL="vit-b-p16"
GPU="0"

# 启动测试实验
echo "开始测试运行exp2_alpha_constraint实验..."
echo "模型: $MODEL, GPU: $GPU"
echo ""

log_file="${LOG_DIR}/${MODEL}_gpu${GPU}_test.log"

echo "在GPU $GPU 上启动 $模型 测试实验，日志保存到 $log_file"
python classifier_ablation/experiments/exp2_alpha_constraint_parallel.py --model $MODEL --gpu $GPU > $log_file 2>&1

echo ""
echo "测试实验完成，请检查日志文件: $log_file"
echo ""
echo "使用以下命令查看实验结果:"
echo "  tail -n 20 $log_file"