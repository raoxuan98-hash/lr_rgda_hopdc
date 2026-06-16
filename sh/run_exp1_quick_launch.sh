#!/bin/bash

# 简化版性能曲面等高线图实验启动脚本

echo "=== 性能曲面等高线图实验启动器 ==="
echo "请选择要运行的实验配置："
echo ""

# 显示所有可用配置
declare -a CONFIGS=(
    "vit-b-p16:16:0"
    "vit-b-p16:32:1" 
    "vit-b-p16:64:2"
    "vit-b-p16:128:3"
    "vit-b-p16-clip:16:0"
    "vit-b-p16-clip:32:1"
    "vit-b-p16-clip:64:2"
    "vit-b-p16-clip:128:3"
    "vit-b-p16-dino:16:0"
    "vit-b-p16-dino:32:1"
    "vit-b-p16-dino:64:2"
    "vit-b-p16-dino:128:3"
    "vit-b-p16-mocov3:16:0"
    "vit-b-p16-mocov3:32:1"
    "vit-b-p16-mocov3:64:2"
    "vit-b-p16-mocov3:128:3"
)

for i in "${!CONFIGS[@]}"; do
    IFS=':' read -ra CONFIG <<< "${CONFIGS[i]}"
    MODEL="${CONFIG[0]}"
    RANK="${CONFIG[1]}"
    GPU="${CONFIG[2]}"
    echo "$((i+1)). $MODEL (rank=$RANK, GPU=$GPU)"
done

echo "0. 退出"
echo ""
read -p "请选择配置编号 (1-16, 0退出): " choice

if [ "$choice" -eq 0 ]; then
    echo "退出实验"
    exit 0
elif [ "$choice" -ge 1 ] && [ "$choice" -le ${#CONFIGS[@]} ]; then
    selected_config="${CONFIGS[$((choice-1))]}"
    IFS=':' read -ra CONFIG <<< "$selected_config"
    MODEL="${CONFIG[0]}"
    RANK="${CONFIG[1]}"
    GPU="${CONFIG[2]}"
    
    echo "启动实验: $MODEL rank=$RANK on GPU $GPU"
    
    # 创建日志目录
    mkdir -p "./logs"
    
    # 启动实验
    nohup python classifier_ablation/experiments/exp1_performance_surface.py \
        --model "$MODEL" \
        --rank "$RANK" \
        --gpu "$GPU" \
        --iterations 0 \
        --num_shots 128 \
        > "./logs/exp1_${MODEL}_rank${RANK}_gpu${GPU}.log" 2>&1 &
    
    PID=$!
    echo "实验已启动，进程ID: $PID"
    echo "日志文件: ./logs/exp1_${MODEL}_rank${RANK}_gpu${GPU}.log"
    echo ""
    echo "监控命令:"
    echo "  实时查看: tail -f ./logs/exp1_${MODEL}_rank${RANK}_gpu${GPU}.log"
    echo "  进程状态: ps aux | grep $PID"
    echo "  终止进程: kill $PID"
    
else
    echo "无效选择"
    exit 1
fi