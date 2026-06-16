#!/bin/bash

# 性能曲面等高线图混合并行训练脚本
# 不同GPU之间并行运行，同个GPU上的不同秩实验串行运行

# =============================================================================
# 实验参数配置 - 可根据需要修改
# =============================================================================

# 要测试的rank值列表
RANKS=(128)

# 要测试的模型列表和对应的GPU
declare -A MODEL_GPU_MAP
MODEL_GPU_MAP["vit-b-p16"]="0"
MODEL_GPU_MAP["vit-b-p16-clip"]="1"
MODEL_GPU_MAP["vit-b-p16-dino"]="2"
MODEL_GPU_MAP["vit-b-p16-mocov3"]="4"

# 临时脚本目录
TEMP_DIR="/tmp/exp1_performance_surface"
mkdir -p $TEMP_DIR

# =============================================================================

# 设置日志目录
LOG_DIR="./logs/exp1_performance_surface"
mkdir -p $LOG_DIR

# 动态生成实验配置
declare -A GPU_EXPERIMENTS

echo "实验配置预览:"
echo "要测试的rank值: ${RANKS[@]}"
echo "模型和GPU映射:"

for model in "${!MODEL_GPU_MAP[@]}"; do
    gpu_id="${MODEL_GPU_MAP[$model]}"
    echo "  $model -> GPU $gpu_id"
    
    # 为每个GPU分配实验
    experiments=""
    for rank in "${RANKS[@]}"; do
        exp_name="${model}-rank${rank}"
        if [ -z "$experiments" ]; then
            experiments="${model}:${rank}:${gpu_id}:${exp_name}"
        else
            experiments="$experiments ${model}:${rank}:${gpu_id}:${exp_name}"
        fi
    done
    
    GPU_EXPERIMENTS[$gpu_id]="$experiments"
    echo "    实验: $experiments"
    echo ""
done

echo "按GPU分组的实验配置:"
for gpu in "${!GPU_EXPERIMENTS[@]}"; do
    echo "  GPU $gpu: ${GPU_EXPERIMENTS[$gpu]}"
done
echo ""

# 获取所有GPU ID
GPUS=(${!GPU_EXPERIMENTS[@]})

# 创建后台进程数组
declare -a PIDS=()

echo "启动性能曲面等高线图实验（混合并行：不同GPU并行，同GPU串行）..."
echo "使用GPU: ${GPUS[@]}"
echo "架构分配:"
echo "每个GPU上串行运行多个rank值的实验:"
for model in "${!MODEL_GPU_MAP[@]}"; do
    gpu_id="${MODEL_GPU_MAP[$model]}"
    echo "  $model: GPU $gpu_id (串行运行rank=${RANKS[@]})"
done
echo "=================================="

# 记录开始时间
start_time=$(date)
echo "实验开始时间: $start_time"
echo ""

# 为每个GPU启动串行实验进程
for gpu_id in "${GPUS[@]}"; do
    experiments="${GPU_EXPERIMENTS[$gpu_id]}"
    
    echo "启动GPU $gpu_id 上的串行实验进程..."
    
    # 创建临时脚本文件来运行串行实验
    temp_script="${TEMP_DIR}/gpu_${gpu_id}_serial_experiments.sh"
    cat > "$temp_script" << EOF
#!/bin/bash

# GPU $gpu_id 上的串行实验脚本
experiments=($experiments)

echo "GPU $gpu_id 开始运行 \${#experiments[@]} 个实验..."

for exp in "\${experiments[@]}"; do
    # 解析实验配置
    IFS=':' read -ra CONFIG <<< "\$exp"
    MODEL_NAME="\${CONFIG[0]}"
    RANK="\${CONFIG[1]}"
    GPU_ID="\${CONFIG[2]}"
    EXP_NAME="\${CONFIG[3]}"
    
    echo "=================================="
    echo "GPU \$GPU_ID 启动实验: \$EXP_NAME"
    echo "  模型: \$MODEL_NAME"
    echo "  Rank: \$RANK"
    echo "  开始时间: \$(date)"
    echo "=================================="
    
    # 运行实验
    LOG_FILE="$LOG_DIR/\${EXP_NAME}.log"
    CUDA_VISIBLE_DEVICES=\$GPU_ID python classifier_ablation/experiments/exp1_performance_surface.py \\
        --model "\$MODEL_NAME" \\
        --rank "\$RANK" \\
        --gpu "\$GPU_ID" \\
        --iterations 0 \\
        --num_shots 128 \\
        > "\$LOG_FILE" 2>&1
    
    # 检查实验是否成功完成
    exit_code=\$?
    if [ \$exit_code -eq 0 ]; then
        echo "✅ GPU \$GPU_ID 实验 \$EXP_NAME 成功完成"
        # 提取最佳性能
        if grep -q "最佳参数:" "\$LOG_FILE"; then
            echo "最佳性能: \$(grep "最佳参数:" "\$LOG_FILE" | tail -1)"
        fi
    else
        echo "❌ GPU \$GPU_ID 实验 \$EXP_NAME 失败，退出码: \$exit_code"
        echo "查看日志文件: \$LOG_FILE"
    fi
    
    echo "  结束时间: \$(date)"
    echo ""
done

echo "GPU $gpu_id 上的所有实验完成!"
EOF

    # 使临时脚本可执行
    chmod +x "$temp_script"
    
    # 在后台启动GPU的串行实验进程
    nohup "$temp_script" > "$LOG_DIR/gpu_${gpu_id}_serial.log" 2>&1 &
    
    # 记录进程ID
    PID=$!
    PIDS+=($PID)
    
    echo "GPU $gpu_id 串行实验进程已启动，进程ID: $PID"
    echo ""
    
    # 给系统一点时间启动进程
    sleep 2
done

echo "所有GPU串行实验进程已启动..."
echo "进程ID列表: ${PIDS[@]}"
echo "=================================="

# 监控进程状态
monitor_experiments() {
    local active_pids=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            active_pids+=($pid)
        else
            echo "GPU进程 $pid 已完成"
        fi
    done
    
    if [ ${#active_pids[@]} -eq 0 ]; then
        echo "所有GPU实验进程已完成!"
        return 1
    else
        echo "运行中的GPU进程: ${#active_pids[@]}/${#PIDS[@]}"
        PIDS=("${active_pids[@]}")
        return 0
    fi
}

# 定期检查进程状态
check_interval=30  # 30秒检查一次
while true; do
    if ! monitor_experiments; then
        break
    fi
    echo "等待 ${check_interval} 秒后再次检查..."
    sleep $check_interval
done

# 记录结束时间
end_time=$(date)
echo "=================================="
echo "所有实验已完成!"
echo "实验开始时间: $start_time"
echo "实验结束时间: $end_time"
echo "=================================="

# 汇总结果
echo "实验结果汇总:"
for gpu_id in "${GPUS[@]}"; do
    experiments="${GPU_EXPERIMENTS[$gpu_id]}"
    for exp in $experiments; do
        IFS=':' read -ra CONFIG <<< "$exp"
        EXP_NAME="${CONFIG[3]}"
        LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
        
        if [ -f "$LOG_FILE" ]; then
            echo "--- $EXP_NAME ---"
            # 检查是否包含实验完成的标记
            if grep -q "实验完成" "$LOG_FILE"; then
                echo "状态: ✅ 完成"
                # 提取最佳性能
                if grep -q "最佳参数:" "$LOG_FILE"; then
                    grep "最佳参数:" "$LOG_FILE" | tail -1
                fi
            else
                echo "状态: ❓ 未知"
            fi
            echo ""
        fi
    done
done

echo "所有性能曲面等高线图实验完成!"

# 清理临时文件
echo "清理临时文件..."
rm -f "${TEMP_DIR}"/*.sh 2>/dev/null || true
echo "临时文件清理完成"