#!/usr/bin/env bash
set -euo pipefail

# 测试并行执行逻辑的脚本
# 使用简单的echo命令代替实际的python训练

echo "Testing parallel execution logic..."

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# 可用的GPU设备
GPUS=(0 1 2 4)

# 创建日志目录
LOG_DIR="logs/test_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 并行运行所有实验
PIDS=()
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    GPU="${GPUS[$((i % ${#GPUS[@]}))]}"
    
    # 为每个数据集创建一个子脚本来处理所有种子
    cat > "$LOG_DIR/run_${DATASET}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
SEEDS=(${SEEDS[*]})

for SEED in "\${SEEDS[@]}"; do
    echo "Simulating experiment: dataset=\$DATASET, seed=\$SEED, GPU=\$GPU"
    sleep 2  # 模拟实验运行时间
    echo "Completed experiment: dataset=\$DATASET, seed=\$SEED, GPU=\$GPU" >> "\$LOG_DIR/\${DATASET}_seed\${SEED}.log"
done
EOF
    
    chmod +x "$LOG_DIR/run_${DATASET}.sh"
    
    # 在后台运行每个数据集的实验
    echo "Starting simulated experiments for $DATASET on GPU $GPU"
    "$LOG_DIR/run_${DATASET}.sh" &
    PIDS+=($!)
done

# 等待所有实验完成
echo "Waiting for all simulated experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "All simulated experiments completed. Logs saved to $LOG_DIR"

# 验证结果
echo "Verifying results..."
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LOG_FILE="$LOG_DIR/${DATASET}_seed${SEED}.log"
        if [ -f "$LOG_FILE" ]; then
            echo "✓ $DATASET seed $SEED completed"
        else
            echo "✗ $DATASET seed $SEED failed"
        fi
    done
done

echo "Test completed successfully!"