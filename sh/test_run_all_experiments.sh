#!/usr/bin/env bash
set -euo pipefail

echo "Testing run_all_main_experiments.sh script logic..."

# 创建测试日志目录
TEST_LOG_DIR="test_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_LOG_DIR"

# 数据集列表
DATASETS=("cifar100_224" "imagenet-r" "cub200_224" "cars196_224")
SEEDS=(1993 1996 1997)

# GPU分配
GPUS=(0 1 2 4)

# 测试函数 - 模拟run_experiment_type函数但不实际运行python
test_experiment_type() {
    local experiment_name="$1"
    local lora_type="$2"
    shift 2
    local additional_params=("$@")
    
    echo "=========================================="
    echo "Testing $experiment_name Experiments"
    echo "=========================================="
    
    # 创建实验类型特定的日志目录
    LOG_DIR="$TEST_LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    
    # 并行运行所有数据集
    PIDS=()
    
    for i in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$i]}"
        GPU="${GPUS[$i]}"
        
        # 为每个数据集创建子脚本
        cat > "$LOG_DIR/run_${DATASET}.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail

DATASET="$DATASET"
GPU="$GPU"
LOG_DIR="$LOG_DIR"
LORA_TYPE="$lora_type"
VIT_TYPE="vit-b-p16-mocov3"
SEEDS=(${SEEDS[*]})
ADDITIONAL_PARAMS=(${additional_params[*]})

echo "Testing $experiment_name experiments for \$DATASET on GPU \$GPU"
echo "Parameters: lora_type=\$LORA_TYPE, vit_type=\$VIT_TYPE"
echo "Additional params: \${ADDITIONAL_PARAMS[@]}"

for SEED in "\${SEEDS[@]}"; do
    echo "Would run: CUDA_VISIBLE_DEVICES=\$GPU python main.py --dataset \$DATASET --smart_defaults --lora_type \$LORA_TYPE --vit_type \$VIT_TYPE \${ADDITIONAL_PARAMS[@]} --seed_list \$SEED"
    echo "Test log for \$DATASET seed \$SEED on GPU \$GPU" > "\$LOG_DIR/\${DATASET}_seed\${SEED}.log"
done

echo "Test completed for $experiment_name on \$DATASET"
EOF
        
        chmod +x "$LOG_DIR/run_${DATASET}.sh"
        
        # 在后台运行每个数据集的实验
        echo "Starting test for $experiment_name on $DATASET (GPU $GPU)"
        "$LOG_DIR/run_${DATASET}.sh" &
        PIDS+=($!)
    done
    
    # 等待所有实验完成
    echo "Waiting for all $experiment_name tests to complete..."
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "$experiment_name tests completed. Logs saved to $LOG_DIR"
}

# 测试所有实验类型
test_experiment_type "basic_lora" "basic_lora" "--gamma_kd" "0.0"

test_experiment_type "lora_kd" "basic_lora" \
    "--gamma_kd" "1.0" \
    "--update_teacher_each_task" \
    "--distillation_transform" "identity" \
    "--kd_type" "feat"

test_experiment_type "nsp_lora" "nsp_lora" \
    "--gamma_kd" "0.0" \
    "--nsp_weight" "0.05" \
    "--nsp_eps" "0.05"

test_experiment_type "sgp_lora" "sgp_lora" \
    "--gamma_kd" "0.0" \
    "--weight_temp" "1.0" \
    "--weight_p" "1.0" \
    "--weight_kind" "log1p"

echo "=========================================="
echo "All tests completed!"
echo "Test logs saved to: $TEST_LOG_DIR"
echo "=========================================="

# 验证所有日志文件是否创建成功
echo "Verifying created log files..."
find "$TEST_LOG_DIR" -name "*.log" | sort

# 计算总实验数量
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SEEDS[@]} * 4))  # 4种实验类型
echo "Total experiments would run: $TOTAL_EXPERIMENTS"
echo "Experiments per type: ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds = $((${#DATASETS[@]} * ${#SEEDS[@]}))"