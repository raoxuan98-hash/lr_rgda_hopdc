#!/bin/bash

# Within-domain数据集评估脚本（可配置版本）
# 支持可配置的架构和LoRA类型
# 评估四个数据集，每个数据集使用一个GPU
# 每个数据集三个随机种子(1993, 1996, 1997)，顺序执行
#
# 使用方法:
# ./run_within_domain_evaluation_configurable.sh [vit_type] [lora_type]
# 例如: ./run_within_domain_evaluation_configurable.sh vit-b-p16 basic_lora

# 默认参数
DEFAULT_VIT_TYPE="vit-b-p16"
DEFAULT_LORA_TYPE="basic_lora"

# 获取命令行参数（可选）
VIT_TYPE=${1:-$DEFAULT_VIT_TYPE}
LORA_TYPE=${2:-$DEFAULT_LORA_TYPE}

# 设置数据集列表
datasets=('imagenet-r' 'cifar100_224' 'cub200_224' 'cars196_224')

# 设置随机种子列表
seeds=(1993 1996 1997)

# 设置GPU列表（每个数据集使用一个GPU）
gpus=(0 1 2 4)

# 其他参数
SMART_DEFAULTS_FLAG="--smart_defaults"
CROSS_DOMAIN_FLAG=""

# 创建日志目录
log_dir="within_domain_evaluation_logs_$(date +%Y-%m-%d)"
mkdir -p $log_dir

# 函数：处理单个数据集
process_dataset() {
    local dataset=$1
    local gpu_id=$2
    local vit_type=$3
    local lora_type=$4
    local smart_defaults_flag=$5
    local cross_domain_flag=$6
    local log_dir=$7
    shift 7
    local seeds=("$@")
    
    echo "=========================================="
    echo "开始处理数据集: $dataset (GPU: $gpu_id)"
    echo "架构: $vit_type, LoRA类型: $lora_type"
    echo "=========================================="
    
    # 为当前数据集创建日志子目录
    dataset_log_dir="$log_dir/${dataset}_${vit_type}"
    mkdir -p $dataset_log_dir
    
    # 遍历每个随机种子（顺序执行）
    for seed in "${seeds[@]}"; do
        echo "----------------------------------------"
        echo "数据集: $dataset, 随机种子: $seed, GPU: $gpu_id"
        echo "----------------------------------------"
        
        # 设置CUDA设备
        export CUDA_VISIBLE_DEVICES=$gpu_id
        
        # 构建命令 - 使用python3 main.py --形式
        cmd="python3 main.py \
            --dataset $dataset \
            --vit_type $vit_type \
            --lora_type $lora_type \
            --seed_list $seed \
            $smart_defaults_flag \
            $cross_domain_flag"
        
        echo "执行命令: $cmd"
        
        # 执行命令并记录日志
        log_file="$dataset_log_dir/seed_${seed}.log"
        eval $cmd 2>&1 | tee "$log_file"
        
        # 检查执行结果
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ 数据集 $dataset 种子 $seed 执行成功"
        else
            echo "✗ 数据集 $dataset 种子 $seed 执行失败，请检查日志: $log_file"
        fi
        
        echo ""
    done
    
    echo "=========================================="
    echo "数据集 $dataset 处理完成！"
    echo "=========================================="
}

echo "=========================================="
echo "开始within-domain数据集评估（可配置版本）"
echo "=========================================="
echo "架构: $VIT_TYPE"
echo "LoRA类型: $LORA_TYPE"
echo "数据集: ${datasets[*]}"
echo "随机种子: ${seeds[*]}"
echo "GPU分配: ${gpus[*]}"
echo "日志目录: $log_dir"
echo "执行策略："
echo "- 不同数据集：并行运行（使用不同GPU）"
echo "- 同一数据集的不同随机种子：串行运行"
echo "=========================================="

# 验证架构和LoRA类型是否有效
echo "验证配置参数..."

# 检查vit_type是否有效
valid_vit_types=("vit-b-p16" "vit-b-p16-dino" "vit-b-p16-mae" "vit-b-p16-clip" "vit-b-p16-mocov3")
vit_type_valid=false
for valid_type in "${valid_vit_types[@]}"; do
    if [ "$VIT_TYPE" = "$valid_type" ]; then
        vit_type_valid=true
        break
    fi
done

if [ "$vit_type_valid" = false ]; then
    echo "错误: 无效的架构类型 '$VIT_TYPE'"
    echo "支持的架构类型: ${valid_vit_types[*]}"
    exit 1
fi

# 检查lora_type是否有效
valid_lora_types=("basic_lora" "sgp_lora" "nsp_lora" "full")
lora_type_valid=false
for valid_type in "${valid_lora_types[@]}"; do
    if [ "$LORA_TYPE" = "$valid_type" ]; then
        lora_type_valid=true
        break
    fi
done

if [ "$lora_type_valid" = false ]; then
    echo "错误: 无效的LoRA类型 '$LORA_TYPE'"
    echo "支持的LoRA类型: ${valid_lora_types[*]}"
    exit 1
fi

echo "✓ 参数验证通过"

# 用于跟踪进程的PID数组
pids=()

# 并行启动每个数据集的处理
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    gpu_id=${gpus[$i]}
    
    # 在后台启动每个数据集的处理进程
    process_dataset "$dataset" "$gpu_id" "$VIT_TYPE" "$LORA_TYPE" "$SMART_DEFAULTS_FLAG" "$CROSS_DOMAIN_FLAG" "$log_dir" "${seeds[@]}" &
    
    # 记录进程PID
    pids+=($!)
    
    echo "已启动数据集 $dataset 的处理进程 (PID: ${pids[$i]}, GPU: $gpu_id)"
done

echo ""
echo "所有数据集处理进程已启动，等待完成..."

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
    echo "进程 $pid 已完成"
done

echo ""
echo "=========================================="
echo "所有within-domain数据集评估完成！"
echo "日志保存在: $log_dir"
echo "=========================================="

# 显示结果汇总
echo ""
echo "结果汇总："
for dataset in "${datasets[@]}"; do
    dataset_log_dir="$log_dir/${dataset}_${VIT_TYPE}"
    if [ -d "$dataset_log_dir" ]; then
        echo "数据集 $dataset:"
        for seed in "${seeds[@]}"; do
            log_file="$dataset_log_dir/seed_${seed}.log"
            if [ -f "$log_file" ]; then
                if grep -q "训练完成\|Training completed\|Finished" "$log_file"; then
                    echo "  ✓ 种子 $seed: 成功"
                else
                    echo "  ✗ 种子 $seed: 失败或未完成"
                fi
            else
                echo "  ? 种子 $seed: 日志文件不存在"
            fi
        done
    else
        echo "数据集 $dataset: 无结果目录"
    fi
done

echo ""
echo "详细日志请查看: $log_dir"