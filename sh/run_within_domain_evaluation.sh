#!/bin/bash

# Within-domain数据集评估脚本（并行版本）
# 使用vit-b-p16架构，并行评估四个数据集，每个数据集使用一个GPU
# 每个数据集三个随机种子(1993, 1996, 1997)，顺序执行
# 使用basic_lora和smart_defaults
#
# 执行策略：
# - 不同数据集：并行运行（使用不同GPU）
# - 同一数据集的不同随机种子：串行运行

# 设置数据集列表
datasets=('imagenet-r' 'cifar100_224' 'cub200_224' 'cars196_224')

# 设置随机种子列表
seeds=(1993 1996 1997)

# 设置GPU列表（每个数据集使用一个GPU）
gpus=(0 1 2 4)

# 基础参数
vit_type="vit-b-p16"
lora_type="basic_lora"
smart_defaults_flag="--smart_defaults"

# 创建日志目录
log_dir="within_domain_evaluation_logs_$(date +%Y-%m-%d)"
mkdir -p $log_dir

# 用于跟踪进程的PID数组
pids=()

# 函数：处理单个数据集
process_dataset() {
    local dataset=$1
    local gpu_id=$2
    local vit_type=$3
    local lora_type=$4
    local smart_defaults_flag=$5
    local log_dir=$6
    shift 6
    local seeds=("$@")
    
    echo "=========================================="
    echo "开始处理数据集: $dataset (GPU: $gpu_id)"
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
        
        # 构建命令
        cmd="python main.py \
            --dataset $dataset \
            --vit_type $vit_type \
            --lora_type $lora_type \
            --seed_list $seed \
            $smart_defaults_flag \
            "
        
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
echo "开始within-domain数据集评估"
echo "执行策略："
echo "- 不同数据集：并行运行（使用不同GPU）"
echo "- 同一数据集的不同随机种子：串行运行"
echo "=========================================="
echo "架构: $vit_type"
echo "LoRA类型: $lora_type"
echo "数据集: ${datasets[*]}"
echo "随机种子: ${seeds[*]}"
echo "日志目录: $log_dir"
echo "=========================================="

# 并行启动每个数据集的处理
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    gpu_id=${gpus[$i]}
    
    # 在后台启动每个数据集的处理进程
    process_dataset "$dataset" "$gpu_id" "$vit_type" "$lora_type" "$smart_defaults_flag" "$log_dir" "${seeds[@]}" &
    
    # 记录进程PID
    pids+=($!)
    
    echo "已启动数据集 $dataset 的处理进程 (PID: ${pids[$i]}, GPU: $gpu_id)"
done

echo "所有数据集处理进程已启动，等待完成..."

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
    echo "进程 $pid 已完成"
done

echo "=========================================="
echo "所有within-domain数据集评估完成！"
echo "日志保存在: $log_dir"
echo "=========================================="