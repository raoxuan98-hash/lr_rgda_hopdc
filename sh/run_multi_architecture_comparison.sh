#!/bin/bash

# 多架构对比实验脚本
# 用于汇集不同架构的已保存结构并在一个图上呈现

echo "=========================================="
echo "多架构分类器性能对比实验"
echo "=========================================="

# 设置环境
source /home/raoxuan/miniconda3/bin/activate raoxuan
cd /home/raoxuan/projects/low_rank_rda

# 设置参数
MODEL_NAMES=("vit-b-p16" "vit-b-p16-clip" "vit-b-p16-dino" "vit-b-p16-mocov3")
BASE_OUTPUT_DIR="实验结果保存/分类器消融实验"
ITERATIONS=0

# 创建输出目录
OUTPUT_DIR="${BASE_OUTPUT_DIR}/multi_architecture_comparison"
mkdir -p $OUTPUT_DIR

echo "模型列表: ${MODEL_NAMES[*]}"
echo "基础输出目录: $BASE_OUTPUT_DIR"
echo "迭代次数: $ITERATIONS"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 检查所有模型的结果文件是否存在
echo "检查模型结果文件..."
ALL_EXISTS=true

for model in "${MODEL_NAMES[@]}"; do
    MODEL_DIR="${BASE_OUTPUT_DIR}/${model}_iter${ITERATIONS}"
    NPZ_FILE="${MODEL_DIR}/${model}_constraint_results.npz"
    CSV_FILE="${MODEL_DIR}/${model}_constraint_results.csv"
    
    if [[ -f "$NPZ_FILE" ]]; then
        echo "✓ 找到 $model 的NPZ结果文件: $NPZ_FILE"
    elif [[ -f "$CSV_FILE" ]]; then
        echo "✓ 找到 $model 的CSV结果文件: $CSV_FILE"
    else
        echo "✗ 未找到 $model 的结果文件"
        ALL_EXISTS=false
    fi
done

if [[ "$ALL_EXISTS" = false ]]; then
    echo "警告: 部分模型的结果文件不存在，将跳过这些模型"
fi

echo "=========================================="
echo "开始生成多架构对比图..."

# 运行多架构对比脚本
python classifier_ablation/experiments/plot_multi_architecture_comparison.py

# 检查是否成功生成文件
if [[ -f "${OUTPUT_DIR}/multi_architecture_subplot_comparison.png" ]] && [[ -f "${OUTPUT_DIR}/multi_architecture_combined_comparison.png" ]]; then
    echo "✓ 成功生成多架构对比图"
else
    echo "✗ 生成多架构对比图失败"
    exit 1
fi

echo "=========================================="
echo "显示结果分析..."

# 运行结果分析脚本
python classifier_ablation/experiments/show_multi_architecture_results.py

echo "=========================================="
echo "多架构对比实验完成"
echo "生成的文件:"
echo "1. 子图对比: ${OUTPUT_DIR}/multi_architecture_subplot_comparison.png"
echo "2. 综合对比图: ${OUTPUT_DIR}/multi_architecture_combined_comparison.png"
echo "3. 性能汇总表: ${OUTPUT_DIR}/performance_summary.csv"
echo "=========================================="

# 询问是否查看图像
read -p "是否要查看生成的图像? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v eog &> /dev/null; then
        echo "使用Eye of GNOME查看图像..."
        eog "${OUTPUT_DIR}/multi_architecture_subplot_comparison.png" &
        eog "${OUTPUT_DIR}/multi_architecture_combined_comparison.png" &
    elif command -v display &> /dev/null; then
        echo "使用ImageMagick display查看图像..."
        display "${OUTPUT_DIR}/multi_architecture_subplot_comparison.png" &
        display "${OUTPUT_DIR}/multi_architecture_combined_comparison.png" &
    else
        echo "未找到图像查看器，请手动查看生成的图像文件"
    fi
fi