#!/usr/bin/env bash
set -euo pipefail

echo "Starting All Experiments (Within-Domain + Cross-Domain)..."

# 创建总日志目录
MASTER_LOG_DIR="logs/all_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

echo "=========================================="
echo "1. Running Within-Domain Experiments"
echo "=========================================="

# 运行within-domain实验
bash sh/within_domain_experiments.sh

echo ""
echo "=========================================="
echo "2. Running Cross-Domain Experiments"
echo "=========================================="

# 运行cross-domain实验
bash sh/cross_domain_experiments_new.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

# 计算总实验数量
WITHIN_DOMAIN_EXPERIMENTS=$((4 * 3 * 8))  # 4 datasets × 3 seeds × 8 variants
CROSS_DOMAIN_EXPERIMENTS=$((3 * 7))  # 3 seeds × 7 variants
TOTAL_EXPERIMENTS=$((WITHIN_DOMAIN_EXPERIMENTS + CROSS_DOMAIN_EXPERIMENTS))

echo "Within-Domain experiments: $WITHIN_DOMAIN_EXPERIMENTS"
echo "Cross-Domain experiments: $CROSS_DOMAIN_EXPERIMENTS"
echo "Total experiments: $TOTAL_EXPERIMENTS"