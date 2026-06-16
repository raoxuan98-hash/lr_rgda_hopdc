#!/usr/bin/env python3
"""
在所有within_domain数据集上运行Rank消融实验的脚本
包括: CIFAR100, ImageNet-R, CUB-200, Cars196
"""

import os
import sys
import subprocess
import argparse

# 设置项目路径
sys.path.append('/home/raoxuan/projects/low_rank_rda')
os.chdir('/home/raoxuan/projects/low_rank_rda')

def run_experiment(dataset, gpu_id, iterations, num_shots):
    """在指定数据集上运行实验"""
    cmd = [
        "python", "classifier_ablation/experiments/exp6_rank_within_domain.py",
        "--gpu", str(gpu_id),
        "--iterations", str(iterations),
        "--num_shots", str(num_shots),
        "--dataset", dataset
    ]
    
    print(f"\n{'='*60}")
    print(f"正在运行数据集: {dataset}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"数据集 {dataset} 运行成功!")
        print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"数据集 {dataset} 运行失败!")
        print("错误:", e.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='在所有within_domain数据集上运行Rank消融实验')
    parser.add_argument('--gpu', type=str, default='0', help='GPU编号')
    parser.add_argument('--iterations', type=int, default=500, help='迭代次数')
    parser.add_argument('--num_shots', type=int, default=128, help='样本数量')
    parser.add_argument('--datasets', nargs='+', 
                        default=['cifar100_224', 'imagenet-r', 'cub200_224', 'cars196_224'],
                        help='要运行的数据集列表')
    
    args = parser.parse_args()
    
    # 数据集列表
    datasets_to_run = args.datasets
    
    print(f"将在以下数据集上运行实验: {datasets_to_run}")
    print(f"GPU: {args.gpu}, 迭代次数: {args.iterations}, 样本数: {args.num_shots}")
    
    # 记录成功和失败的数据集
    successful_datasets = []
    failed_datasets = []
    
    # 依次运行每个数据集
    for dataset in datasets_to_run:
        success = run_experiment(dataset, args.gpu, args.iterations, args.num_shots)
        if success:
            successful_datasets.append(dataset)
        else:
            failed_datasets.append(dataset)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("实验运行总结:")
    print(f"成功的数据集 ({len(successful_datasets)}): {successful_datasets}")
    print(f"失败的数据集 ({len(failed_datasets)}): {failed_datasets}")
    print(f"{'='*60}")
    
    if failed_datasets:
        print("注意: 有数据集运行失败，请检查错误信息")
        sys.exit(1)
    else:
        print("所有数据集运行成功!")
    sys.exit(0)