#!/usr/bin/env python3
"""
在多个数据集上运行Rank消融实验的脚本
依次在CIFAR100、ImageNet-R、CUB-200、Cars196上运行exp6_rank_within_domain.py
"""

import os
import sys
import subprocess
import argparse

# 根据你的环境保留路径设置
sys.path.append('/home/raoxuan/projects/low_rank_rda')
try:
    os.chdir('/home/raoxuan/projects/low_rank_rda')
    print("当前工作目录:", os.getcwd())
except FileNotFoundError:
    print("注意: 目录不存在，请检查路径。当前在:", os.getcwd())

def run_experiment(dataset, gpu_id=0, iterations=500, num_shots=128):
    """
    在指定数据集上运行Rank消融实验
    """
    print(f"\n{'='*60}")
    print(f"开始运行数据集: {dataset}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        "python", "classifier_ablation/experiments/exp6_rank_within_domain.py",
        "--gpu", str(gpu_id),
        "--iterations", str(iterations),
        "--num_shots", str(num_shots),
        "--dataset", dataset
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 打印输出
    if result.stdout:
        print("标准输出:")
        print(result.stdout)
    
    if result.stderr:
        print("错误输出:")
        print(result.stderr)
    
    # 检查返回码
    if result.returncode == 0:
        print(f"✅ 数据集 {dataset} 实验成功完成")
        return True
    else:
        print(f"❌ 数据集 {dataset} 实验失败，返回码: {result.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='在多个数据集上运行Rank消融实验')
    parser.add_argument('--gpu', type=str, default='0', help='GPU编号')
    parser.add_argument('--iterations', type=int, default=500, help='迭代次数')
    parser.add_argument('--num_shots', type=int, default=128, help='样本数量')
    parser.add_argument('--datasets', nargs='+', 
                        default=['cifar100_224', 'imagenet-r', 'cub200_224', 'cars196_224'],
                        help='要运行的数据集列表')
    
    args = parser.parse_args()
    
    # 要运行的数据集列表
    datasets = args.datasets
    
    print(f"将在以下数据集上运行Rank消融实验:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    print(f"使用GPU: {args.gpu}")
    print(f"迭代次数: {args.iterations}")
    print(f"样本数量: {args.num_shots}")
    
    # 记录成功和失败的数据集
    success_datasets = []
    failed_datasets = []
    
    # 依次运行每个数据集
    for dataset in datasets:
        success = run_experiment(dataset, args.gpu, args.iterations, args.num_shots)
        
        if success:
            success_datasets.append(dataset)
        else:
            failed_datasets.append(dataset)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("实验总结")
    print(f"{'='*60}")
    
    if success_datasets:
        print(f"✅ 成功完成的数据集 ({len(success_datasets)}/{len(datasets)}):")
        for dataset in success_datasets:
            print(f"  - {dataset}")
    
    if failed_datasets:
        print(f"❌ 失败的数据集 ({len(failed_datasets)}/{len(datasets)}):")
        for dataset in failed_datasets:
            print(f"  - {dataset}")
    
    if len(success_datasets) == len(datasets):
        print("\n🎉 所有数据集实验都成功完成!")
        return 0
    else:
        print(f"\n⚠️  有 {len(failed_datasets)} 个数据集实验失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)