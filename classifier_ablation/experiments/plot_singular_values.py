#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奇异值数据可视化脚本
读取预保存的奇异值数据并进行绘图
"""
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())
sys.path.append('/home/raoxuan/projects/low_rank_rda')

# 设置matplotlib参数
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_singular_values_data(data_path):
    """
    加载奇异值数据
    
    Args:
        data_path: 数据文件路径（.npz或.csv格式）
    
    Returns:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
    """
    if data_path.endswith('.npz'):
        # 加载NumPy格式的数据
        data = np.load(data_path, allow_pickle=True)
        all_singular_values = data['singular_values']
        class_ids = data['class_ids']
        num_samples_per_class = data['num_samples_per_class']
        print(f"从NPZ文件加载了 {len(all_singular_values)} 个类别的奇异值数据")
        
    elif data_path.endswith('.csv'):
        # 加载CSV格式的数据
        df = pd.read_csv(data_path)
        
        all_singular_values = []
        class_ids = []
        num_samples_per_class = []
        
        for _, row in df.iterrows():
            class_id = row['class_id']
            n_samples = row['num_samples']
            n_sv = int(row['num_singular_values'])
            
            # 提取奇异值（跳过前3列）
            singular_values = []
            for i in range(n_sv):
                sv_col = f'singular_value_{i+1}'
                if sv_col in row and not pd.isna(row[sv_col]):
                    singular_values.append(float(row[sv_col]))
            
            if singular_values:
                all_singular_values.append(np.array(singular_values))
                class_ids.append(class_id)
                num_samples_per_class.append(n_samples)
        
        print(f"从CSV文件加载了 {len(all_singular_values)} 个类别的奇异值数据")
    
    else:
        raise ValueError(f"不支持的文件格式: {data_path}，仅支持.npz和.csv格式")
    
    return all_singular_values, class_ids, num_samples_per_class

def plot_singular_value_curves(all_singular_values, class_ids, num_samples_per_class, 
                             save_path=None, show_legend=True, max_dims=64, 
                             plot_type='cumulative', figsize=(3.5, 2.6)):
    """
    绘制奇异值曲线
    
    Args:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
        save_path: 保存路径
        show_legend: 是否显示图例
        max_dims: 最大显示维度
        plot_type: 绘图类型 ('cumulative', 'raw', 'normalized')
        figsize: 图像尺寸
    """
    if not all_singular_values:
        print("没有有效的奇异值数据可绘制")
        return
    
    plt.figure(figsize=figsize)
    
    # 创建颜色映射
    n_classes = len(all_singular_values)
    colors = [cm.get_cmap('viridis')(i) for i in np.linspace(0.2, 0.9, n_classes)]
    
    # 为每个类别绘制曲线
    for i, (singular_values, class_id, n_samples) in enumerate(zip(all_singular_values, class_ids, num_samples_per_class)):
        if plot_type == 'cumulative':
            # 计算累计比例
            sorted_singular_values = np.sort(singular_values)[::-1]  # 降序排列
            cumulative_sum = 0
            total_sum = np.sum(sorted_singular_values)
            cumulative_ratios = []
            
            for k in range(min(max_dims, len(sorted_singular_values))):
                cumulative_sum += sorted_singular_values[k]
                cumulative_ratio = cumulative_sum / total_sum if total_sum > 0 else 0
                cumulative_ratios.append(cumulative_ratio)
            
            dimensions = np.arange(1, len(cumulative_ratios) + 1)
            plt.plot(dimensions, cumulative_ratios,
                    color=colors[i], linewidth=0.8, alpha=0.8,
                    label=f'{class_id} (n={n_samples})')
            
            plt.ylabel('Cumulative ratio of top singular values', fontsize=12)
            
        elif plot_type == 'raw':
            # 绘制原始奇异值
            sorted_singular_values = np.sort(singular_values)[::-1]  # 降序排列
            dimensions = np.arange(1, min(max_dims, len(sorted_singular_values)) + 1)
            plt.plot(dimensions, sorted_singular_values[:max_dims],
                    color=colors[i], linewidth=0.8, alpha=0.8,
                    label=f'{class_id} (n={n_samples})')
            
            plt.ylabel('Singular value magnitude', fontsize=12)
            
        elif plot_type == 'normalized':
            # 绘制归一化奇异值
            sorted_singular_values = np.sort(singular_values)[::-1]  # 降序排列
            max_sv = sorted_singular_values[0] if len(sorted_singular_values) > 0 else 1
            normalized_values = sorted_singular_values / max_sv
            
            dimensions = np.arange(1, min(max_dims, len(normalized_values)) + 1)
            plt.plot(dimensions, normalized_values[:max_dims],
                    color=colors[i], linewidth=0.8, alpha=0.8,
                    label=f'{class_id} (n={n_samples})')
            
            plt.ylabel('Normalized singular value', fontsize=12)
    
    # 设置标签和标题
    plt.xlabel('Singular index', fontsize=12)
    
    # 设置坐标轴
    if plot_type == 'cumulative':
        plt.ylim(0, 1.05)
    else:
        plt.yscale('log')  # 对原始和归一化值使用对数尺度
    
    plt.xlim(1, max_dims)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    if show_legend and len(all_singular_values) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    elif len(all_singular_values) > 20:
        plt.text(0.98, 0.05, f'Total classes: {len(all_singular_values)}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"奇异值曲线图已保存到: {save_path}")
    
    plt.show()

def plot_singular_value_statistics(all_singular_values, class_ids, num_samples_per_class, 
                                 save_path=None, figsize=(10, 8)):
    """
    绘制奇异值统计信息
    
    Args:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
        save_path: 保存路径
        figsize: 图像尺寸
    """
    if not all_singular_values:
        print("没有有效的奇异值数据可分析")
        return
    
    # 计算统计信息
    total_singular_values = []
    max_singular_values = []
    mean_singular_values = []
    std_singular_values = []
    effective_rank_values = []
    
    for singular_values in all_singular_values:
        total_sv = len(singular_values)
        max_sv = np.max(singular_values)
        mean_sv = np.mean(singular_values)
        std_sv = np.std(singular_values)
        
        # 计算有效秩
        eigenvals_sq_sum = np.sum(singular_values**2)
        effective_rank = eigenvals_sq_sum / (max_sv**2) if max_sv > 0 else 0
        
        total_singular_values.append(total_sv)
        max_singular_values.append(max_sv)
        mean_singular_values.append(mean_sv)
        std_singular_values.append(std_sv)
        effective_rank_values.append(effective_rank)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # 1. 奇异值数量分布
    axes[0].hist(total_singular_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Number of singular values')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of singular value counts')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 最大奇异值分布
    axes[1].hist(max_singular_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Maximum singular value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of maximum singular values')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 平均奇异值分布
    axes[2].hist(mean_singular_values, bins=20, alpha=0.7, color='salmon', edgecolor='black')
    axes[2].set_xlabel('Mean singular value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of mean singular values')
    axes[2].grid(True, alpha=0.3)
    
    # 4. 有效秩分布
    axes[3].hist(effective_rank_values, bins=20, alpha=0.7, color='gold', edgecolor='black')
    axes[3].set_xlabel('Effective rank')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Distribution of effective ranks')
    axes[3].grid(True, alpha=0.3)
    
    # 5. 样本数量 vs 有效秩散点图
    axes[4].scatter(num_samples_per_class, effective_rank_values, alpha=0.6, color='purple')
    axes[4].set_xlabel('Number of samples per class')
    axes[4].set_ylabel('Effective rank')
    axes[4].set_title('Samples vs Effective Rank')
    axes[4].grid(True, alpha=0.3)
    
    # 6. 奇异值数量 vs 样本数量散点图
    axes[5].scatter(num_samples_per_class, total_singular_values, alpha=0.6, color='orange')
    axes[5].set_xlabel('Number of samples per class')
    axes[5].set_ylabel('Number of singular values')
    axes[5].set_title('Samples vs Singular Value Count')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"奇异值统计图已保存到: {save_path}")
    
    plt.show()

def print_data_summary(all_singular_values, class_ids, num_samples_per_class):
    """
    打印数据摘要信息
    
    Args:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
    """
    print("\n" + "="*60)
    print("奇异值数据摘要")
    print("="*60)
    
    print(f"类别总数: {len(all_singular_values)}")
    print(f"样本总数: {sum(num_samples_per_class)}")
    print(f"平均每类样本数: {np.mean(num_samples_per_class):.2f} ± {np.std(num_samples_per_class):.2f}")
    
    # 计算奇异值统计
    total_singular_values = [len(sv) for sv in all_singular_values]
    max_singular_values = [np.max(sv) for sv in all_singular_values]
    mean_singular_values = [np.mean(sv) for sv in all_singular_values]
    
    print(f"平均奇异值数量: {np.mean(total_singular_values):.2f} ± {np.std(total_singular_values):.2f}")
    print(f"最大奇异值范围: {np.min(max_singular_values):.4f} - {np.max(max_singular_values):.4f}")
    print(f"平均奇异值范围: {np.min(mean_singular_values):.4f} - {np.max(mean_singular_values):.4f}")
    
    # 显示前几个类别的详细信息
    print(f"\n前5个类别的详细信息:")
    for i in range(min(5, len(all_singular_values))):
        singular_values = all_singular_values[i]
        class_id = class_ids[i]
        n_samples = num_samples_per_class[i]
        
        print(f"{class_id}: n={n_samples:4d}, "
              f"dims={len(singular_values):3d}, "
              f"max_sv={np.max(singular_values):8.4f}, "
              f"mean_sv={np.mean(singular_values):6.4f}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='奇异值数据可视化')
    parser.add_argument('--data_path', type=str, required=True,
                        help='奇异值数据文件路径 (.npz或.csv格式)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='输出图像保存目录')
    parser.add_argument('--plot_type', type=str, default='cumulative',
                        choices=['cumulative', 'raw', 'normalized'],
                        help='绘图类型')
    parser.add_argument('--max_dims', type=int, default=64,
                        help='最大显示维度')
    parser.add_argument('--show_legend', action='store_true',
                        help='是否显示图例')
    parser.add_argument('--plot_stats', action='store_true',
                        help='是否绘制统计信息图')
    parser.add_argument('--figsize', type=float, nargs=2, default=[3.5, 2.6],
                        help='图像尺寸 (宽度, 高度)')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载数据从: {args.data_path}")
    all_singular_values, class_ids, num_samples_per_class = load_singular_values_data(args.data_path)
    
    # 打印数据摘要
    print_data_summary(all_singular_values, class_ids, num_samples_per_class)
    
    # 生成基础文件名
    base_name = Path(args.data_path).stem
    
    # 绘制奇异值曲线
    print(f"\n绘制{args.plot_type}类型奇异值曲线...")
    curve_save_path = os.path.join(args.output_dir, f"{base_name}_{args.plot_type}_curves.png")
    plot_singular_value_curves(
        all_singular_values, class_ids, num_samples_per_class,
        save_path=curve_save_path,
        show_legend=args.show_legend,
        max_dims=args.max_dims,
        plot_type=args.plot_type,
        figsize=tuple(args.figsize)
    )
    
    # 绘制统计信息图
    if args.plot_stats:
        print("\n绘制奇异值统计信息图...")
        stats_save_path = os.path.join(args.output_dir, f"{base_name}_statistics.png")
        plot_singular_value_statistics(
            all_singular_values, class_ids, num_samples_per_class,
            save_path=stats_save_path
        )
    
    print("\n" + "="*50)
    print("奇异值可视化完成!")
    print("="*50)

if __name__ == '__main__':
    main()