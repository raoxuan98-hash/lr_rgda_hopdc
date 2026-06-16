# SGD分类器等高线图绘制实现计划

## 概述
本文档详细描述了如何结合exp1的等高线图绘制方法和exp4的SGD分类器评估功能，创建针对SGD优化器的等高线图绘制代码。

## 实现步骤

### 1. 创建新文件 `exp4_sgd_contour.py`
这个文件将包含以下主要组件：

#### 1.1 导入必要的库
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4扩展: SGD分类器参数敏感性等高线图
基于exp1的等高线图绘制方法和exp4的SGD分类器评估功能
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import json
from contextlib import contextmanager
os.chdir('/home/raoxuan/projects/fancy_sgp_lora_vit')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import os
sys.path.append('/home/raoxuan/projects/fancy_sgp_lora_vit')

from classifier.da_classifier_builder import QDAClassifierBuilder, LDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.ncm_classifier import NCMClassifier
from compensator.gaussian_statistics import GaussianStatistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels
```

#### 1.2 复用exp4中的辅助函数
- `build_gaussian_statistics`: 构建高斯统计量
- `evaluate_classifier_with_timing`: 评估分类器并计时
- `evaluate_sgd_classifier`: 评估SGD分类器（稍作修改以支持等高线图生成）

#### 1.3 创建SGD分类器网格搜索函数
```python
def grid_search_sgd_alpha1_alpha2(train_stats, test_features, test_labels, test_dataset_ids,
                                alpha1_min=0.8, alpha1_max=1.2, alpha2_min=0.0, alpha2_max=3.0,
                                alpha1_points=9, alpha2_points=21, return_class_wise=True,
                                sgd_epochs=3, sgd_lr=1e-3, cached_Z=None, device="cuda"):
    """
    对SGD分类器进行二维网格搜索，生成性能矩阵
    
    Args:
        train_stats: 训练集高斯统计量
        test_features: 测试集特征
        test_labels: 测试集标签
        test_dataset_ids: 测试集数据集ID
        alpha1_min, alpha1_max: alpha1搜索范围
        alpha2_min, alpha2_max: alpha2搜索范围
        alpha1_points, alpha2_points: 网格点数
        return_class_wise: 是否返回类别平均准确率
        sgd_epochs: SGD训练轮数
        sgd_lr: SGD学习率
        cached_Z: 缓存的随机向量
        device: 计算设备
        
    Returns:
        alpha1_values: alpha1值数组
        alpha2_values: alpha2值数组
        accuracy_matrix: 准确率矩阵
    """
    alpha1_values = np.linspace(alpha1_min, alpha1_max, alpha1_points)
    alpha2_values = np.linspace(alpha2_min, alpha2_max, alpha2_points)
    alpha3_fixed = 0.5  # 固定alpha3值
    
    # 初始化准确率矩阵
    accuracy_matrix = np.zeros((alpha1_points, alpha2_points))
    print(f"开始SGD二维网格搜索: {alpha1_points} x {alpha2_points} = {alpha1_points * alpha2_points} 个组合")
    print(f"Alpha3固定为: {alpha3_fixed}")
    print(f"SGD参数: epochs={sgd_epochs}, lr={sgd_lr}")
    
    total_tests = alpha1_points * alpha2_points
    
    with tqdm(total=total_tests, desc="SGD测试进度") as pbar:
        for i, alpha1 in enumerate(alpha1_values):
            for j, alpha2 in enumerate(alpha2_values):
                # 评估SGD分类器
                acc, timing = evaluate_sgd_classifier(
                    train_stats, test_features, test_labels, test_dataset_ids,
                    sgd_epochs=sgd_epochs, sgd_lr=sgd_lr, cached_Z=cached_Z,
                    linear=False, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3_fixed, 
                    return_class_wise=return_class_wise, device=device)
                
                accuracy_matrix[i, j] = acc * 100  # 转换为百分比
                pbar.update(1)
    
    return alpha1_values, alpha2_values, accuracy_matrix
```

#### 1.4 创建SGD分类器等高线图绘制函数
```python
def plot_sgd_alpha1_alpha2_contour(alpha1_values, alpha2_values, accuracy_matrix, save_path=None, cmap='viridis'):
    """
    绘制SGD分类器的alpha1-alpha2等高线图
    
    Args:
        alpha1_values: alpha1值数组
        alpha2_values: alpha2值数组
        accuracy_matrix: 准确率矩阵
        save_path: 保存路径
        cmap: 颜色映射
        
    Returns:
        best_alpha1: 最佳alpha1值
        best_alpha2: 最佳alpha2值
        best_acc: 最佳准确率
    """
    plt.figure(figsize=(3.5, 2.5))  # IEEE单栏标准尺寸
    
    # 创建网格
    alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values)
    
    # 绘制等高线图
    contour = plt.contourf(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=15, cmap=cmap)
    cbar = plt.colorbar(contour, shrink=0.8)
    cbar.set_label('Average accuracy (%)', fontsize=8)
    
    # 设置colorbar刻度为5个
    vmin = np.min(accuracy_matrix)
    vmax = np.max(accuracy_matrix)
    tick_positions = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_positions)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticklabels([f'{tick:.1f}' for tick in tick_positions])
    
    # 添加等高线
    contour_lines = plt.contour(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=15, colors='black', alpha=0.4)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    
    # 找到最佳准确率及其对应的参数
    max_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_alpha1 = alpha1_values[max_idx[0]]
    best_alpha2 = alpha2_values[max_idx[1]]
    best_acc = accuracy_matrix[max_idx]
    plt.plot(best_alpha1, best_alpha2, 'r*', markersize=6, label=f'Best: ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.2f}%')

    plt.xlabel(r'$\alpha_1^{\rm SGD}$', fontsize=8)
    plt.ylabel(r'$\alpha_2^{\rm SGD}$', fontsize=8)
    plt.title('SGD Classifier Parameter Sensitivity', fontsize=10)
    
    # 设置x轴和y轴刻度
    plt.xticks(np.linspace(min(alpha1_values), max(alpha1_values), 5), fontsize=8)
    plt.yticks(np.linspace(min(alpha2_values), max(alpha2_values), 6), fontsize=8)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"SGD等高线图已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, best_alpha2, best_acc
```

#### 1.5 创建保存结果函数
```python
def save_sgd_results(alpha1_values, alpha2_values, accuracy_matrix, model_name, save_dir):
    """
    保存SGD分类器的计算结果
    
    Args:
        alpha1_values: alpha1值数组
        alpha2_values: alpha2值数组
        accuracy_matrix: 准确率矩阵
        model_name: 模型名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_sgd_results.npz")
    np.savez(save_path,
             alpha1_values=alpha1_values,
             alpha2_values=alpha2_values,
             accuracy_matrix=accuracy_matrix)
    print(f"SGD结果已保存到: {save_path}")
    return save_path
```

#### 1.6 主函数
```python
if __name__ == '__main__':
    """
    主函数：针对CLIP架构绘制SGD分类器的等高线图
    """
    # 使用CLIP架构
    model_name = "vit-b-p16-clip"
    num_shots = 128
    iterations = 0
    base_output_dir = "实验结果保存/分类器消融实验"
    
    # 创建输出目录
    model_output_dir = os.path.join(base_output_dir, f"{model_name}_sgd_contour")
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始运行SGD分类器等高线图实验: {model_name}")
    print(f"参数设置: num_shots={num_shots}, iterations={iterations}")
    print(f"{'='*60}")

    # 加载数据
    print("加载数据...")
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)

    vit = get_vit(model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)

    # 提取特征
    print("提取特征...")
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations)

    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_dataset_ids = torch.tensor(test_dataset_ids)

    # 构建高斯统计量
    print("构建高斯统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)

    # 生成缓存的随机向量用于SGD
    cached_Z = torch.randn(1024, list(train_stats.values())[0].mean.size(0))

    # SGD分类器等高线图实验
    print("\n" + "="*60)
    print("运行SGD分类器等高线图实验")
    print("="*60)
    
    # 执行二维网格搜索
    alpha1_values, alpha2_values, accuracy_matrix = grid_search_sgd_alpha1_alpha2(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_min=0.8, alpha1_max=1.2, alpha2_min=0.0, alpha2_max=3.0,
        alpha1_points=9, alpha2_points=21,
        sgd_epochs=3, sgd_lr=1e-3, cached_Z=cached_Z,
        return_class_wise=True, device="cuda")

    # 保存计算结果
    save_sgd_results(alpha1_values, alpha2_values, accuracy_matrix, model_name, model_output_dir)

    # 使用四种不同的cmap绘制等高线图
    cmaps = ["viridis", "jet", "Blues", "Spectral"]
    for cmap in cmaps:
        save_path = os.path.join(model_output_dir, f"exp4_sgd_contour_{cmap}.png")
        best_alpha1, best_alpha2, best_acc = plot_sgd_alpha1_alpha2_contour(
            alpha1_values, alpha2_values, accuracy_matrix, save_path, cmap)
        print(f"使用 {cmap} cmap 的SGD等高线图已保存，最佳参数: ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print("SGD分类器等高线图实验完成!")
    print(f"结果保存在: {model_output_dir}")
    print(f"{'='*60}")
```

### 2. 实现细节

#### 2.1 参数设置
- alpha1范围：0.8-1.2（与exp4一致）
- alpha2范围：0.0-3.0（与exp4一致）
- alpha3固定：0.5（与exp4一致）
- SGD参数：epochs=3, lr=1e-3（与exp4一致）
- 网格点数：alpha1_points=9, alpha2_points=21（与exp1一致）

#### 2.2 输出
- 保存SGD分类器的性能数据（.npz文件）
- 生成多种颜色的等高线图（viridis, jet, Blues, Spectral）
- 记录最佳参数组合和对应的准确率

#### 2.3 目录结构
```
实验结果保存/分类器消融实验/
└── vit-b-p16-clip_sgd_contour/
    ├── vit-b-p16-clip_sgd_results.npz
    ├── exp4_sgd_contour_viridis.png
    ├── exp4_sgd_contour_jet.png
    ├── exp4_sgd_contour_Blues.png
    └── exp4_sgd_contour_Spectral.png
```

## 总结
这个实现计划详细描述了如何创建一个专门用于绘制SGD分类器等高线图的实验代码。该代码将结合exp1的等高线图绘制方法和exp4的SGD分类器评估功能，专注于alpha1和alpha2的参数敏感性分析，并针对CLIP架构进行实验。