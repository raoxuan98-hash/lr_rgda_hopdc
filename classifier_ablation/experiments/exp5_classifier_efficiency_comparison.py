#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验5: SGD与RGDA分类器计算效率对比
专注于比较分类器重构时间和预测时间，不涉及准确度计算
"""
# In[]
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import json
from contextlib import contextmanager
import seaborn as sns
import pandas as pd
os.chdir('/home/raoxuan/projects/fancy_sgp_lora_vit')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
sys.path.append('/home/raoxuan/projects/fancy_sgp_lora_vit')

from classifier.da_classifier_builder import QDAClassifierBuilder, LDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.ncm_classifier import NCMClassifier
from compensator.gaussian_statistics import GaussianStatistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels

# 设置matplotlib参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Georgia']
plt.rcParams['mathtext.fontset'] = 'stix'
# 添加字体警告处理
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

def build_gaussian_statistics(features, labels, cholesky=True):
    """构建高斯统计量"""
    features = features.cpu()
    labels = labels.cpu()
    unique_labels = torch.unique(labels)
    
    stats = {}
    for lbl in tqdm(unique_labels, desc="构建高斯统计量"):
        mask = (labels == lbl)
        feats_class = features[mask]
        
        mu = feats_class.mean(0)
        if feats_class.size(0) >= 2:
            cov = torch.cov(feats_class.T) + torch.eye(feats_class.size(1)) * 1e-4
        else:
            cov = torch.eye(feats_class.size(1)) * 1e-4
            
        stats[int(lbl.item())] = GaussianStatistics(mu, cov, cholesky=cholesky)
    
    return stats

def generate_synthetic_data(num_classes, samples_per_class=50, feature_dim=768):
    """
    生成合成数据用于效率测试
    
    Args:
        num_classes: 类别数量
        samples_per_class: 每个类别的样本数
        feature_dim: 特征维度

    Returns:
        train_features: 训练特征
        train_labels: 训练标签
        train_dataset_ids: 训练数据集ID
        test_features: 测试特征
        test_labels: 测试标签
        test_dataset_ids: 测试数据集ID
    """
    print(f"生成合成数据: {num_classes}个类别，每类{samples_per_class}个样本...")

    train_features = []
    train_labels = []
    train_dataset_ids = []
    test_features = []
    test_labels = []
    test_dataset_ids = []

    for class_id in range(num_classes):
        mean = torch.randn(feature_dim) * 2
        U = torch.randn(feature_dim, min(feature_dim, 50))
        cov = torch.eye(feature_dim) + (U @ U.T) / feature_dim * 0.5

        try:
            train_samples = torch.distributions.MultivariateNormal(mean, cov).sample([samples_per_class])
            test_samples = torch.distributions.MultivariateNormal(mean, cov).sample([samples_per_class])
        except ValueError:
            train_samples = mean + torch.randn(samples_per_class, feature_dim) * 0.5
            test_samples = mean + torch.randn(samples_per_class, feature_dim) * 0.5

        train_features.append(train_samples)
        train_labels.extend([class_id] * samples_per_class)
        train_dataset_ids.extend([0] * samples_per_class)

        test_features.append(test_samples)
        test_labels.extend([class_id] * samples_per_class)
        test_dataset_ids.extend([0] * samples_per_class)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.tensor(train_labels)
    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.tensor(test_labels)
    test_dataset_ids = torch.tensor(test_dataset_ids)

    print("合成数据生成完成:")
    print(f"  训练特征: {train_features.shape}")
    print(f"  训练标签: {train_labels.shape}")
    print(f"  测试特征: {test_features.shape}")
    print(f"  测试标签: {test_labels.shape}")

    return train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids


def plot_efficiency_comparison_ieee(results, model_name, save_path=None):
    """
    绘制分类器效率对比图 - IEEE单栏论文格式
    
    Args:
        results: 包含所有效率数据的字典
        model_name: 模型名称
        save_path: 保存路径
    """
    # IEEE单栏论文图形尺寸 (约3.5英寸宽)
    fig_width = 3.5  # inches
    fig_height = 4.0  # inches
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # IEEE推荐的颜色和样式
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 对比明显的颜色
    patterns = ['', '//', '\\\\', 'xx', '++']  # 图案用于黑白打印
    
    # 1. 重构时间对比
    ax = axes[0]
    build_means = []
    build_stds = []
    
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['build_times']
        elif classifier == 'SGD':
            times = results['sgd']['build_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['build_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['build_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['build_times']
        
        if times is not None:
            build_means.append(np.mean(times))
            build_stds.append(np.std(times))
    
    # 绘制柱状图
    x_pos = np.arange(len(classifiers))
    bars = ax.bar(x_pos, build_means, color=colors, alpha=0.8, 
                  yerr=build_stds, capsize=2, width=0.6, 
                  error_kw={'linewidth': 0.8})
    
    # 添加图案（用于黑白打印）
    for i, bar in enumerate(bars):
        if patterns[i]:
            bar.set_hatch(patterns[i])
    
    ax.set_ylabel('Build Time (s)', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifiers, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 预测时间对比
    ax = axes[1]
    predict_means = []
    predict_stds = []
    
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['predict_times']
        elif classifier == 'SGD':
            times = results['sgd']['predict_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['predict_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['predict_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['predict_times']
        
        if times is not None:
            predict_means.append(np.mean(times))
            predict_stds.append(np.std(times))
    
    # 绘制柱状图
    bars = ax.bar(x_pos, predict_means, color=colors, alpha=0.8, 
                  yerr=predict_stds, capsize=2, width=0.6,
                  error_kw={'linewidth': 0.8})
    
    # 添加图案（用于黑白打印）
    for i, bar in enumerate(bars):
        if patterns[i]:
            bar.set_hatch(patterns[i])
    
    ax.set_ylabel('Inference Time (s)', fontsize=9)
    ax.set_xlabel('Classifiers', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifiers, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    if save_path:
        # 保存多种格式
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # 同时保存EPS格式（IEEE推荐）
        eps_path = save_path.replace('.png', '.eps').replace('.jpg', '.eps')
        plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight')
        print(f"效率对比图已保存到: {save_path}")
        print(f"EPS格式已保存到: {eps_path}")
    
    plt.show()
    
    return build_means, predict_means

def plot_efficiency_vs_class_sizes_ieee(all_results, model_name, save_path=None):
    """
    绘制不同类别数量下的分类器效率对比图 - IEEE单栏论文格式
    
    Args:
        all_results: 包含不同类别数量下所有效率数据的字典
        model_name: 模型名称
        save_path: 保存路径
    """
    # IEEE单栏论文图形尺寸
    fig_width = 3.5  # inches
    fig_height = 5.0  # inches
    
    # 准备数据
    class_sizes = sorted(all_results.keys())
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']  # 不同的标记形状
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # 1. 重构时间对比
    ax = axes[0]
    
    for i, classifier in enumerate(classifiers):
        build_means = []
        build_stds = []
        
        for class_size in class_sizes:
            results = all_results[class_size]
            times = None
            
            if classifier == 'RGDA' and 'rgda' in results:
                times = results['rgda']['build_times']
            elif classifier == 'SGD' and 'sgd' in results:
                times = results['sgd']['build_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                times = results['linear_sgd']['build_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                times = results['baselines']['LDA']['build_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                times = results['baselines']['NCM']['build_times']
            
            if times is not None:
                build_means.append(np.mean(times))
                build_stds.append(np.std(times))
            else:
                build_means.append(0)
                build_stds.append(0)
        
        # 绘制带误差线的折线图
        ax.errorbar(class_sizes, build_means, yerr=build_stds, 
                   label=classifier, color=colors[i], marker=markers[i],
                   markersize=4, linewidth=1.0, capsize=2, elinewidth=0.8)
    
    ax.set_ylabel('Build Time (s)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 预测时间对比
    ax = axes[1]
    
    for i, classifier in enumerate(classifiers):
        predict_means = []
        predict_stds = []
        
        for class_size in class_sizes:
            results = all_results[class_size]
            times = None
            
            if classifier == 'RGDA' and 'rgda' in results:
                times = results['rgda']['predict_times']
            elif classifier == 'SGD' and 'sgd' in results:
                times = results['sgd']['predict_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                times = results['linear_sgd']['predict_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                times = results['baselines']['LDA']['predict_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                times = results['baselines']['NCM']['predict_times']
            
            if times is not None:
                predict_means.append(np.mean(times))
                predict_stds.append(np.std(times))
            else:
                predict_means.append(0)
                predict_stds.append(0)
        
        # 绘制带误差线的折线图
        ax.errorbar(class_sizes, predict_means, yerr=predict_stds,
                   label=classifier, color=colors[i], marker=markers[i],
                   markersize=4, linewidth=1.0, capsize=2, elinewidth=0.8)
    
    ax.set_xlabel('Number of Classes', fontsize=9)
    ax.set_ylabel('Inference Time (s)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加图例（放在图形外部）
    axes[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1.2),
                  ncol=3, frameon=True, fancybox=False, shadow=False,
                  framealpha=0.8, edgecolor='black')
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    if save_path:
        # 保存多种格式
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # 同时保存EPS格式（IEEE推荐）
        eps_path = save_path.replace('.png', '.eps').replace('.jpg', '.eps')
        plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight')
        print(f"效率对比图已保存到: {save_path}")
        print(f"EPS格式已保存到: {eps_path}")
    
    plt.show()

def create_comprehensive_efficiency_table(all_results, save_path=None):
    """
    创建综合效率表格 - IEEE论文格式
    
    Args:
        all_results: 包含所有结果的字典
        save_path: 保存路径
    """
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    class_sizes = sorted(all_results.keys())
    
    # 创建表格数据
    table_data = []
    
    for class_size in class_sizes:
        results = all_results[class_size]
        
        for classifier in classifiers:
            build_times = None
            predict_times = None
            
            if classifier == 'RGDA' and 'rgda' in results:
                build_times = results['rgda']['build_times']
                predict_times = results['rgda']['predict_times']
            elif classifier == 'SGD' and 'sgd' in results:
                build_times = results['sgd']['build_times']
                predict_times = results['sgd']['predict_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                build_times = results['linear_sgd']['build_times']
                predict_times = results['linear_sgd']['predict_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                build_times = results['baselines']['LDA']['build_times']
                predict_times = results['baselines']['LDA']['predict_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                build_times = results['baselines']['NCM']['build_times']
                predict_times = results['baselines']['NCM']['predict_times']
            
            if build_times is not None and predict_times is not None:
                build_mean = np.mean(build_times)
                build_std = np.std(build_times)
                predict_mean = np.mean(predict_times)
                predict_std = np.std(predict_times)
                
                table_data.append({
                    'Classes': class_size,
                    'Classifier': classifier,
                    'Build_Time_Mean': f"{build_mean:.4f}",
                    'Build_Time_Std': f"{build_std:.4f}",
                    'Inference_Time_Mean': f"{predict_mean:.4f}",
                    'Inference_Time_Std': f"{predict_std:.4f}",
                    'Build_Time_Formatted': f"{build_mean:.4f} ± {build_std:.4f}",
                    'Inference_Time_Formatted': f"{predict_mean:.4f} ± {predict_std:.4f}"
                })
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存为CSV
    if save_path:
        csv_path = save_path.replace('.png', '.csv').replace('.jpg', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"效率表格已保存到: {csv_path}")
    
    # 打印LaTeX格式表格
    print("\nLaTeX格式表格:")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Classifier Efficiency Comparison (Time in seconds)}")
    print("\\label{tab:efficiency_comparison}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Classes & Classifier & Build Time & Inference Time \\\\")
    print("\\midrule")
    
    for class_size in class_sizes:
        first_in_class = True
        for classifier in classifiers:
            row_data = df[(df['Classes'] == class_size) & (df['Classifier'] == classifier)]
            if not row_data.empty:
                data = row_data.iloc[0]
                if first_in_class:
                    print(f"\\multirow{{5}}{{*}}{{{class_size}}} & {classifier} & {data['Build_Time_Formatted']} & {data['Inference_Time_Formatted']} \\\\")
                    first_in_class = False
                else:
                    print(f" & {classifier} & {data['Build_Time_Formatted']} & {data['Inference_Time_Formatted']} \\\\")
        print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    return df

def measure_rgda_efficiency(alpha1, alpha2, alpha3, train_stats, test_features, test_labels, test_dataset_ids,
                          device="cuda", num_runs=5):
    build_times = []
    predict_times = []
    
    for run in range(num_runs):
        # 测量重构时间
        build_start = time.perf_counter()
        
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=alpha1,
            qda_reg_alpha2=alpha2,
            qda_reg_alpha3=alpha3,
            device=device)
        
        classifier = builder.build(train_stats)
        build_time = time.perf_counter() - build_start
        build_times.append(build_time)
        
        # 测量预测时间
        classifier.to(device)
        classifier.eval()
        classifier_device = next(classifier.parameters()).device
        
        dataset = torch.utils.data.TensorDataset(test_features, test_labels, test_dataset_ids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        predict_start = time.perf_counter()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(classifier_device)
                _ = classifier(inputs)  # 不需要保存预测结果
        predict_time = time.perf_counter() - predict_start
        predict_times.append(predict_time)
        
        # 清理GPU内存
        del classifier
        torch.cuda.empty_cache()
    
    return build_times, predict_times

def measure_sgd_efficiency(train_stats, test_features, test_labels, test_dataset_ids,
                          sgd_epochs=3, sgd_lr=0.01, cached_Z=None, device="cuda",
                          linear=False, alpha1=1.0, alpha2=0.0, alpha3=0.0, num_runs=3):
    if cached_Z is None:
        cached_Z = torch.randn(40000, list(train_stats.values())[0].mean.size(0))
    
    build_times = []
    predict_times = []
    
    for run in range(num_runs):
        # 测量重构时间
        build_start = time.perf_counter()
        
        sgd_builder = SGDClassifierBuilder(
            cached_Z=cached_Z,
            device=device,
            epochs=sgd_epochs,
            lr=sgd_lr
        )
        
        classifier = sgd_builder.build(train_stats, linear=linear,
                                       alpha1=alpha1, alpha2=alpha2, alpha3=alpha3)
        build_time = time.perf_counter() - build_start
        build_times.append(build_time)
        
        # 测量预测时间
        classifier.to(device)
        classifier.eval()
        classifier_device = next(classifier.parameters()).device
        
        dataset = torch.utils.data.TensorDataset(test_features, test_labels, test_dataset_ids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        predict_start = time.perf_counter()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(classifier_device)
                _ = classifier(inputs)  # 不需要保存预测结果
        predict_time = time.perf_counter() - predict_start
        predict_times.append(predict_time)
        
        # 清理GPU内存
        del classifier
        torch.cuda.empty_cache()
    
    return build_times, predict_times

def measure_baseline_efficiency(train_stats, test_features, test_labels, test_dataset_ids, device="cuda", num_runs=5):
    """
    测量基准分类器（LDA和NCM）的计算效率
    """
    baseline_results = {}
    
    # 测量LDA分类器
    lda_build_times = []
    lda_predict_times = []
    
    for run in range(num_runs):
        # 测量重构时间
        build_start = time.perf_counter()
        lda_builder = LDAClassifierBuilder(reg_alpha=0.3, device=device)
        lda_classifier = lda_builder.build(train_stats)
        lda_build_time = time.perf_counter() - build_start
        lda_build_times.append(lda_build_time)
        
        # 测量预测时间
        lda_classifier.to(device)
        lda_classifier.eval()
        classifier_device = next(lda_classifier.parameters()).device
        
        dataset = torch.utils.data.TensorDataset(test_features, test_labels, test_dataset_ids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        predict_start = time.perf_counter()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(classifier_device)
                _ = lda_classifier(inputs)
        predict_time = time.perf_counter() - predict_start
        lda_predict_times.append(predict_time)
        
        # 清理GPU内存
        del lda_classifier
        torch.cuda.empty_cache()
    
    # 测量NCM分类器
    ncm_build_times = []
    ncm_predict_times = []
    
    for run in range(num_runs):
        # 测量重构时间
        build_start = time.perf_counter()
        ncm_classifier = NCMClassifier(train_stats).to(device)
        ncm_build_time = time.perf_counter() - build_start
        ncm_build_times.append(ncm_build_time)
        
        # 测量预测时间
        ncm_classifier.eval()
        classifier_device = next(ncm_classifier.parameters()).device
        
        dataset = torch.utils.data.TensorDataset(test_features, test_labels, test_dataset_ids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        predict_start = time.perf_counter()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(classifier_device)
                _ = ncm_classifier(inputs)
        predict_time = time.perf_counter() - predict_start
        ncm_predict_times.append(predict_time)
        
        # 清理GPU内存
        del ncm_classifier
        torch.cuda.empty_cache()
    
    baseline_results['LDA'] = {
        'build_times': lda_build_times,
        'predict_times': lda_predict_times
    }
    
    baseline_results['NCM'] = {
        'build_times': ncm_build_times,
        'predict_times': ncm_predict_times
    }
    
    return baseline_results

def plot_efficiency_comparison_ieee(results, model_name, save_path=None):
    """
    绘制分类器效率对比图 - IEEE单栏论文格式
    
    Args:
        results: 包含所有效率数据的字典
        model_name: 模型名称
        save_path: 保存路径
    """
    # IEEE单栏论文图形尺寸 (约3.5英寸宽)
    fig_width = 3.5  # inches
    fig_height = 4.0  # inches
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # IEEE推荐的颜色和样式
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 对比明显的颜色
    patterns = ['', '//', '\\\\', 'xx', '++']  # 图案用于黑白打印
    
    # 1. 重构时间对比
    ax = axes[0]
    build_means = []
    build_stds = []
    
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['build_times']
        elif classifier == 'SGD':
            times = results['sgd']['build_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['build_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['build_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['build_times']
        
        if times is not None:
            build_means.append(np.mean(times))
            build_stds.append(np.std(times))
    
    # 绘制柱状图
    x_pos = np.arange(len(classifiers))
    bars = ax.bar(x_pos, build_means, color=colors, alpha=0.8, 
                  yerr=build_stds, capsize=2, width=0.6, 
                  error_kw={'linewidth': 0.8})
    
    # 添加图案（用于黑白打印）
    for i, bar in enumerate(bars):
        if patterns[i]:
            bar.set_hatch(patterns[i])
    
    ax.set_ylabel('Build Time (s)', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifiers, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 预测时间对比
    ax = axes[1]
    predict_means = []
    predict_stds = []
    
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['predict_times']
        elif classifier == 'SGD':
            times = results['sgd']['predict_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['predict_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['predict_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['predict_times']
        
        if times is not None:
            predict_means.append(np.mean(times))
            predict_stds.append(np.std(times))
    
    # 绘制柱状图
    bars = ax.bar(x_pos, predict_means, color=colors, alpha=0.8, 
                  yerr=predict_stds, capsize=2, width=0.6,
                  error_kw={'linewidth': 0.8})
    
    # 添加图案（用于黑白打印）
    for i, bar in enumerate(bars):
        if patterns[i]:
            bar.set_hatch(patterns[i])
    
    ax.set_ylabel('Inference Time (s)', fontsize=9)
    ax.set_xlabel('Classifiers', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifiers, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    if save_path:
        # 保存多种格式
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # 同时保存EPS格式（IEEE推荐）
        eps_path = save_path.replace('.png', '.eps').replace('.jpg', '.eps')
        plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight')
        print(f"效率对比图已保存到: {save_path}")
        print(f"EPS格式已保存到: {eps_path}")
    
    plt.show()
    
    return build_means, predict_means

def plot_efficiency_vs_class_sizes_ieee(all_results, model_name, save_path=None):
    """
    绘制不同类别数量下的分类器效率对比图 - IEEE单栏论文格式
    
    Args:
        all_results: 包含不同类别数量下所有效率数据的字典
        model_name: 模型名称
        save_path: 保存路径
    """
    # IEEE单栏论文图形尺寸
    fig_width = 3.5  # inches
    fig_height = 5.0  # inches
    
    # 准备数据
    class_sizes = sorted(all_results.keys())
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']  # 不同的标记形状
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # 1. 重构时间对比
    ax = axes[0]
    
    for i, classifier in enumerate(classifiers):
        build_means = []
        build_stds = []
        
        for class_size in class_sizes:
            results = all_results[class_size]
            times = None
            
            if classifier == 'RGDA' and 'rgda' in results:
                times = results['rgda']['build_times']
            elif classifier == 'SGD' and 'sgd' in results:
                times = results['sgd']['build_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                times = results['linear_sgd']['build_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                times = results['baselines']['LDA']['build_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                times = results['baselines']['NCM']['build_times']
            
            if times is not None:
                build_means.append(np.mean(times))
                build_stds.append(np.std(times))
            else:
                build_means.append(0)
                build_stds.append(0)
        
        # 绘制带误差线的折线图
        ax.errorbar(class_sizes, build_means, yerr=build_stds, 
                   label=classifier, color=colors[i], marker=markers[i],
                   markersize=4, linewidth=1.0, capsize=2, elinewidth=0.8)
    
    ax.set_ylabel('Build Time (s)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 预测时间对比
    ax = axes[1]
    
    for i, classifier in enumerate(classifiers):
        predict_means = []
        predict_stds = []
        
        for class_size in class_sizes:
            results = all_results[class_size]
            times = None
            
            if classifier == 'RGDA' and 'rgda' in results:
                times = results['rgda']['predict_times']
            elif classifier == 'SGD' and 'sgd' in results:
                times = results['sgd']['predict_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                times = results['linear_sgd']['predict_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                times = results['baselines']['LDA']['predict_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                times = results['baselines']['NCM']['predict_times']
            
            if times is not None:
                predict_means.append(np.mean(times))
                predict_stds.append(np.std(times))
            else:
                predict_means.append(0)
                predict_stds.append(0)
        
        # 绘制带误差线的折线图
        ax.errorbar(class_sizes, predict_means, yerr=predict_stds,
                   label=classifier, color=colors[i], marker=markers[i],
                   markersize=4, linewidth=1.0, capsize=2, elinewidth=0.8)
    
    ax.set_xlabel('Number of Classes', fontsize=9)
    ax.set_ylabel('Inference Time (s)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加图例（放在图形外部）
    axes[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1.2),
                  ncol=3, frameon=True, fancybox=False, shadow=False,
                  framealpha=0.8, edgecolor='black')
    
    # 调整布局
    plt.tight_layout(pad=1.0)
    
    if save_path:
        # 保存多种格式
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # 同时保存EPS格式（IEEE推荐）
        eps_path = save_path.replace('.png', '.eps').replace('.jpg', '.eps')
        plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight')
        print(f"效率对比图已保存到: {save_path}")
        print(f"EPS格式已保存到: {eps_path}")
    
    plt.show()

def create_comprehensive_efficiency_table(all_results, save_path=None):
    """
    创建综合效率表格 - IEEE论文格式
    
    Args:
        all_results: 包含所有结果的字典
        save_path: 保存路径
    """
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    class_sizes = sorted(all_results.keys())
    
    # 创建表格数据
    table_data = []
    
    for class_size in class_sizes:
        results = all_results[class_size]
        
        for classifier in classifiers:
            build_times = None
            predict_times = None
            
            if classifier == 'RGDA' and 'rgda' in results:
                build_times = results['rgda']['build_times']
                predict_times = results['rgda']['predict_times']
            elif classifier == 'SGD' and 'sgd' in results:
                build_times = results['sgd']['build_times']
                predict_times = results['sgd']['predict_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                build_times = results['linear_sgd']['build_times']
                predict_times = results['linear_sgd']['predict_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                build_times = results['baselines']['LDA']['build_times']
                predict_times = results['baselines']['LDA']['predict_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                build_times = results['baselines']['NCM']['build_times']
                predict_times = results['baselines']['NCM']['predict_times']
            
            if build_times is not None and predict_times is not None:
                build_mean = np.mean(build_times)
                build_std = np.std(build_times)
                predict_mean = np.mean(predict_times)
                predict_std = np.std(predict_times)
                
                table_data.append({
                    'Classes': class_size,
                    'Classifier': classifier,
                    'Build_Time_Mean': f"{build_mean:.4f}",
                    'Build_Time_Std': f"{build_std:.4f}",
                    'Inference_Time_Mean': f"{predict_mean:.4f}",
                    'Inference_Time_Std': f"{predict_std:.4f}",
                    'Build_Time_Formatted': f"{build_mean:.4f} ± {build_std:.4f}",
                    'Inference_Time_Formatted': f"{predict_mean:.4f} ± {predict_std:.4f}"
                })
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存为CSV
    if save_path:
        csv_path = save_path.replace('.png', '.csv').replace('.jpg', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"效率表格已保存到: {csv_path}")
    
    # 打印LaTeX格式表格
    print("\nLaTeX格式表格:")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Classifier Efficiency Comparison (Time in seconds)}")
    print("\\label{tab:efficiency_comparison}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Classes & Classifier & Build Time & Inference Time \\\\")
    print("\\midrule")
    
    for class_size in class_sizes:
        first_in_class = True
        for classifier in classifiers:
            row_data = df[(df['Classes'] == class_size) & (df['Classifier'] == classifier)]
            if not row_data.empty:
                data = row_data.iloc[0]
                if first_in_class:
                    print(f"\\multirow{{5}}{{*}}{{{class_size}}} & {classifier} & {data['Build_Time_Formatted']} & {data['Inference_Time_Formatted']} \\\\")
                    first_in_class = False
                else:
                    print(f" & {classifier} & {data['Build_Time_Formatted']} & {data['Inference_Time_Formatted']} \\\\")
        print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    return df
def save_efficiency_results(results, model_name, save_dir):
    """
    保存效率实验结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存JSON格式结果
    json_path = os.path.join(save_dir, f"{model_name}_efficiency_results.json")
    
    # 将torch和numpy类型转换为Python基本类型以便JSON序列化
    def convert_types(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    json_results = convert_types(results)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"效率结果已保存到: {json_path}")
    return json_path

def print_efficiency_summary(results):
    """
    打印效率实验摘要
    """
    print("\n" + "="*60)
    print("分类器效率对比摘要:")
    print("="*60)
    
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    
    # 重构时间摘要
    print("\n重构时间对比:")
    print(f"{'分类器':<15} {'平均时间(秒)':<15} {'标准差(秒)':<15} {'最快(秒)':<15} {'最慢(秒)':<15}")
    print("-" * 75)
    
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['build_times']
        elif classifier == 'SGD':
            times = results['sgd']['build_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['build_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['build_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['build_times']
        
        if times is not None:
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{classifier:<15} {mean_time:<15.4f} {std_time:<15.4f} {min_time:<15.4f} {max_time:<15.4f}")
    
    # 预测时间摘要
    print("\n预测时间对比:")
    print(f"{'分类器':<15} {'平均时间(秒)':<15} {'标准差(秒)':<15} {'最快(秒)':<15} {'最慢(秒)':<15}")
    print("-" * 75)
    
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['predict_times']
        elif classifier == 'SGD':
            times = results['sgd']['predict_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['predict_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['predict_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['predict_times']
        
        if times is not None:
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            print(f"{classifier:<15} {mean_time:<15.4f} {std_time:<15.4f} {min_time:<15.4f} {max_time:<15.4f}")
    
    # 效率排名
    print("\n效率排名 (时间越短越好):")
    
    # 重构时间排名
    build_ranking = []
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['build_times']
        elif classifier == 'SGD':
            times = results['sgd']['build_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['build_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['build_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['build_times']
        
        if times is not None:
            build_ranking.append((classifier, np.mean(times)))
    
    build_ranking.sort(key=lambda x: x[1])
    print("\n重构时间排名 (从快到慢):")
    for i, (classifier, time) in enumerate(build_ranking, 1):
        print(f"  {i}. {classifier}: {time:.4f}秒")
    
    # 预测时间排名
    predict_ranking = []
    for classifier in classifiers:
        times = None
        if classifier == 'RGDA':
            times = results['rgda']['predict_times']
        elif classifier == 'SGD':
            times = results['sgd']['predict_times']
        elif classifier == 'Linear SGD':
            times = results['linear_sgd']['predict_times']
        elif classifier == 'LDA':
            times = results['baselines']['LDA']['predict_times']
        elif classifier == 'NCM':
            times = results['baselines']['NCM']['predict_times']
        
        if times is not None:
            predict_ranking.append((classifier, np.mean(times)))
    
    predict_ranking.sort(key=lambda x: x[1])
    print("\n预测时间排名 (从快到慢):")
    for i, (classifier, time) in enumerate(predict_ranking, 1):
        print(f"  {i}. {classifier}: {time:.4f}秒")
    
    print("="*60)

def plot_efficiency_vs_class_sizes(all_results, model_name, save_path=None):
    """
    绘制不同类别数量下的分类器效率对比图
    
    Args:
        all_results: 包含不同类别数量下所有效率数据的字典
        model_name: 模型名称
        save_path: 保存路径
    """
    # 准备数据
    class_sizes = sorted(all_results.keys())
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    
    # 创建DataFrame用于seaborn绘图
    build_data = []
    predict_data = []
    
    for class_size in class_sizes:
        results = all_results[class_size]
        for classifier in classifiers:
            if classifier == 'RGDA' and 'rgda' in results:
                build_times = results['rgda']['build_times']
                predict_times = results['rgda']['predict_times']
            elif classifier == 'SGD' and 'sgd' in results:
                build_times = results['sgd']['build_times']
                predict_times = results['sgd']['predict_times']
            elif classifier == 'Linear SGD' and 'linear_sgd' in results:
                build_times = results['linear_sgd']['build_times']
                predict_times = results['linear_sgd']['predict_times']
            elif classifier == 'LDA' and 'baselines' in results and 'LDA' in results['baselines']:
                build_times = results['baselines']['LDA']['build_times']
                predict_times = results['baselines']['LDA']['predict_times']
            elif classifier == 'NCM' and 'baselines' in results and 'NCM' in results['baselines']:
                build_times = results['baselines']['NCM']['build_times']
                predict_times = results['baselines']['NCM']['predict_times']
            else:
                continue
                
            # 添加数据到DataFrame
            for bt in build_times:
                build_data.append({
                    'Class Size': class_size,
                    'Classifier': classifier,
                    'Time': bt,
                    'Type': 'Build'
                })
            
            for pt in predict_times:
                predict_data.append({
                    'Class Size': class_size,
                    'Classifier': classifier,
                    'Time': pt,
                    'Type': 'Predict'
                })
    
    build_df = pd.DataFrame(build_data)
    predict_df = pd.DataFrame(predict_data)
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    
    # 1. 重构时间对比
    plt.subplot(2, 1, 1)
    ax = sns.barplot(x='Class Size', y='Time', hue='Classifier', data=build_df,
                     palette=['red', 'blue', 'green', 'gray', 'orange'], alpha=0.7)
    plt.title(f'{model_name}: 不同类别数量下的分类器重构时间对比', fontsize=14)
    plt.ylabel('重构时间 (秒)', fontsize=12)
    plt.xlabel('类别数量', fontsize=12)
    plt.legend(title='分类器', loc='upper left')
    
    # 2. 预测时间对比
    plt.subplot(2, 1, 2)
    ax = sns.barplot(x='Class Size', y='Time', hue='Classifier', data=predict_df,
                     palette=['red', 'blue', 'green', 'gray', 'orange'], alpha=0.7)
    plt.title(f'{model_name}: 不同类别数量下的分类器预测时间对比', fontsize=14)
    plt.ylabel('预测时间 (秒)', fontsize=12)
    plt.xlabel('类别数量', fontsize=12)
    plt.legend(title='分类器', loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"不同类别数量下的效率对比图已保存到: {save_path}")
    
    plt.show()
    
    return build_df, predict_df

def run_experiment_with_class_sizes(model_name, class_sizes, samples_per_class=50, iterations=0, num_runs=3):
    """
    运行不同类别数量下的分类器效率对比实验
    
    Args:
        model_name: 模型名称
        class_sizes: 类别数量列表
        samples_per_class: 每个类别的样本数
        iterations: 迭代次数
        num_runs: 每个分类器运行次数
    
    Returns:
        all_results: 包含不同类别数量下所有效率数据的字典
    """
    all_results = {}
    
    for class_size in class_sizes:
        print(f"\n{'='*60}")
        print(f"开始运行类别数量为 {class_size} 的效率对比实验: {model_name}")
        print(f"参数设置: samples_per_class={samples_per_class}, iterations={iterations}, num_runs={num_runs}")
        print(f"{'='*60}")
        
        # 生成合成数据
        train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = generate_synthetic_data(
            class_size, samples_per_class=samples_per_class, feature_dim=768)

        train_dataset_ids = torch.tensor(train_dataset_ids)
        test_dataset_ids = torch.tensor(test_dataset_ids)

        # 构建高斯统计量
        print("构建高斯统计量...")
        train_stats = build_gaussian_statistics(train_features, train_labels)

        # 实验参数
        alpha3_fixed = 0.5
        alpha2_fixed = 0.2
        alpha1_fixed = 0.2
        
        # 结果字典
        results = {
            'rgda': {},
            'sgd': {},
            'linear_sgd': {},
            'baselines': {}
        }
        
        print(f"\n测量RGDA分类器效率...")
        rgda_build_times, rgda_predict_times = measure_rgda_efficiency(
            alpha1_fixed, alpha2_fixed, alpha3_fixed, train_stats,
            test_features, test_labels, test_dataset_ids,
            device="cuda", num_runs=num_runs)
        results['rgda']['build_times'] = rgda_build_times
        results['rgda']['predict_times'] = rgda_predict_times
        print(f"RGDA - 重构时间: {np.mean(rgda_build_times):.4f}±{np.std(rgda_build_times):.4f}秒, 预测时间: {np.mean(rgda_predict_times):.4f}±{np.std(rgda_predict_times):.4f}秒")
        
        print(f"\n测量SGD分类器效率...")
        cached_Z = torch.randn(1024, list(train_stats.values())[0].mean.size(0))
        sgd_build_times, sgd_predict_times = measure_sgd_efficiency(
            train_stats, test_features, test_labels, test_dataset_ids,
            sgd_epochs=3, sgd_lr=1e-3, cached_Z=cached_Z,
            linear=False, alpha1=alpha1_fixed, alpha2=alpha2_fixed, alpha3=alpha3_fixed, num_runs=num_runs)
        
        results['sgd']['build_times'] = sgd_build_times
        results['sgd']['predict_times'] = sgd_predict_times
        print(f"SGD - 重构时间: {np.mean(sgd_build_times):.4f}±{np.std(sgd_build_times):.4f}秒, 预测时间: {np.mean(sgd_predict_times):.4f}±{np.std(sgd_predict_times):.4f}秒")
        
        print(f"\n测量线性SGD分类器效率...")
        linear_sgd_build_times, linear_sgd_predict_times = measure_sgd_efficiency(
            train_stats, test_features, test_labels, test_dataset_ids,
            sgd_epochs=3, sgd_lr=1e-3, cached_Z=cached_Z,
            linear=True, alpha1=alpha1_fixed, alpha2=alpha2_fixed, alpha3=alpha3_fixed, num_runs=num_runs)
        results['linear_sgd']['build_times'] = linear_sgd_build_times
        results['linear_sgd']['predict_times'] = linear_sgd_predict_times
        print(f"线性SGD - 重构时间: {np.mean(linear_sgd_build_times):.4f}±{np.std(linear_sgd_build_times):.4f}秒, 预测时间: {np.mean(linear_sgd_predict_times):.4f}±{np.std(linear_sgd_predict_times):.4f}秒")
        
        print(f"\n测量基准分类器效率...")
        baseline_results = measure_baseline_efficiency(train_stats, test_features, test_labels, test_dataset_ids, device="cuda", num_runs=num_runs)
        results['baselines'] = baseline_results
        print(f"LDA - 重构时间: {np.mean(baseline_results['LDA']['build_times']):.4f}±{np.std(baseline_results['LDA']['build_times']):.4f}秒, 预测时间: {np.mean(baseline_results['LDA']['predict_times']):.4f}±{np.std(baseline_results['LDA']['predict_times']):.4f}秒")
        print(f"NCM - 重构时间: {np.mean(baseline_results['NCM']['build_times']):.4f}±{np.std(baseline_results['NCM']['build_times']):.4f}秒, 预测时间: {np.mean(baseline_results['NCM']['predict_times']):.4f}±{np.std(baseline_results['NCM']['predict_times']):.4f}秒")
        
        # 保存结果
        all_results[class_size] = results
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    return all_results

# In[]
if __name__ == '__main__':
    """
    主函数：运行分类器效率对比实验
    """
    # 使用CLIP架构
    model_name = "vit-b-p16-clip"
    num_shots = 128
    iterations = 0
    base_output_dir = "实验结果保存/分类器消融实验"
    num_runs = 3  # 每个分类器运行3次以获得稳定的时间测量
    
    # 设置不同的类别数量
    class_sizes = [200, 400, 800]
    
    print(f"\n{'='*60}")
    print(f"开始运行不同类别数量的效率对比实验: {model_name}")
    print(f"参数设置: num_shots={num_shots}, iterations={iterations}, num_runs={num_runs}")
    print(f"类别数量: {class_sizes}")
    print(f"{'='*60}")

    # 运行不同类别数量下的实验
    all_results = run_experiment_with_class_sizes(
        model_name, class_sizes, samples_per_class=50,
        iterations=iterations, num_runs=num_runs
    )
    
    # 保存所有结果
    for class_size, results in all_results.items():
        save_efficiency_results(results, f"{model_name}_{class_size}classes", base_output_dir)
    
    # 绘制不同类别数量下的效率对比图
    plot_path = os.path.join(base_output_dir, f"{model_name}_efficiency_vs_class_sizes.png")
    build_df, predict_df = plot_efficiency_vs_class_sizes(all_results, model_name, save_path=plot_path)
    
    # 保存DataFrame到CSV
    build_csv_path = os.path.join(base_output_dir, f"{model_name}_build_times_vs_class_sizes.csv")
    predict_csv_path = os.path.join(base_output_dir, f"{model_name}_predict_times_vs_class_sizes.csv")
    build_df.to_csv(build_csv_path, index=False)
    predict_df.to_csv(predict_csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("实验5完成!")
    print(f"结果保存在: {base_output_dir}")
    print(f"效率对比图: {plot_path}")
    print(f"重构时间CSV: {build_csv_path}")
    print(f"预测时间CSV: {predict_csv_path}")
    print(f"{'='*60}")
# %%
