#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4: RGDA与SGD分类器参数敏感性对比及时间效率分析
基于面向过程的编程思路
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

@contextmanager
def timer():
    """计时器上下文管理器"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    return elapsed

def measure_time(func, *args, **kwargs):
    """测量函数执行时间"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def build_gaussian_statistics(features, labels, cholesky=True):
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

def evaluate_classifier_with_timing(classifier, test_features, test_labels, test_dataset_ids, 
                                  device="cuda", batch_size=256, return_class_wise=True):
    classifier.to(device)
    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    
    dataset = torch.utils.data.TensorDataset(test_features, test_labels, test_dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    # 预测时间计时
    predict_start = time.perf_counter()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(classifier_device)
            all_targets.append(batch[1])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    predict_time = time.perf_counter() - predict_start
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # 计算准确率
    if return_class_wise:
        # 计算每个类别的准确率
        unique_classes = torch.unique(all_targets)
        class_accuracies = []
        
        for class_id in unique_classes:
            mask = (all_targets == class_id)
            if mask.sum() > 0:
                class_correct = (all_predictions[mask] == all_targets[mask]).float().sum().item()
                class_total = mask.sum().item()
                class_acc = class_correct / class_total
                class_accuracies.append(class_acc)
        
        accuracy = np.mean(class_accuracies) if class_accuracies else 0.0
    else:
        # 计算所有样本的总体准确度
        total_correct = (all_predictions == all_targets).float().sum().item()
        total_samples = len(all_targets)
        accuracy = total_correct / total_samples
    
    timing_info = {
        'predict_time': predict_time,
        'predict_time_per_sample': predict_time / len(all_predictions),
        'total_samples': len(all_predictions)
    }
    
    torch.cuda.empty_cache()
    return accuracy, timing_info

def evaluate_rgda_classifier(alpha1, alpha2, alpha3, train_stats, test_features, test_labels, test_dataset_ids,
                           device="cuda", return_class_wise=True):
    """
    评估RGDA分类器并记录重构时间
    """
    # 重构时间计时
    build_start = time.perf_counter()
    
    builder = QDAClassifierBuilder(
        qda_reg_alpha1=alpha1,
        qda_reg_alpha2=alpha2,
        qda_reg_alpha3=alpha3,
        device=device)
    
    classifier = builder.build(train_stats)
    build_time = time.perf_counter() - build_start
    
    accuracy, timing_info = evaluate_classifier_with_timing(
        classifier, test_features, test_labels, test_dataset_ids,
        device=device, return_class_wise=return_class_wise)
    
    timing_info['build_time'] = build_time
    timing_info['total_time'] = build_time + timing_info['predict_time']
    
    return accuracy, timing_info

def evaluate_sgd_classifier(train_stats, test_features, test_labels, test_dataset_ids,
                          sgd_epochs=5, sgd_lr=0.01, cached_Z=None, device="cuda",
                          linear=False, alpha1=1.0, alpha2=0.0, alpha3=0.0, return_class_wise=True):
    """
    评估SGD分类器并记录重构时间
    """
    if cached_Z is None:
        cached_Z = torch.randn(40000, list(train_stats.values())[0].mean.size(0))
    
    # 重构时间计时
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
    
    accuracy, timing_info = evaluate_classifier_with_timing(
        classifier, test_features, test_labels, test_dataset_ids,
        device=device, return_class_wise=return_class_wise)
    
    timing_info['build_time'] = build_time
    timing_info['total_time'] = build_time + timing_info['predict_time']
    
    return accuracy, timing_info

def evaluate_baseline_classifiers(train_stats, test_features, test_labels, test_dataset_ids, device="cuda"):
    """
    评估基准分类器（LDA和NCM）
    """
    baseline_results = {}
    
    # 评估LDA分类器
    print("评估LDA分类器...")
    build_start = time.perf_counter()
    
    lda_builder = LDAClassifierBuilder(reg_alpha=0.3, device=device)
    lda_classifier = lda_builder.build(train_stats)
    
    lda_build_time = time.perf_counter() - build_start
    lda_accuracy, lda_timing = evaluate_classifier_with_timing(
        lda_classifier, test_features, test_labels, test_dataset_ids, device=device)
    
    baseline_results['LDA'] = {
        'accuracy': lda_accuracy,
        'build_time': lda_build_time,
        'predict_time': lda_timing['predict_time'],
        'total_time': lda_build_time + lda_timing['predict_time']
    }
    
    # 评估NCM分类器
    print("评估NCM分类器...")
    build_start = time.perf_counter()
    
    ncm_classifier = NCMClassifier(train_stats).to(device)
    
    ncm_build_time = time.perf_counter() - build_start
    ncm_accuracy, ncm_timing = evaluate_classifier_with_timing(
        ncm_classifier, test_features, test_labels, test_dataset_ids, device=device)
    
    baseline_results['NCM'] = {
        'accuracy': ncm_accuracy,
        'build_time': ncm_build_time,
        'predict_time': ncm_timing['predict_time'],
        'total_time': ncm_build_time + ncm_timing['predict_time']
    }
    
    return baseline_results

# In[]
def plot_sensitivity_comparison(results, model_name, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    lda_acc = results['baselines']['LDA']['accuracy']
    ncm_acc = results['baselines']['NCM']['accuracy']
    
    ax = axes[0, 0]
    x = results['fixed_alpha1']['alpha2']
    ax.plot(x, results['fixed_alpha1']['rgda_accuracy'], 'r-', marker='o',
            label='RGDA', linewidth=1.5, markersize=4)
    ax.plot(x, results['fixed_alpha1']['sgd_accuracy'], 'b-', marker='s',
            label='SGD', linewidth=1.5, markersize=4)
    ax.plot(x, results['fixed_alpha1']['linear_sgd_accuracy'], 'g-', marker='^',
            label='Linear SGD', linewidth=1.5, markersize=4)
    ax.axhline(y=lda_acc, color='gray', linestyle='--', alpha=0.7, label=f'LDA: {lda_acc:.3f}')
    ax.axhline(y=ncm_acc, color='orange', linestyle='--', alpha=0.7, label=f'NCM: {ncm_acc:.3f}')
    
    ax.set_xlabel('Alpha2 (Alpha1=0.2)', fontsize=10)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=10)
    ax.set_title(f'{model_name}: Fixed Alpha1', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    
    # 2. 固定alpha2，变动alpha1 - 准确率对比
    ax = axes[0, 1]
    x = results['fixed_alpha2']['alpha1']
    ax.plot(x, results['fixed_alpha2']['rgda_accuracy'], 'r-', marker='o',
            label='RGDA', linewidth=1.5, markersize=4)
    ax.plot(x, results['fixed_alpha2']['sgd_accuracy'], 'b-', marker='s',
            label='SGD', linewidth=1.5, markersize=4)
    ax.plot(x, results['fixed_alpha2']['linear_sgd_accuracy'], 'g-', marker='^',
            label='Linear SGD', linewidth=1.5, markersize=4)
    ax.axhline(y=lda_acc, color='gray', linestyle='--', alpha=0.7, label=f'LDA: {lda_acc:.3f}')
    ax.axhline(y=ncm_acc, color='orange', linestyle='--', alpha=0.7, label=f'NCM: {ncm_acc:.3f}')
    
    ax.set_xlabel('Alpha1 (Alpha2=0.2)', fontsize=10)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=10)
    ax.set_title(f'{model_name}: Fixed Alpha2', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    
    # 3. 时间对比 - 重构时间
    ax = axes[1, 0]
    classifiers = ['RGDA', 'SGD', 'Linear SGD', 'LDA', 'NCM']
    build_times = [
        np.mean([t['build_time'] for t in results['fixed_alpha1']['rgda_timing']]),
        np.mean([t['build_time'] for t in results['fixed_alpha1']['sgd_timing']]),
        np.mean([t['build_time'] for t in results['fixed_alpha1']['linear_sgd_timing']]),
        results['baselines']['LDA']['build_time'],
        results['baselines']['NCM']['build_time']
    ]
    colors = ['red', 'blue', 'green', 'gray', 'orange']
    bars = ax.bar(classifiers, build_times, color=colors, alpha=0.7)
    ax.set_ylabel('重构时间 (秒)', fontsize=10)
    ax.set_title('分类器重构时间对比', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, time_val in zip(bars, build_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(build_times)*0.01,
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. 时间对比 - 预测时间
    ax = axes[1, 1]
    predict_times = [
        np.mean([t['predict_time'] for t in results['fixed_alpha1']['rgda_timing']]),
        np.mean([t['predict_time'] for t in results['fixed_alpha1']['sgd_timing']]),
        np.mean([t['predict_time'] for t in results['fixed_alpha1']['linear_sgd_timing']]),
        results['baselines']['LDA']['predict_time'],
        results['baselines']['NCM']['predict_time']
    ]
    
    bars = ax.bar(classifiers, predict_times, color=colors, alpha=0.7)
    ax.set_ylabel('预测时间 (秒)', fontsize=10)
    ax.set_title('分类器预测时间对比', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, time_val in zip(bars, predict_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(predict_times)*0.01,
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"敏感性对比图已保存到: {save_path}")
    
    plt.show()

def save_results(results, model_name, save_dir):
    """
    保存实验结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存JSON格式结果
    json_path = os.path.join(save_dir, f"{model_name}_sensitivity_results.json")
    
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
    
    print(f"结果已保存到: {json_path}")
    return json_path

# In[]
if __name__ == '__main__':
    """
    主函数
    """
    # 使用CLIP架构
    model_name = "vit-b-p16-clip"
    num_shots = 128
    iterations = 0
    base_output_dir = "实验结果保存/分类器消融实验"
    
    # 创建输出目录

    print(f"\n{'='*60}")
    print(f"开始运行敏感性实验: {model_name}")
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

    # 实验参数
    alpha3_fixed = 0.5
    alpha2_fixed = 0.2
    alpha1_fixed = 0.2

    # 结果字典
    results = {
        'fixed_alpha1': {
            'alpha2': [],
            'rgda_accuracy': [],
            'sgd_accuracy': [],
            'linear_sgd_accuracy': [],
            'rgda_timing': [],
            'sgd_timing': [],
            'linear_sgd_timing': []
        },
        'fixed_alpha2': {
            'alpha1': [],
            'rgda_accuracy': [],
            'sgd_accuracy': [],
            'linear_sgd_accuracy': [],
            'rgda_timing': [],
            'sgd_timing': [],
            'linear_sgd_timing': []
        },
        'baselines': {}
    }

    # 收集基准分类器结果
    print("评估基准分类器...")
    baseline_results = evaluate_baseline_classifiers(train_stats, test_features, test_labels, test_dataset_ids)
    results['baselines'] = baseline_results

    # 生成缓存的随机向量用于SGD
    cached_Z = torch.randn(1024, list(train_stats.values())[0].mean.size(0))
    print(f"\n固定alpha2={alpha2_fixed}，变动alpha1...")
    alpha1_range = torch.linspace(0.8, 1.2, 3)

    for alpha1 in alpha1_range:
        print(f"  测试: alpha1={alpha1:.3f}, alpha2={alpha2_fixed:.3f}")
        
        # 评估RGDA分类器
        rgda_acc, rgda_timing = evaluate_rgda_classifier(
            alpha1, alpha2_fixed, alpha3_fixed, train_stats,
            test_features, test_labels, test_dataset_ids,
            return_class_wise=True)
        
        # 评估SGD分类器（使用与RGDA相同的正则化参数）
        sgd_acc, sgd_timing = evaluate_sgd_classifier(
            train_stats, test_features, test_labels, test_dataset_ids,
            sgd_epochs=3, sgd_lr=1e-3, cached_Z=cached_Z,
            linear=False, alpha1=alpha1.item(), alpha2=alpha2_fixed, alpha3=alpha3_fixed, return_class_wise=True)
        
        # 评估线性SGD分类器（使用与RGDA相同的正则化参数）
        linear_sgd_acc, linear_sgd_timing = evaluate_sgd_classifier(
            train_stats, test_features, test_labels, test_dataset_ids,
            sgd_epochs=3, sgd_lr=1e-3, cached_Z=cached_Z,
            linear=True, alpha1=alpha1.item(), alpha2=alpha2_fixed, alpha3=alpha3_fixed, return_class_wise=True)
        
        results['fixed_alpha2']['alpha1'].append(alpha1)
        results['fixed_alpha2']['rgda_accuracy'].append(rgda_acc)
        results['fixed_alpha2']['sgd_accuracy'].append(sgd_acc)
        results['fixed_alpha2']['linear_sgd_accuracy'].append(linear_sgd_acc)
        results['fixed_alpha2']['rgda_timing'].append(rgda_timing)
        results['fixed_alpha2']['sgd_timing'].append(sgd_timing)
        results['fixed_alpha2']['linear_sgd_timing'].append(linear_sgd_timing)
        
        print(f"RGDA准确率: {rgda_acc:.4f}, SGD准确率: {sgd_acc:.4f}, 线性SGD准确率: {linear_sgd_acc:.4f}")
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(base_output_dir, exist_ok=True)

    print(f"\n固定alpha1={alpha1_fixed}，变动alpha2...")
    alpha2_range = torch.linspace(0.0, 3.0, 11)

    for alpha2 in alpha2_range:
        print(f"  测试: alpha1={alpha1_fixed:.3f}, alpha2={alpha2:.3f}")
        
        # 评估RGDA分类器
        rgda_acc, rgda_timing = evaluate_rgda_classifier(
            alpha1_fixed, alpha2, alpha3_fixed, train_stats,
            test_features, test_labels, test_dataset_ids,
            return_class_wise=True)
        
        # 评估SGD分类器（使用与RGDA相同的正则化参数）
        sgd_acc, sgd_timing = evaluate_sgd_classifier(
            train_stats, test_features, test_labels, test_dataset_ids,
            sgd_epochs=2, sgd_lr=1e-3, cached_Z=cached_Z,
            linear=False, alpha1=alpha1_fixed, alpha2=alpha2.item(), alpha3=alpha3_fixed, return_class_wise=True)
        
        # 评估线性SGD分类器（使用与RGDA相同的正则化参数）
        linear_sgd_acc, linear_sgd_timing = evaluate_sgd_classifier(
            train_stats, test_features, test_labels, test_dataset_ids,
            sgd_epochs=2, sgd_lr=1e-3, cached_Z=cached_Z,
            linear=True, alpha1=alpha1_fixed, alpha2=alpha2.item(), alpha3=alpha3_fixed, return_class_wise=True)
        
        results['fixed_alpha1']['alpha2'].append(alpha2)
        results['fixed_alpha1']['rgda_accuracy'].append(rgda_acc)
        results['fixed_alpha1']['sgd_accuracy'].append(sgd_acc)
        results['fixed_alpha1']['linear_sgd_accuracy'].append(linear_sgd_acc)
        results['fixed_alpha1']['rgda_timing'].append(rgda_timing)
        results['fixed_alpha1']['sgd_timing'].append(sgd_timing)
        results['fixed_alpha1']['linear_sgd_timing'].append(linear_sgd_timing)
        
        print(f"    RGDA准确率: {rgda_acc:.4f}, SGD准确率: {sgd_acc:.4f}, 线性SGD准确率: {linear_sgd_acc:.4f}")

    # 保存结果
    save_results(results, model_name, base_output_dir)
    
    # 绘制敏感性对比图
    plot_path = os.path.join(base_output_dir, f"{model_name}_sensitivity_comparison.png")
    plot_sensitivity_comparison(results, model_name, save_path=plot_path)
    
    print(f"\n{'='*60}")
    print("实验4完成!")
    print(f"结果保存在: {base_output_dir}")
    print(f"敏感性对比图: {plot_path}")
    print(f"{'='*60}")
# %%
