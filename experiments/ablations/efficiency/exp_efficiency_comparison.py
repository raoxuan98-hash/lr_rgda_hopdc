#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理时间效率对比实验
比较Full-rank QDA, Low-rank QDA (rank-1,8,16,32), SGD-based linear classifier, LDA的推理时间
包含：构建时间、推理时间、吞吐量 (Throughput) 的均值与标准差统计，并保存至CSV
"""
# In[]

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "4"
import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union

# 导入项目模块
from classifier.da_classifier_builder import QDAClassifierBuilder, LDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.ncm_classifier import NCMClassifier
from classifier.gaussian_classifier import LinearLDAClassifier
from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_memory_info() -> Dict[str, float]:
    """获取当前GPU显存信息"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
}

def create_class_subset(
    full_features: torch.Tensor,
    full_labels: torch.Tensor,
    num_classes: int,
    samples_per_class: int = 32,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    np.random.seed(random_seed)
    unique_classes = torch.unique(full_labels)
    
    # 按顺序选择类别
    if len(unique_classes) < num_classes:
        raise ValueError(f"数据集只有{len(unique_classes)}个类别，无法选择{num_classes}个类别")
    
    # 按顺序选择前num_classes个类别
    selected_classes = unique_classes[:num_classes].tolist()
    
    # 筛选数据
    mask = torch.tensor([label in selected_classes for label in full_labels])
    subset_features = full_features[mask]
    subset_labels = full_labels[mask]
    
    # 为每个类别限制样本数量
    final_features = []
    final_labels = []
    
    for class_id in selected_classes:
        class_mask = (subset_labels == class_id)
        class_features = subset_features[class_mask]
        
        # 随机采样指定数量的样本
        if len(class_features) > samples_per_class:
            indices = torch.randperm(len(class_features))[:samples_per_class]
            class_features = class_features[indices]
        
        final_features.append(class_features)
        final_labels.extend([class_id] * len(class_features))
    
    final_features = torch.cat(final_features, dim=0)
    final_labels = torch.tensor(final_labels)
    
    logger.info(f"创建了{num_classes}个类别的子集，共{len(final_features)}个样本")
    
    return final_features, final_labels, selected_classes

def measure_inference_time(
    classifier: nn.Module,
    test_features: torch.Tensor,
    batch_size: int = 64,
    warmup_runs: int = 5,
    measure_runs: int = 20,
    device: str = "cuda"
) -> Tuple[float, float, float]:
    """
    测量分类器推理时间
    Returns:
        avg_time_per_sample: ms
        avg_time_per_batch: ms
        throughput: samples/s
    """
    classifier.eval()
    
    # 安全地设置设备和移动数据
    try:
        classifier.to(device)
        test_features = test_features.to(device)
    except Exception as e:
        logger.warning(f"设备设置失败，回退到CPU: {e}")
        device = "cpu"
        classifier.to(device)
        test_features = test_features.to(device)
    
    num_samples = len(test_features)
    
    # 预热运行
    with torch.no_grad():
        for _ in range(warmup_runs):
            start_idx = 0
            while start_idx < num_samples:
                end_idx = min(start_idx + batch_size, num_samples)
                batch = test_features[start_idx:end_idx]
                _ = classifier(batch)
                start_idx = end_idx
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
            except Exception:
                pass
    
    # 测量运行
    total_time = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for _ in range(measure_runs):
            start_time = time.time()
            
            start_idx = 0
            while start_idx < num_samples:
                end_idx = min(start_idx + batch_size, num_samples)
                batch = test_features[start_idx:end_idx]
                _ = classifier(batch)
                start_idx = end_idx
            
            # 安全地同步CUDA
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA同步失败，继续执行: {e}")
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_batches += 1
    
    # 计算指标
    avg_time_per_batch = (total_time / total_batches) * 1000  # 毫秒
    avg_time_per_sample = (total_time / total_batches) * 1000 / batch_size  # 毫秒/样本
    throughput = (num_samples * total_batches) / total_time  # 样本/秒
    
    return avg_time_per_sample, avg_time_per_batch, throughput

# In[]
def measure_backbone_inference_time(
    vit: nn.Module,
    data_loader,
    warmup_runs: int = 5,
    measure_runs: int = 20,
    device: str = "cuda"
) -> Tuple[float, float, float]:
    """
    测量ViT主干的推理时间
    Returns:
        avg_time_per_sample: ms
        avg_time_per_batch: ms
        throughput: samples/s
    """
    vit.eval()
    
    # 安全地设置设备
    try:
        vit.to(device)
    except Exception as e:
        logger.warning(f"设备设置失败，回退到CPU: {e}")
        device = "cpu"
        vit.to(device)
    
    # 收集所有样本用于测量
    all_samples = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            all_samples.append(inputs)
            if len(all_samples) * inputs.size(0) >= 1000:  # 最多使用1000个样本
                break
    
    test_samples = torch.cat(all_samples, dim=0)[:1000]
    num_samples = len(test_samples)
    batch_size = test_samples.size(0) // measure_runs
    if batch_size == 0:
        batch_size = 1
    
    # 预热运行
    with torch.no_grad():
        for _ in range(warmup_runs):
            start_idx = 0
            while start_idx < num_samples:
                end_idx = min(start_idx + batch_size, num_samples)
                batch = test_samples[start_idx:end_idx]
                _ = vit(batch)
                start_idx = end_idx
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
            except Exception:
                pass
    
    # 测量运行
    total_time = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for _ in range(measure_runs):
            start_time = time.time()
            
            start_idx = 0
            while start_idx < num_samples:
                end_idx = min(start_idx + batch_size, num_samples)
                batch = test_samples[start_idx:end_idx]
                _ = vit(batch)
                start_idx = end_idx
            
            # 安全地同步CUDA
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA同步失败，继续执行: {e}")
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_batches += 1
    
    # 计算指标
    avg_time_per_batch = (total_time / total_batches) * 1000  # 毫秒
    avg_time_per_sample = (total_time / total_batches) * 1000 / batch_size  # 毫秒/样本
    throughput = (num_samples * total_batches) / total_time  # 样本/秒
    
    return avg_time_per_sample, avg_time_per_batch, throughput

def initialize_results_dict(classifier_types: List[str]) -> Dict:
    """初始化结果字典，包含推理、构建时间、吞吐量的均值与标准差"""
    # 分类器相关指标
    classifier_metrics = [
        "inference_time_mean", "inference_time_std",
        "build_time_mean", "build_time_std",
        "throughput_mean", "throughput_std"
    ]
    
    results = {
        metric: {ct: [] for ct in classifier_types} for metric in classifier_metrics
    }
    
    # 添加 backbone 相关指标（backbone 是共享的，不按分类器类型组织）
    results["backbone_inference_time_mean"] = None
    results["backbone_inference_time_std"] = None
    results["backbone_throughput_mean"] = None
    results["backbone_throughput_std"] = None
    
    return results

def load_and_prepare_data(num_shots: int, model_name: str):
    """加载数据并创建数据加载器"""
    logger.info("加载数据...")
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    
    logger.info("创建数据加载器...")
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    return dataset, train_loader, test_loader

def setup_and_adapt_model(dataset, train_subsets, model_name: str):
    """获取和适配Vision Transformer模型"""
    logger.info("获取和适配Vision Transformer模型...")
    vit = get_vit(vit_name=model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=0)
    return vit

def extract_features(vit, dataset, train_loader, test_loader, model_name: str, num_shots: int):
    """提取特征"""
    logger.info("提取特征...")
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=0)
    return train_features, train_labels, test_features

def prepare_experiment_data(train_features, train_labels, test_features):
    """准备实验数据"""
    logger.info("构建高斯统计量...")
    full_stats = build_gaussian_statistics(train_features, train_labels)
    
    # 准备测试数据（用于推理时间测量）
    test_subset = test_features[:1000]  # 使用前1000个样本
    
    return full_stats, test_subset

def build_single_classifier(classifier_type: str, subset_stats, device: str) -> nn.Module:
    """辅助函数：根据类型构建分类器"""
    if classifier_type == "full_qda":
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=0.2, qda_reg_alpha2=0.2, qda_reg_alpha3=0.2,
            low_rank=False, device=device)
        return builder.build(subset_stats)
        
    elif classifier_type.startswith("low_qda_r"):
        rank = int(classifier_type.split("_r")[1])
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=0.2, qda_reg_alpha2=0.2, qda_reg_alpha3=0.2,
            low_rank=True, rank=rank, device=device
        )
        return builder.build(subset_stats)
        
    elif classifier_type == "sgd_linear":
        builder = SGDClassifierBuilder(device=device, max_steps=5000, lr=1e-3)
        return builder.build(subset_stats, linear=True, alpha1=0.5, alpha2=0.5, alpha3=0.5)
        
    elif classifier_type == "lda":
        builder = LDAClassifierBuilder(reg_alpha=0.3, device=device)
        return builder.build(subset_stats)
        
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")

def run_experiment_for_class_count(
    num_classes: int,
    classifier_types: List[str],
    num_repeats: int,
    train_features,
    train_labels,
    num_shots: int,
    test_subset,
    device: str,
    results: Dict
):
    """为特定类别数量运行实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试类别数量: {num_classes}")
    logger.info(f"{'='*60}")
    
    # 创建类别子集
    subset_features, subset_labels, selected_classes = create_class_subset(
        train_features, train_labels, num_classes, samples_per_class=num_shots
    )
    
    # 构建子集统计量
    subset_stats = build_gaussian_statistics(subset_features, subset_labels)
    
    # 为每个分类器类型进行实验
    for classifier_type in classifier_types:
        logger.info(f"测试分类器: {classifier_type}")
        
        current_build_times = []
        current_inference_times = []
        current_throughputs = [] 

        # 重复实验以获取平均值和标准差
        for i in range(num_repeats):
            # 1. 测量构建时间
            try:
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
                t_build_start = time.time()
                
                classifier = build_single_classifier(classifier_type, subset_stats, device)
                
                if torch.cuda.is_available() and "cuda" in device:
                    torch.cuda.synchronize()
                t_build_end = time.time()
                
                build_time_ms = (t_build_end - t_build_start) * 1000
                current_build_times.append(build_time_ms)
                
            except Exception as e:
                logger.error(f"构建失败: {e}")
                current_build_times.append(float('inf'))
                continue

            # 2. 测量推理时间和吞吐量
            inference_time, _, throughput = measure_inference_time(
                classifier, test_subset, device=device)
            current_inference_times.append(inference_time)
            current_throughputs.append(throughput)

        # 统计分析 - 构建时间
        valid_build = [t for t in current_build_times if t != float('inf')]
        if valid_build:
            b_mean = np.mean(valid_build)
            b_std = np.std(valid_build)
        else:
            b_mean, b_std = float('inf'), 0.0

        # 统计分析 - 推理时间
        valid_inf = [t for t in current_inference_times if t != float('inf')]
        if valid_inf:
            i_mean = np.mean(valid_inf)
            i_std = np.std(valid_inf)
        else:
            i_mean, i_std = float('inf'), 0.0
            
        # 统计分析 - 吞吐量
        valid_thr = [t for t in current_throughputs if t != 0]
        if valid_thr:
            t_mean = np.mean(valid_thr)
            t_std = np.std(valid_thr)
        else:
            t_mean, t_std = 0.0, 0.0

        # 【修改点 2】记录结果到字典中
        results["build_time_mean"][classifier_type].append(float(b_mean))
        results["build_time_std"][classifier_type].append(float(b_std))
        results["inference_time_mean"][classifier_type].append(float(i_mean))
        results["inference_time_std"][classifier_type].append(float(i_std))
        results["throughput_mean"][classifier_type].append(float(t_mean)) 
        results["throughput_std"][classifier_type].append(float(t_std))   

        logger.info(f"  -> 构建时间: {b_mean:.2f}ms ± {b_std:.2f}")
        logger.info(f"  -> 推理时间: {i_mean:.4f}ms ± {i_std:.4f}")
        logger.info(f"  -> 吞 吐 量: {t_mean:.2f} samp/s ± {t_std:.2f}")

def run_efficiency_experiment(
    class_counts: List[int] = [50, 100],
    classifier_types: List[str] = ["full_qda", "low_qda_r1", "sgd_linear", "lda"],
    num_repeats: int = 3,
    model_name: str = "vit-b-p16-clip",
    num_shots: int = 128,
    device: str = "cuda"
) -> Dict:
    
    results = initialize_results_dict(classifier_types)
    dataset, train_loader, test_loader = load_and_prepare_data(num_shots, model_name)
    
    vit = setup_and_adapt_model(dataset, train_loader, model_name)
    
    # 测量 ViT 主干的推理效率（在提取特征之前测量，使用原始图像数据）
    logger.info("\n" + "="*60)
    logger.info("测量 ViT 主干的推理效率...")
    logger.info("="*60)
    backbone_inference_times = []
    backbone_throughputs = []
    
    for i in range(num_repeats):
        backbone_time, _, backbone_throughput = measure_backbone_inference_time(
            vit, test_loader, device=device)
        backbone_inference_times.append(backbone_time)
        backbone_throughputs.append(backbone_throughput)
    
    # 统计 backbone 推理效率
    results["backbone_inference_time_mean"] = float(np.mean(backbone_inference_times))
    results["backbone_inference_time_std"] = float(np.std(backbone_inference_times))
    results["backbone_throughput_mean"] = float(np.mean(backbone_throughputs))
    results["backbone_throughput_std"] = float(np.std(backbone_throughputs))
    
    logger.info(f"  -> Backbone 推理时间: {results['backbone_inference_time_mean']:.4f}ms ± {results['backbone_inference_time_std']:.4f}")
    logger.info(f"  -> Backbone 吐 吐 量: {results['backbone_throughput_mean']:.2f} samp/s ± {results['backbone_throughput_std']:.2f}")
    
    # 提取特征
    train_features, train_labels, test_features = extract_features(
        vit, dataset, train_loader, test_loader, model_name, num_shots
    )
    
    # 准备实验数据
    full_stats, test_subset = prepare_experiment_data(train_features, train_labels, test_features)
    
    # 主实验循环
    for num_classes in class_counts:
        run_experiment_for_class_count(
            num_classes, classifier_types, num_repeats,
            train_features, train_labels, num_shots, test_subset, device, results)
    
    return results

def plot_inference_time_comparison(
    class_counts: List[int],
    inference_results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
):
    """绘制优化的推理时间对比图"""
    # 尝试获取 inference_time_mean，如果没有则回退到旧键名 inference_time
    inference_means = inference_results.get("inference_time_mean", inference_results.get("inference_time", {}))

    # IEEE单栏图片标准尺寸 (3.5英寸宽)
    plt.figure(figsize=(3.5, 2.6))
    
    low_qda_colors = {
        'low_qda_r1': "#33b4e6",   # 浅蓝
        'low_qda_r8': "#137ab5",   # 中浅蓝
        'low_qda_r32': "#0b3a8c",  # 深蓝
    }
    
    styles = {
        'full_qda': {'color': '#e41a1c', 'linestyle': '-', 'marker': 'o', 'linewidth': 1.8},
        'sgd_linear': {'color': '#a65628', 'linestyle': '--', 'marker': 'p', 'linewidth': 1.6},
        'lda': {'color': '#f781bf', 'linestyle': '-.', 'marker': '*', 'linewidth': 1.6},
    }
    for i, (k, v) in enumerate(low_qda_colors.items()):
        styles[k] = {
            'color': v,
            'linestyle': ['-', '--', '-.'][i],
            'marker': ['s', '^', 'D'][i],
            'linewidth': 1.6
        }
    
    labels = {
        'full_qda': 'RGDA',
        'low_qda_r1': 'LR-RGDA ($r=1$)',
        'low_qda_r8': 'LR-RGDA ($r=8$)',
        'low_qda_r32': 'LR-RGDA ($r=32$)',
        'sgd_linear': 'SGD Linear',
        'lda': 'LDA'
    }
    
    methods_to_plot = set(styles.keys())
    
    valid_data = {}
    for classifier_type, times in inference_means.items():
        if classifier_type in methods_to_plot:
            valid_mask = [t != float('inf') for t in times]
            valid_counts = [c for c, m in zip(class_counts, valid_mask) if m]
            valid_times = [t for t, m in zip(times, valid_mask) if m]
            if valid_times:
                valid_data[classifier_type] = (valid_counts, valid_times)
    
    for classifier_type, (valid_counts, valid_times) in valid_data.items():
        style = styles[classifier_type]
        plt.plot(valid_counts, valid_times,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                label=labels[classifier_type],
                linewidth=style['linewidth'],
                markersize=4,
                markeredgewidth=0.5)
    
    plt.xlim(min(class_counts) - 20, max(class_counts) + 20)
    plt.ylim(0.005, 2.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of Classes', fontsize=7, labelpad=2)
    plt.ylabel('Inference Time (ms per sample)', fontsize=7, labelpad=2)
    plt.grid(True, which='major', linestyle='-', alpha=0.25, linewidth=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.15, linewidth=0.3)
    
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor='#e0e0e0',
        handletextpad=0.4,
        borderpad=0.3
    )
    
    plt.xticks([100, 200, 400, 800], fontsize=7)
    plt.yticks(fontsize=7)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.72)
    
    if save_path:
        plt.savefig(
            save_path + "/efficiency_comparison_plot.png",
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.02,
        )
        logger.info(f"优化的推理时间对比图已保存到: {save_path}")
    
    plt.show()

def save_results(results: Dict, class_counts: List[int],
                model_name: str, output_dir: str = "实验结果保存"):
    """保存实验结果到CSV文件，包含构建时间、推理时间、吞吐量及其标准差"""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"full_efficiency_results_{model_name}.csv")
    
    with open(csv_path, 'w') as f:
        # 写入表头，包含 backbone 相关的指标
        f.write("classifier_type,class_count,inference_time_mean,inference_time_std,build_time_mean,build_time_std,throughput_mean,throughput_std\n")
        
        # 写入 backbone 结果（使用 "backbone" 作为 classifier_type）
        if results["backbone_inference_time_mean"] is not None:
            f.write(f"backbone,N/A,{results['backbone_inference_time_mean']},{results['backbone_inference_time_std']},N/A,N/A,{results['backbone_throughput_mean']},{results['backbone_throughput_std']}\n")
        
        # 获取所有可用的分类器类型
        classifier_types = list(results["inference_time_mean"].keys())
        
        for classifier_type in classifier_types:
            for i, class_count in enumerate(class_counts):
                inf_mean = results["inference_time_mean"][classifier_type][i]
                inf_std = results["inference_time_std"][classifier_type][i]
                build_mean = results["build_time_mean"][classifier_type][i]
                build_std = results["build_time_std"][classifier_type][i]
                # 获取吞吐量数据
                thr_mean = results["throughput_mean"][classifier_type][i]
                thr_std = results["throughput_std"][classifier_type][i]
                
                # 写入所有列
                f.write(f"{classifier_type},{class_count},{inf_mean},{inf_std},{build_mean},{build_std},{thr_mean},{thr_std}\n")
    
    logger.info(f"详细结果已保存到: {csv_path}")
    return csv_path

def load_results(model_name: str, output_dir: str = "实验结果保存") -> Tuple[Optional[Dict], Optional[List[int]]]:
    """从CSV文件加载实验结果，包含新增的吞吐量字段和 backbone 相关指标"""
    csv_path = os.path.join(output_dir, f"full_efficiency_results_{model_name}.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"结果文件不存在: {csv_path}")
        return None, None
    
    # 初始化包含吞吐量的字典结构
    metrics = [
        "inference_time_mean", "inference_time_std",
        "build_time_mean", "build_time_std",
        "throughput_mean", "throughput_std"
    ]
    results = {m: {} for m in metrics}
    class_counts_set = set()
    
    # 初始化 backbone 指标
    results["backbone_inference_time_mean"] = None
    results["backbone_inference_time_std"] = None
    results["backbone_throughput_mean"] = None
    results["backbone_throughput_std"] = None
    
    temp_data = {}

    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines[1:]: # 跳过表头
            parts = line.strip().split(',')
            # 检查是否有足够的列（现在应该是8列）
            if len(parts) < 8:
                continue
            
            # 解析所有字段
            c_type, c_count, i_mean, i_std, b_mean, b_std, t_mean, t_std = parts
            
            # 处理 backbone 结果
            if c_type == "backbone":
                results["backbone_inference_time_mean"] = float(i_mean)
                results["backbone_inference_time_std"] = float(i_std)
                results["backbone_throughput_mean"] = float(t_mean)
                results["backbone_throughput_std"] = float(t_std)
                continue
            
            # 处理分类器结果
            c_count = int(c_count)
            class_counts_set.add(c_count)
            
            if c_type not in temp_data:
                temp_data[c_type] = {}
            
            temp_data[c_type][c_count] = {
                "inference_time_mean": float(i_mean),
                "inference_time_std": float(i_std),
                "build_time_mean": float(b_mean),
                "build_time_std": float(b_std),
                "throughput_mean": float(t_mean),
                "throughput_std": float(t_std)
            }
    
    class_counts = sorted(list(class_counts_set))
    
    # 重组数据
    for c_type in temp_data:
        for metric in metrics:
            results[metric][c_type] = []
            for cc in class_counts:
                val = temp_data[c_type].get(cc, {}).get(metric, 0.0)
                results[metric][c_type].append(val)
                
    logger.info(f"已从 {csv_path} 加载结果")
    if results["backbone_inference_time_mean"] is not None:
        logger.info(f"  -> Backbone 推理时间: {results['backbone_inference_time_mean']:.4f}ms ± {results['backbone_inference_time_std']:.4f}")
        logger.info(f"  -> Backbone 吐 吐 量: {results['backbone_throughput_mean']:.2f} samp/s ± {results['backbone_throughput_std']:.2f}")
    return results, class_counts

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='推理时间效率对比实验')
    
    # 实验参数
    parser.add_argument('--class_counts', type=int, nargs='+', default=[50, 100, 200, 400, 800],
                        help='测试的类别数量列表')
    parser.add_argument('--classifier_types', type=str, nargs='+',
                        default=["full_qda", "low_qda_r1", "low_qda_r4", "low_qda_r16", "low_qda_r64", "low_qda_r128", "sgd_linear", "lda"],
                        help='测试的分类器类型列表')
    parser.add_argument('--num_repeats', type=int, default=3,
                        help='每个实验的重复次数')
    parser.add_argument('--model_name', type=str, default="vit-b-p16-clip",
                        help='使用的模型名称')
    parser.add_argument('--num_shots', type=int, default=128,
                        help='每个类别的样本数量')
    parser.add_argument('--device', type=str, default="cuda",
                        help='计算设备')
    
    # 输出控制
    parser.add_argument('--output_dir', type=str, default="实验结果保存",
                        help='结果保存目录')
    parser.add_argument('--save_plot', type=str, default="实验结果保存/效率对比实验",
                        help='图片保存路径（可选）')
    
    # 功能控制
    parser.add_argument('--load_only',type=bool, default=False,
                        help='仅加载已有结果并绘图，不运行新实验')
    parser.add_argument('--plot_only', type=bool, default=False,
                        help='仅绘图，需要已有结果')
    
    return parser.parse_args()

def set_global_variables(results: Dict[str, Dict[str, List[float]]], class_counts: List[int]):
    """设置全局变量"""
    global inference_results, inference_class_counts
    inference_results = results
    inference_class_counts = class_counts
    logger.info("已设置全局变量")

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.load_only or args.plot_only:
        results, class_counts = load_results(args.model_name, args.output_dir)
        if results is None or class_counts is None:
            logger.error("无法加载结果")
            exit(1)
        
        plot_inference_time_comparison(class_counts, results, save_path=args.save_plot)
        exit(0)
    
    logger.info("开始运行推理与构建时间效率实验...")
    results = run_efficiency_experiment(
        class_counts=args.class_counts,
        classifier_types=args.classifier_types,
        num_repeats=args.num_repeats,
        model_name=args.model_name,
        num_shots=args.num_shots,
        device=args.device
    )
    
    save_results(results, args.class_counts, args.model_name, args.output_dir)
    plot_inference_time_comparison(args.class_counts, results, save_path=args.save_plot)
    logger.info("实验完成！")