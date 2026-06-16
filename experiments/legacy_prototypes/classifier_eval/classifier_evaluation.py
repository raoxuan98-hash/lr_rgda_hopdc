#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类器评估模块
封装SGD和QDA评估过程，输出class-wise的平均准确度
"""

import torch
import numpy as np
from typing import Dict, Tuple
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.da_classifier_builder import QDAClassifierBuilder
from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics


def evaluate_sgd_and_qda_classifiers(alpha1: float, alpha2: float, alpha3: float, 
                                   train_features: torch.Tensor, train_labels: torch.Tensor,
                                   test_features: torch.Tensor, test_labels: torch.Tensor,
                                   device: str = "cuda", batch_size: int = 512) -> Tuple[float, float]:
    """
    封装SGD和QDA评估过程，输出class-wise的平均准确度
    
    Args:
        alpha1: 第一正则化参数
        alpha2: 第二正则化参数
        alpha3: 第三正则化参数
        train_features: 训练特征 [N, D]
        train_labels: 训练标签 [N]
        test_features: 测试特征 [M, D]
        test_labels: 测试标签 [M]
        device: 计算设备
        batch_size: 批次大小
    
    Returns:
        Tuple[float, float]: (SGD的class-wise平均准确度, QDA的class-wise平均准确度)
    """
    # 构建高斯统计量
    print("构建高斯统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)
    
    # 评估SGD分类器
    print("评估SGD分类器...")
    sgd_accuracy = _evaluate_sgd_classifier(alpha1, alpha2, alpha3, train_stats, 
                                          test_features, test_labels, device, batch_size)
    
    # 评估QDA分类器
    print("评估QDA分类器...")
    qda_accuracy = _evaluate_qda_classifier(alpha1, alpha2, alpha3, train_stats, 
                                          test_features, test_labels, device, batch_size)
    
    return sgd_accuracy, qda_accuracy


def _evaluate_sgd_classifier(alpha1: float, alpha2: float, alpha3: float, 
                           train_stats: Dict[int, object], 
                           test_features: torch.Tensor, test_labels: torch.Tensor,
                           device: str, batch_size: int) -> float:
    """评估SGD分类器的class-wise平均准确度"""
    # 构建SGD分类器
    sgd_builder = SGDClassifierBuilder(device=device, max_steps=100, lr=5e-4)
    sgd_classifier = sgd_builder.build(train_stats, linear=True, 
                                     alpha1=alpha1, alpha2=alpha2, alpha3=alpha3)
    
    # 评估分类器
    return _compute_class_wise_accuracy(sgd_classifier, test_features, test_labels, device, batch_size)


def _evaluate_qda_classifier(alpha1: float, alpha2: float, alpha3: float, 
                           train_stats: Dict[int, object], 
                           test_features: torch.Tensor, test_labels: torch.Tensor,
                           device: str, batch_size: int) -> float:
    """评估QDA分类器的class-wise平均准确度"""
    # 构建QDA分类器
    qda_builder = QDAClassifierBuilder(
        qda_reg_alpha1=alpha1,
        qda_reg_alpha2=alpha2,
        qda_reg_alpha3=alpha3,
        low_rank=True,
        rank=64,
        device=device
    )
    qda_classifier = qda_builder.build(train_stats)
    
    # 评估分类器
    return _compute_class_wise_accuracy(qda_classifier, test_features, test_labels, device, batch_size)


def _compute_class_wise_accuracy(classifier, test_features: torch.Tensor, test_labels: torch.Tensor,
                                device: str, batch_size: int) -> float:
    """计算分类器的class-wise平均准确度"""
    classifier.to(device)
    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(classifier_device)
            all_targets.append(batch[1])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
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
    
    # 返回class-wise平均准确度
    accuracy = np.mean(class_accuracies) if class_accuracies else 0.0
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    return accuracy


def evaluate_with_dataset(alpha1: float, alpha2: float, alpha3: float, 
                         dataset, train_subsets, test_subsets,
                         model_name: str = "vit-b-p16", 
                         num_shots: int = 128, 
                         iterations: int = 0,
                         device: str = "cuda") -> Tuple[float, float]:
    """
    使用完整数据集评估SGD和QDA分类器
    
    Args:
        alpha1: 第一正则化参数
        alpha2: 第二正则化参数
        alpha3: 第三正则化参数
        dataset: 数据集管理器
        train_subsets: 训练数据子集
        test_subsets: 测试数据子集
        model_name: 模型名称
        num_shots: 每个数据集的样本数
        iterations: 训练迭代次数
        device: 计算设备
    
    Returns:
        Tuple[float, float]: (SGD的class-wise平均准确度, QDA的class-wise平均准确度)
    """
    from classifier_ablation.data.data_loader import create_data_loaders, create_adapt_loader
    from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels
    
    print(f"开始评估分类器性能: alpha1={alpha1}, alpha2={alpha2}, alpha3={alpha3}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    # 获取和适配模型
    print("获取和适配Vision Transformer模型...")
    vit = get_vit(vit_name=model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
    
    # 提取特征
    print("提取特征...")
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations)
    
    # 评估分类器
    sgd_accuracy, qda_accuracy = evaluate_sgd_and_qda_classifiers(
        alpha1, alpha2, alpha3, 
        train_features, train_labels,
        test_features, test_labels,
        device=device
    )
    
    print(f"SGD分类器class-wise平均准确度: {sgd_accuracy:.4f}")
    print(f"QDA分类器class-wise平均准确度: {qda_accuracy:.4f}")
    
    return sgd_accuracy, qda_accuracy


if __name__ == "__main__":
    # 示例用法
    print("这是一个分类器评估模块的示例用法")
    print("请使用 evaluate_sgd_and_qda_classifiers 或 evaluate_with_dataset 函数")