#!/usr/bin/env python3
"""
QDA参数网格搜索优化脚本

替代原有的贝叶斯优化方法，使用网格搜索优化QDA分类器参数。

优化策略：
- 固定alpha3 = 0.1
- alpha2 = 0.9 - alpha1  
- alpha1从0.0到0.9，步长0.05（共19个点）
- 绘制性能曲线：横轴为alpha1，纵轴为准确度

使用方法：
python scripts/qda_grid_search_optimize.py --dataset imagenet-r --init_cls 20
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入项目相关模块
from classifier.da_classifier_builder import QDAClassifierBuilder
from trainer import train_single_run, build_log_dirs, _import_default_args


def setup_logging(output_dir: str) -> None:
    """设置日志配置"""
    log_file = os.path.join(output_dir, "qda_optimization.log")
    
    # 清除现有的日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def prepare_args(template: dict, dataset: str, init_cls: int, seed: int) -> dict:
    """准备训练参数"""
    args = copy.deepcopy(dict(template))
    args.update({
        "dataset": "cross_domain_elevater",
        "init_cls": init_cls,
        "increment": 0,
        "seed": seed,
        "seed_list": [seed],
        "run_id": 0,
        "cross_domain": True,
        "cross_domain_datasets": [dataset],
        "iterations": 4,
        "vit_type": "vit-b-p16",
    })
    return args


def extract_features_and_labels(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """提取测试集的特征和标签"""
    model.network.eval()
    features = []
    targets = []
    device = model._device
    
    with torch.no_grad():
        for batch in model.test_loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            feats = model.network.forward_features(inputs).cpu()
            features.append(feats)
            targets.append(labels.cpu())
    
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def get_gaussian_statistics_and_features(dataset: str, init_cls: int, seed: int = 1993) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """
    获取预训练模型的高斯统计和测试特征
    
    Args:
        dataset: 数据集名称
        init_cls: 初始类别数
        seed: 随机种子
    
    Returns:
        reference_stats: 参考高斯统计
        features: 测试特征
        targets: 测试标签
    """
    logging.info(f"开始处理数据集: {dataset}")
    
    # 准备训练参数
    base_args = _import_default_args()
    args = prepare_args(base_args, dataset, init_cls, seed)
    
    # 设置日志目录
    _, log_dir = build_log_dirs(args)
    args["log_path"] = log_dir
    
    # 训练模型
    logging.info("开始训练模型...")
    results, model = train_single_run(args, return_model=True)
    logging.info(f"训练完成，结果: {results}")
    
    # 确保model是SubspaceLoRA实例
    if isinstance(model, str):
        raise ValueError("Expected SubspaceLoRA model instance, got string")
    
    # 获取高斯统计
    variants = model.drift_compensator.variants
    selected_variant = None
    reference_stats = None
    
    for variant, stats in variants.items():
        if stats:
            selected_variant = variant
            reference_stats = stats
            break
    
    if reference_stats is None:
        raise RuntimeError("No valid Gaussian statistics found in variants.")
    
    logging.info(f"使用 {selected_variant} 统计进行优化")
    
    # 提取测试特征和标签
    logging.info("提取测试特征和标签...")
    features, targets = extract_features_and_labels(model)
    
    logging.info(f"特征形状: {features.shape}, 标签形状: {targets.shape}")
    
    # 释放模型资源
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return reference_stats, features, targets


def evaluate_qda_classifier(
    alpha1: float,
    reference_stats: Dict,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu"
) -> float:
    """
    评估给定alpha1值的QDA分类器性能
    
    Args:
        alpha1: QDA的alpha1参数
        reference_stats: 参考高斯统计
        features: 测试特征
        targets: 测试标签
        device: 评估设备
    
    Returns:
        准确度百分比
    """
    alpha2 = 0.9 - alpha1  # 根据需求设置
    alpha3 = 0.1  # 固定值
    
    # 构建QDA分类器
    builder = QDAClassifierBuilder(
        qda_reg_alpha1=alpha1,
        qda_reg_alpha2=alpha2,
        qda_reg_alpha3=alpha3,
        device=device,
    )
    
    classifier = builder.build(reference_stats)
    
    # 评估分类器
    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    
    if features.device != classifier_device:
        features_eval = features.to(classifier_device)
    else:
        features_eval = features
    
    with torch.no_grad():
        logits = classifier(features_eval)
        preds = logits.argmax(dim=1).cpu()
    
    accuracy = (preds == targets).float().mean().item() * 100.0
    
    # 释放资源
    del classifier
    if torch.cuda.is_available() and classifier_device.type != "cpu":
        torch.cuda.empty_cache()
    
    return float(round(accuracy, 4))


def grid_search_qda_params(
    reference_stats: Dict,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu",
    step_size: float = 0.05
) -> Dict[str, Any]:
    """
    执行网格搜索优化QDA参数
    
    Args:
        reference_stats: 参考高斯统计
        features: 测试特征
        targets: 测试标签
        device: 评估设备
        step_size: alpha1的步长
    
    Returns:
        包含所有结果和最佳参数的字典
    """
    alpha1_values = np.arange(0.0, 0.95, step_size)  # [0.0, 0.05, ..., 0.9]
    alpha2_values = 0.9 - alpha1_values
    accuracies = []
    
    logging.info(f"开始QDA网格搜索，共{len(alpha1_values)}个点")
    
    for i, alpha1 in enumerate(alpha1_values):
        alpha2 = alpha2_values[i]
        accuracy = evaluate_qda_classifier(
            alpha1, reference_stats, features, targets, device
        )
        accuracies.append(accuracy)
        
        logging.info(f"alpha1={alpha1:.2f}, alpha2={alpha2:.2f}, accuracy={accuracy:.2f}%")
    
    # 找到最佳参数
    best_idx = np.argmax(accuracies)
    best_alpha1 = alpha1_values[best_idx]
    best_alpha2 = alpha2_values[best_idx]
    best_accuracy = accuracies[best_idx]
    
    results = {
        "alpha1_values": alpha1_values.tolist(),
        "alpha2_values": alpha2_values.tolist(),
        "accuracies": accuracies,
        "best_alpha1": float(best_alpha1),
        "best_alpha2": float(best_alpha2),
        "best_accuracy": float(best_accuracy),
        "step_size": step_size
    }
    
    logging.info(f"最佳参数: alpha1={best_alpha1:.2f}, alpha2={best_alpha2:.2f}, accuracy={best_accuracy:.2f}%")
    
    return results


def plot_performance_curve(
    results: Dict[str, Any],
    dataset_name: str,
    output_dir: str
) -> str:
    """
    绘制QDA参数性能曲线
    
    Args:
        results: 网格搜索结果
        dataset_name: 数据集名称
        output_dir: 输出目录
    
    Returns:
        图片保存路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    alpha1_values = results["alpha1_values"]
    accuracies = results["accuracies"]
    best_alpha1 = results["best_alpha1"]
    best_accuracy = results["best_accuracy"]
    
    plt.figure(figsize=(12, 8))
    plt.plot(alpha1_values, accuracies, 'bo-', linewidth=2, markersize=8, label='准确度')
    
    # 标记最佳点
    plt.plot(best_alpha1, best_accuracy, 'r*', markersize=20, 
             label=f'最佳点: α1={best_alpha1:.2f}, 准确度={best_accuracy:.2f}%')
    
    plt.xlabel('Alpha1', fontsize=14)
    plt.ylabel('准确度 (%)', fontsize=14)
    plt.title(f'QDA参数性能曲线 - {dataset_name}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加alpha2的标注
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(alpha1_values[::2])  # 每隔一个点显示
    ax2.set_xticklabels([f'{0.9-x:.2f}' for x in alpha1_values[::2]])
    ax2.set_xlabel('Alpha2 = 0.9 - Alpha1', fontsize=14)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存图片
    img_path = output_path / f"qda_performance_curve_{dataset_name}.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"性能曲线已保存到: {img_path}")
    return str(img_path)


def save_results(
    results: Dict[str, Any],
    dataset_name: str,
    output_dir: str
) -> str:
    """
    保存优化结果到JSON文件
    
    Args:
        results: 网格搜索结果
        dataset_name: 数据集名称
        output_dir: 输出目录
    
    Returns:
        JSON文件保存路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_path = output_path / f"qda_optimization_results_{dataset_name}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"优化结果已保存到: {json_path}")
    return str(json_path)


def optimize_qda_for_dataset(
    dataset: str,
    init_cls: int = 20,
    seed: int = 1993,
    device: str = "cpu",
    step_size: float = 0.05,
    output_dir: str = "./qda_optimization_results"
) -> Dict[str, Any]:
    """
    为单个数据集执行完整的QDA参数优化流程
    
    Args:
        dataset: 数据集名称
        init_cls: 初始类别数
        seed: 随机种子
        device: 评估设备
        step_size: alpha1的步长
        output_dir: 输出目录
    
    Returns:
        优化结果字典
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"开始优化数据集: {dataset}")
    logging.info(f"{'='*60}")
    
    # 模块1：获取预训练特征和标签，获取类别的近似分布
    logging.info("\n=== 模块1：获取预训练特征和标签 ===")
    reference_stats, features, targets = get_gaussian_statistics_and_features(
        dataset, init_cls, seed
    )
    
    # 模块2：评估每组参数的准确度
    logging.info("\n=== 模块2：评估每组参数的准确度 ===")
    optimization_results = grid_search_qda_params(
        reference_stats, features, targets, device, step_size
    )
    
    # 模块3：绘制性能曲线图
    logging.info("\n=== 模块3：绘制性能曲线图 ===")
    curve_path = plot_performance_curve(
        optimization_results, dataset, output_dir
    )
    
    # 保存结果
    json_path = save_results(optimization_results, dataset, output_dir)
    
    # 添加额外信息到结果中
    optimization_results.update({
        "dataset": dataset,
        "init_cls": init_cls,
        "seed": seed,
        "device": device,
        "performance_curve_path": curve_path,
        "results_json_path": json_path
    })
    
    logging.info(f"\n数据集 {dataset} 优化完成！")
    logging.info(f"最佳参数: alpha1={optimization_results['best_alpha1']:.2f}, "
                f"alpha2={optimization_results['best_alpha2']:.2f}, "
                f"alpha3=0.1")
    logging.info(f"最佳准确度: {optimization_results['best_accuracy']:.2f}%")
    
    return optimization_results


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["imagenet-r", "imagenet-a", "vtab", "caltech-101"],
        help="要优化的数据集名称"
    )
    parser.add_argument(
        "--init_cls",
        type=int,
        default=20,
        help="初始类别数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1993,
        help="随机种子"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="评估设备"
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.05,
        help="alpha1的步长"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qda_optimization_results",
        help="输出目录"
    )
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.output_dir)
    
    # 打印配置信息
    logging.info("=== QDA参数网格搜索优化 ===")
    logging.info(f"数据集: {args.dataset}")
    logging.info(f"初始类别数: {args.init_cls}")
    logging.info(f"随机种子: {args.seed}")
    logging.info(f"设备: {args.device}")
    logging.info(f"步长: {args.step_size}")
    logging.info(f"输出目录: {args.output_dir}")
    
    # 执行优化
    try:
        results = optimize_qda_for_dataset(
            dataset=args.dataset,
            init_cls=args.init_cls,
            seed=args.seed,
            device=args.device,
            step_size=args.step_size,
            output_dir=args.output_dir
        )
        
        # 打印最终结果
        logging.info("\n" + "="*60)
        logging.info("优化完成！最终结果：")
        logging.info(f"最佳参数: alpha1={results['best_alpha1']:.2f}, "
                    f"alpha2={results['best_alpha2']:.2f}, alpha3=0.1")
        logging.info(f"最佳准确度: {results['best_accuracy']:.2f}%")
        logging.info(f"性能曲线: {results['performance_curve_path']}")
        logging.info(f"详细结果: {results['results_json_path']}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"优化过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
