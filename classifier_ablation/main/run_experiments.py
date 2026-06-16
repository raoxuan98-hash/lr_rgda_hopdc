"""
主运行脚本：运行所有分类器消融实验
"""
import os
import json
import torch
from compensator.gaussian_statistics import GaussianStatistics
from data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels, infer_dataset_ids_from_labels
from experiments.exp1_performance_surface import run_experiment1_performance_surface
from experiments.exp2_sensitivity_analysis import run_experiment2_sensitivity_analysis
from experiments.exp3_classifier_comparison import run_experiment3_classifier_comparison
from experiments.exp4_efficiency_comparison import run_experiment4_efficiency_comparison


def run_all_ablation_experiments(output_dir="实验结果保存/分类器消融实验"):
    """
    运行所有消融实验
    
    Args:
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始运行所有消融实验...")
    
    # 实验参数
    num_shots = 128
    model_name = "vit-b-p16-clip"
    adapt_backbone = True
    iterations = 0
    
    # 加载数据
    print("\n加载数据...")
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    # 获取模型
    print("\n获取模型...")
    vit = get_vit(vit_name=model_name)
    
    # 适应网络主干
    if adapt_backbone:
        adapt_loader = create_adapt_loader(train_subsets)
        vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
    
    # 提取特征
    print("\n提取特征...")
    (train_features, train_labels, train_dataset_ids,
     test_features, test_labels, test_dataset_ids) = extract_features_and_labels(
        vit, train_loader, test_loader, model_name, num_shots=num_shots, 
        dataset_manager=dataset, iterations=iterations
    )
    
    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_dataset_ids = torch.tensor(test_dataset_ids)
    
    # 构建高斯统计量
    print("\n构建高斯统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)
    
    # 实验1: 性能曲面等高线图
    print("\n" + "="*60)
    print("运行实验1: 性能曲面等高线图")
    print("="*60)
    best_alpha1, best_alpha2, best_acc = run_experiment1_performance_surface(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_min=0, alpha1_max=5.0, alpha2_min=0, alpha2_max=5.0,
        alpha1_points=11, alpha2_points=11,
        save_path=f"{output_dir}/exp1_contour.png"
    )
    
    # 实验2: 参数敏感性分析
    print("\n" + "="*60)
    print("运行实验2: 参数敏感性分析")
    print("="*60)
    exp2_results = run_experiment2_sensitivity_analysis(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_range=(0.0, 5.0), alpha2_range=(0.0, 5.0),
        alpha_sum=3.0, fixed_points=11,
        save_path=f"{output_dir}/exp2_sensitivity.png"
    )
    
    # 实验3: 分类器对比
    print("\n" + "="*60)
    print("运行实验3: 分类器对比")
    print("="*60)
    exp3_results = run_experiment3_classifier_comparison(
        train_stats, test_features, test_labels, test_dataset_ids,
        sgd_epochs=5, sgd_lr=0.01,
        save_path=f"{output_dir}/exp3_comparison.png"
    )
    
    # 实验4: 效率对比
    print("\n" + "="*60)
    print("运行实验4: 效率对比")
    print("="*60)
    exp4_results = run_experiment4_efficiency_comparison(
        train_stats, test_features,
        save_path=f"{output_dir}/exp4_efficiency.png"
    )
    
    # 保存所有结果
    all_results = {
        'exp1': {
            'best_alpha1': best_alpha1,
            'best_alpha2': best_alpha2,
            'best_accuracy': best_acc,
        },
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results
    }
    
    with open(f"{output_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n所有实验完成！结果已保存到 {output_dir}/")
    print(f"包含文件:")
    print(f"  - exp1_contour.png: 实验1性能曲面等高线图")
    print(f"  - exp2_sensitivity.png: 实验2参数敏感性分析图")
    print(f"  - exp3_comparison.png: 实验3分类器对比图")
    print(f"  - exp4_efficiency.png: 实验4效率对比图")
    print(f"  - all_results.json: 所有实验结果数据")

def quick_test_all_experiments():
    """
    快速测试所有实验（用于验证代码正确性）
    """
    print("开始快速测试所有实验...")
    
    # 实验参数
    num_shots = 128
    model_name = "vit-b-p16-clip"
    
    # 加载数据
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    # 获取模型
    vit = get_vit(vit_name=model_name)
    
    # 提取特征（使用较少数据）
    (train_features, train_labels, train_dataset_ids,
     test_features, test_labels, test_dataset_ids) = extract_features_and_labels(
        vit, train_loader, test_loader, model_name, num_shots=num_shots,
        dataset_manager=dataset, iterations=0
    )
    
    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_dataset_ids = torch.tensor(test_dataset_ids)
    
    # 构建高斯统计量
    train_stats = build_gaussian_statistics(train_features, train_labels)
    
    # 快速测试实验1: 5x5网格
    print("\n快速测试实验1: 5x5网格...")
    run_experiment1_performance_surface(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_min=0, alpha1_max=2.0, alpha2_min=0, alpha2_max=2.0,
        alpha1_points=5, alpha2_points=5
    )
    
    # 快速测试实验2: 5个点
    print("\n快速测试实验2: 参数敏感性...")
    run_experiment2_sensitivity_analysis(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_range=(0.0, 2.0), alpha2_range=(0.0, 2.0),
        alpha_sum=2.0, fixed_points=5
    )
    
    # 快速测试实验3: 基础对比
    print("\n快速测试实验3: 分类器对比...")
    run_experiment3_classifier_comparison(
        train_stats, test_features, test_labels, test_dataset_ids,
        sgd_epochs=2, sgd_lr=0.01
    )
    
    # 快速测试实验4: 效率测试
    print("\n快速测试实验4: 效率对比...")
    run_experiment4_efficiency_comparison(train_stats, test_features)
    
    print("\n快速测试完成！")

if __name__ == "__main__":
    # 可以选择运行完整实验或快速测试
    # run_all_ablation_experiments()  # 完整实验
    quick_test_all_experiments()  # 快速测试