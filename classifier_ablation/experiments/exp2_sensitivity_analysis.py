# In[]
import os
from tqdm import tqdm
os.chdir('/home/raoxuan/projects/fancy_sgp_lora_vit')
print("当前工作目录:", os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
"""
实验2: 参数敏感性分析
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from classifier.da_classifier_builder import QDAClassifierBuilder
from classifier_ablation.experiments.exp1_performance_surface import evaluate_qda_classifier, build_gaussian_statistics

def plot_experiment2_sensitivity(results, save_path=None):
    """
    绘制参数敏感性分析图，每个架构使用一条曲线
    
    Args:
        results: 字典，键为模型名称，值为对应的实验结果
        save_path: 保存路径
    """
    # 定义不同架构的颜色和标记
    model_styles = {
        'vit-b-p16-clip': {'color': 'b', 'marker': 'o', 'linestyle': '-'},
        'vit-b-p16': {'color': 'r', 'marker': 's', 'linestyle': '--'},
        'vit-b-p16-dino': {'color': 'g', 'marker': '^', 'linestyle': '-.'},
        'vit-b-p16-mocov3': {'color': 'm', 'marker': 'd', 'linestyle': ':'}
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(3.5*2, 2.5))  # 只需要两个子图
    
    # 1. 固定alpha1，变动alpha2
    ax = axes[0]
    for model_name, model_results in results.items():
        if model_name in model_styles:
            style = model_styles[model_name]
            ax.plot(model_results['fixed_alpha1']['alpha2'],
                    model_results['fixed_alpha1']['accuracy'],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=1.0,
                    marker=style['marker'],
                    markersize=3,
                    label=model_name)
    
    ax.set_xlabel('Alpha2 (Alpha1=0.1)', fontsize=7)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=7)
    ax.set_title('Fixed Alpha1', fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=6, loc='best')
    
    # 2. 固定alpha2，变动alpha1
    ax = axes[1]
    for model_name, model_results in results.items():
        if model_name in model_styles:
            style = model_styles[model_name]
            ax.plot(model_results['fixed_alpha2']['alpha1'],
                    model_results['fixed_alpha2']['accuracy'],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=1.0,
                    marker=style['marker'],
                    markersize=3,
                    label=model_name)
    
    ax.set_xlabel('Alpha1 (Alpha2=2.0)', fontsize=7)
    ax.set_ylabel('Dataset-wise Avg-Acc', fontsize=7)
    ax.set_title('Fixed Alpha2', fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=6, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"敏感性分析图已保存到: {save_path}")
    plt.show()

# In[]
 
import os
import json
import torch
from compensator.gaussian_statistics import GaussianStatistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels, infer_dataset_ids_from_labels

model_names = ["vit-b-p16-clip", "vit-b-p16", "vit-b-p16-dino", "vit-b-p16-mocov3"]
output_dir="实验结果保存/分类器消融实验"
num_shots = 128
iterations = 0
results = {}
for model_name in model_names:
    results[model_name] = {
        "fixed_alpha1": {"alpha1": [], "alpha2": [], "accuracy": []},
        "fixed_alpha2": {"alpha1": [], "alpha2": [], "accuracy": []}
    }


alpha3_fixed = 0.5
alpha2_range = (0.0, 3.0)
alpha1_range = (0, 0.5)

for model_name in model_names:
    vit = get_vit(model_name)
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=0)

    # 2. 固定alpha1，变动alpha2
    print(f"\n2. 固定alpha1，变动alpha2")
    fixed_alpha1 = 0.1
    alpha2_values = np.linspace(alpha2_range[0], alpha2_range[1], 11)
    train_stats = build_gaussian_statistics(train_features, train_labels)

    for alpha2 in alpha2_values:
        print(f"测试: alpha1={fixed_alpha1:.4f}, alpha2={alpha2:.4f}")
        
        
        acc = evaluate_qda_classifier(fixed_alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                        test_labels, test_dataset_ids, device="cuda", return_class_wise=True)
        
        results[model_name]['fixed_alpha1']['alpha1'].append(fixed_alpha1)
        results[model_name]['fixed_alpha1']['alpha2'].append(alpha2)
        results[model_name]['fixed_alpha1']['accuracy'].append(acc)
        
        print(f"准确率: {acc:.4f}")

    print(f"\n3. 固定alpha2，变动alpha1")
    fixed_alpha2 = 2.0
    alpha1_values = np.linspace(alpha1_range[0], alpha1_range[1], 11)

    for alpha1 in alpha1_values:
        print(f"测试: alpha1={alpha1:.4f}, alpha2={fixed_alpha2:.4f}")
        acc = evaluate_qda_classifier(alpha1, fixed_alpha2, alpha3_fixed, train_stats, test_features,
                                        test_labels, test_dataset_ids, device="cuda", return_class_wise=True)
        results[model_name]['fixed_alpha2']['alpha1'].append(alpha1)
        results[model_name]['fixed_alpha2']['alpha2'].append(fixed_alpha2)
        results[model_name]['fixed_alpha2']['accuracy'].append(acc)
        print(f"准确率: {acc:.4f}")
        
# In[]
import matplotlib.pyplot as plt
def plot_experiment2_sensitivity_individual(results, save_path=None):
    """
    为每个模型分别绘制两张独立的敏感性曲线图（共8个子图）：
    - 左列：固定 alpha1，变动 alpha2
    - 右列：固定 alpha2，变动 alpha1
    y轴范围保持自然（不手动缩放），仅展示原始数据波动。

    Args:
        results: dict，键为模型名，值为实验结果
        save_path: 保存路径（可选）
    """
    model_names = [
        'vit-b-p16-clip',
        'vit-b-p16',
        'vit-b-p16-dino',
        'vit-b-p16-mocov3'
    ]
    
    model_colors = {
        'vit-b-p16-clip': 'b',
        'vit-b-p16': 'r',
        'vit-b-p16-dino': 'g',
        'vit-b-p16-mocov3': 'm'
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 8))
    if axes.ndim == 1:
        axes = axes.reshape(4, 2)

    for row, model_name in enumerate(model_names):
        color = model_colors.get(model_name, 'k')
        
        # 左图：固定 alpha1，变动 alpha2
        ax_left = axes[row, 0]
        if model_name in results and 'fixed_alpha1' in results[model_name]:
            x1 = results[model_name]['fixed_alpha1']['alpha2']
            y1 = results[model_name]['fixed_alpha1']['accuracy']
            ax_left.plot(x1, y1, color=color, marker='o', markersize=4, linewidth=1.2)
            ax_left.set_xlabel(r'$\alpha_2$', fontsize=9)
            ax_left.set_ylabel('Avg-Acc', fontsize=9)
            ax_left.set_title(f'{model_name}\n(α₁=0.1)', fontsize=9)
            ax_left.grid(True, linestyle='--', alpha=0.4)
            ax_left.tick_params(labelsize=8)
        else:
            ax_left.set_visible(False)

        # 右图：固定 alpha2，变动 alpha1
        ax_right = axes[row, 1]
        if model_name in results and 'fixed_alpha2' in results[model_name]:
            x2 = results[model_name]['fixed_alpha2']['alpha1']
            y2 = results[model_name]['fixed_alpha2']['accuracy']
            ax_right.plot(x2, y2, color=color, marker='s', markersize=4, linewidth=1.2)
            ax_right.set_xlabel(r'$\alpha_1$', fontsize=9)
            ax_right.set_ylabel('Avg-Acc', fontsize=9)
            ax_right.set_title(f'{model_name}\n(α₂=2.0)', fontsize=9)
            ax_right.grid(True, linestyle='--', alpha=0.4)
            ax_right.tick_params(labelsize=8)
        else:
            ax_right.set_visible(False)

    plt.tight_layout(pad=1.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"8子图敏感性分析（自然y轴）已保存到: {save_path}")
    plt.show()

plot_experiment2_sensitivity_individual(results)
# %%
