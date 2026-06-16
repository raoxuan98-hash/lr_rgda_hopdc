# In[]
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())
sys.path.append('/home/raoxuan/projects/low_rank_rda')

def load_constraint_results(model_name, save_dir):
    """
    加载已保存的约束条件实验结果
    """
    load_path = os.path.join(save_dir, f"{model_name}_constraint_results.npz")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到已保存的结果文件: {load_path}")
    
    data = np.load(load_path)
    alpha1_values = data['alpha1_values']
    qda_accuracies = data['qda_accuracies']
    sgd_linear_accuracies = data['sgd_linear_accuracies']
    sgd_nonlinear_accuracies = data['sgd_nonlinear_accuracies']
    ncm_accuracies = data['ncm_accuracies']
    lda_accuracies = data['lda_accuracies']
    
    print(f"已加载约束条件实验结果: {load_path}")
    
    return alpha1_values, qda_accuracies, sgd_linear_accuracies, sgd_nonlinear_accuracies, ncm_accuracies, lda_accuracies

def load_constraint_results_from_csv(model_name, save_dir):
    """
    从CSV文件加载约束条件实验结果
    """
    csv_path = os.path.join(save_dir, f"{model_name}_constraint_results.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到已保存的结果文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    alpha1_values = df['alpha1'].values
    qda_accuracies = df['qda_accuracy'].values
    sgd_linear_accuracies = df['sgd_linear_accuracy'].values
    ncm_accuracies = df['ncm_accuracy'].values
    lda_accuracies = df['lda_accuracy'].values
    
    print(f"已加载约束条件实验结果: {csv_path}")
    
    return alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies

def plot_architecture_comparison_grid_pretty(model_names, base_output_dir, iterations=0,
                                             save_path=None, figsize=(6.2, 2.1)):
    """
    绘制三图横排（VIT-B-P16, CLIP, DINO）版本（论文级美化版）
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.15)

    colors = {
        "QDA": "#0779DC",
        "SGD-linear": "#F50202",
        "SGD-nonlinear": "#3C8D86",
        "NCM": "#6FA8DC",
        "LDA": "#A27CCD"}

    model_data = {}

    for model_name in model_names:
        # 修改文件路径以适应实际文件结构
        model_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}")
        csv_path = os.path.join(model_dir, f"{model_name}_constraint_results.csv")

        if os.path.exists(csv_path):
            alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies = load_constraint_results_from_csv(model_name, model_dir)
            model_data[model_name] = {
                "alpha1_values": alpha1_values,
                "qda_means": qda_accuracies,
                "sgd_linear_means": sgd_linear_accuracies,
                "lda_means": lda_accuracies,
                "ncm_means": ncm_accuracies
            }
        else:
            print(f"缺文件: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, (model_name, data) in zip(axes, model_data.items()):
        alpha = data["alpha1_values"]
        # 转换为百分比
        qda_acc = data["qda_means"] * 100
        sgd_linear_acc = data["sgd_linear_means"] * 100
        
        ax.plot(alpha, qda_acc, marker='*', markersize=3, linewidth=1.4,
                label="QDA", color=colors["QDA"], linestyle='-.')
        ax.plot(alpha, sgd_linear_acc, marker='X', markersize=3, linewidth=1.4,
                label="SGD", color=colors["SGD-linear"], linestyle='--')
        
        if model_name == "vit-b-p16":
            title = "ViT/B-Sup21K"
        elif model_name == "vit-b-p16-clip":
            title = "ViT/B-CLIP"
        elif model_name == "vit-b-p16-dino":
            title = "ViT/B-DINO"
        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_xlabel(r'$\alpha_1$', fontsize=9)
        if model_name == "vit-b-p16":
            ax.set_ylabel("Accuracy (%)", fontsize=9)
        
        # 设置y轴为5个刻度，精确到小数点后1位
        y_min = min(min(qda_acc), min(sgd_linear_acc))
        y_max = max(max(qda_acc), max(sgd_linear_acc))
        
        # 扩展范围以确保数据点不会被截断
        y_range = y_max - y_min
        y_min_adj = y_min - 0.05 * y_range
        y_max_adj = y_max + 0.05 * y_range
        
        ax.set_ylim(y_min_adj, y_max_adj)
        
        # 设置5个等间距的y轴刻度
        y_ticks = np.linspace(y_min_adj, y_max_adj, 5)
        ax.set_yticks(y_ticks)
        
        # 格式化y轴刻度标签，保留1位小数
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.68)
        ax.tick_params(axis='both', labelsize=8, width=0.6)

    # 只在第一个子图添加图例，放在右下方
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower right', fontsize=8, frameon=True, 
                   fancybox=True, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # 🔥 必须加这一句返回数据！
    return model_data


# In[]
if __name__ == '__main__':
    # 设置参数
    model_names = ["vit-b-p16", "vit-b-p16-clip", "vit-b-p16-dino"]
    base_output_dir = "实验结果保存/分类器消融实验"
    iterations = 0
    
    # 创建输出目录
    output_dir = os.path.join(base_output_dir, "complete_multi_architecture_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("完整四架构性能对比分析")
    print("="*80)
    
    # 绘制2x2网格形式的四架构对比图
    grid_save_path = os.path.join(output_dir, "four_architecture_grid_comparison.png")
    model_data = plot_architecture_comparison_grid_pretty(
        model_names, base_output_dir, iterations, grid_save_path)
# %%
