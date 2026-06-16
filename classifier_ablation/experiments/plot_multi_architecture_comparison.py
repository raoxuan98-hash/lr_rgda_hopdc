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
import seaborn as sns

def load_constraint_results(model_name, save_dir):
    """
    加载已保存的约束条件实验结果
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
        
    Returns:
        alpha1_values: α1值数组
        qda_accuracies: QDA准确度数组
        sgd_linear_accuracies: 线性SGD准确度数组
        sgd_nonlinear_accuracies: 非线性SGD准确度数组
        ncm_accuracies: NCM准确度数组
        lda_accuracies: LDA准确度数组
    """
    load_path = os.path.join(save_dir, f"{model_name}_constraint_results.npz")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到已保存的结果文件: {load_path}")
    
    data = np.load(load_path)
    alpha1_values = data['alpha1_values']
    qda_accuracies = data['qda_accuracies']
    sgd_linear_accuracies = data['sgd_linear_accuracies']
    ncm_accuracies = data['ncm_accuracies']
    lda_accuracies = data['lda_accuracies']
    
    print(f"已加载约束条件实验结果: {load_path}")
    
    return alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies

def load_constraint_results_from_csv(model_name, save_dir):
    """
    从CSV文件加载约束条件实验结果
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
        
    Returns:
        alpha1_values: α1值数组
        qda_accuracies: QDA准确度数组
        sgd_linear_accuracies: 线性SGD准确度数组
        sgd_nonlinear_accuracies: 非线性SGD准确度数组
        ncm_accuracies: NCM准确度数组
        lda_accuracies: LDA准确度数组
    """
    csv_path = os.path.join(save_dir, f"{model_name}_constraint_results.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到已保存的结果文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    alpha1_values = df['alpha1'].values
    qda_accuracies = df['qda_accuracy'].values
    sgd_linear_accuracies = df['sgd_linear_accuracy'].values
    sgd_nonlinear_accuracies = df['sgd_nonlinear_accuracy'].values
    ncm_accuracies = df['ncm_accuracy'].values
    lda_accuracies = df['lda_accuracy'].values
    
    print(f"已加载约束条件实验结果: {csv_path}")
    
    return alpha1_values, qda_accuracies, sgd_linear_accuracies, sgd_nonlinear_accuracies, ncm_accuracies, lda_accuracies

def plot_architecture_comparison_grid_pretty(model_names, base_output_dir, iterations=0,
                                             save_path=None, figsize=(7.2, 2.4)):
    """
    绘制三图横排（VIT-B-P16, CLIP, DINO）版本（论文级美化版）
    """

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.15)

    colors = {
        "QDA": "#1270C2",
        "SGD-linear": "#B59A30",
        "SGD-nonlinear": "#3C8D86",
        "NCM": "#6FA8DC",
        "LDA": "#A27CCD",
    }

    model_data = {}

    for model_name in model_names:
        model_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}")
        
        try:
            # 尝试从npz文件加载
            alpha1_values, qda_acc, sgd_linear_acc, ncm_acc, lda_acc = load_constraint_results(
                model_name, model_dir)
        except FileNotFoundError:
            try:
                # 尝试从CSV文件加载
                alpha1_values, qda_acc, sgd_linear_acc, sgd_nonlinear_acc, ncm_acc, lda_acc = load_constraint_results_from_csv(
                    model_name, model_dir)
            except FileNotFoundError:
                print(f"警告: 无法找到模型 {model_name} 的结果文件，跳过该模型")
                continue
        
        model_data[model_name] = {
            "alpha1_values": alpha1_values,
            "qda_acc": qda_acc,
            "sgd_linear_acc": sgd_linear_acc,
            "ncm_acc": ncm_acc,
            "lda_acc": lda_acc
        }

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, (model_name, data) in zip(axes, model_data.items()):
        alpha = data["alpha1_values"]
        # 转换为百分比
        qda_pct = np.array(data["qda_acc"]) * 100
        sgd_linear_pct = np.array(data["sgd_linear_acc"]) * 100
        
        ax.plot(alpha, qda_pct, marker='o', markersize=4, linewidth=2.0,
                label="QDA", color=colors["QDA"])
        ax.plot(alpha, sgd_linear_pct, marker='X', markersize=4, linewidth=2.0,
                label="SGD-linear", color=colors["SGD-linear"])

        ax.set_title(model_name.upper(), fontsize=10, weight='bold')
        ax.set_xlabel(r'$\alpha_1$', fontsize=9)
        if model_name == "vit-b-p16":
            ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.68)
        ax.tick_params(axis='both', labelsize=8, width=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return model_data

def create_performance_summary_table(model_data, save_path=None):
    """
    创建性能汇总表
    
    Args:
        model_data: 模型数据字典
        save_path: 保存路径
    """
    classifier_types = ['qda', 'sgd_linear', 'sgd_nonlinear', 'ncm', 'lda']
    classifier_labels = ['QDA', 'SGD-linear', 'SGD-nonlinear', 'NCM', 'LDA']
    
    # 创建汇总表
    summary_data = []
    
    for model_name, data in model_data.items():
        for classifier_type, classifier_label in zip(classifier_types, classifier_labels):
            # 修正键名以匹配实际数据结构
            key = f'{classifier_type}_acc'
            if key in data:
                accuracies = data[key]
                best_acc = np.max(accuracies) * 100
                mean_acc = np.mean(accuracies) * 100
                std_acc = np.std(accuracies) * 100
                
                if classifier_type in ['ncm', 'lda']:
                    best_alpha1 = "N/A"
                else:
                    best_idx = np.argmax(accuracies)
                    best_alpha1 = data['alpha1_values'][best_idx]
                
                summary_data.append({
                    'Model': model_name,
                    'Classifier': classifier_label,
                    'Best Accuracy (%)': f"{best_acc:.2f}",
                    'Mean Accuracy (%)': f"{mean_acc:.2f}",
                    'Std Accuracy (%)': f"{std_acc:.2f}",
                    'Best α1': best_alpha1
                })
    
    df = pd.DataFrame(summary_data)
    
    # 保存表格
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"性能汇总表已保存到: {save_path}")
    
    return df
# In[]
if __name__ == '__main__':
    # 设置参数
    model_names = ["vit-b-p16", "vit-b-p16-clip", "vit-b-p16-dino"]
    base_output_dir = "实验结果保存/分类器消融实验"
    iterations = 0
    
    # 创建输出目录
    output_dir = os.path.join(base_output_dir, "multi_architecture_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("多架构性能对比分析")
    print("="*60)
    
    # 绘制网格形式的架构对比图
    grid_save_path = os.path.join(output_dir, "architecture_grid_comparison.png")
    
    model_data = plot_architecture_comparison_grid_pretty(
        model_names, base_output_dir, iterations, grid_save_path)
    
    # 创建性能汇总表
    table_save_path = os.path.join(output_dir, "performance_summary.csv")
    summary_df = create_performance_summary_table(model_data, table_save_path)
    
    print("\n性能汇总表:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("多架构对比分析完成")
    print("="*60)

# %%
