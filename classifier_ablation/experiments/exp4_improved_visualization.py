#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的实验4可视化: SGD分类器参数敏感性等高线图
加载已保存的数据并重新绘制更美观的可视化效果
"""
# In[]
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
os.chdir('/home/raoxuan/projects/fancy_sgp_lora_vit')
print("当前工作目录:", os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 设置matplotlib参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Georgia']
plt.rcParams['mathtext.fontset'] = 'stix'
# 添加字体警告处理
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def load_saved_data(sgd_results_path, comparison_results_path):
    print("加载SGD结果数据...")
    sgd_data = np.load(sgd_results_path)
    
    print("加载对比结果数据...")
    comparison_data = np.load(comparison_results_path)
    
    data = {
        'alpha1_values': sgd_data['alpha1_values'],
        'alpha2_values': sgd_data['alpha2_values'],
        'sgd_accuracy_matrix': sgd_data['accuracy_matrix'],
        'rgda_accuracy_matrix': comparison_data['rgda_accuracy_matrix'],
        'diff_accuracy_matrix': comparison_data['diff_accuracy_matrix']
    }
    
    print(f"数据加载完成:")
    print(f"  Alpha1范围: {data['alpha1_values'].min():.3f} - {data['alpha1_values'].max():.3f}")
    print(f"  Alpha2范围: {data['alpha2_values'].min():.3f} - {data['alpha2_values'].max():.3f}")
    print(f"  SGD准确率范围: {data['sgd_accuracy_matrix'].min():.2f}% - {data['sgd_accuracy_matrix'].max():.2f}%")
    print(f"  RGDA准确率范围: {data['rgda_accuracy_matrix'].min():.2f}% - {data['rgda_accuracy_matrix'].max():.2f}%")
    print(f"  差值范围: {data['diff_accuracy_matrix'].min():.2f}% - {data['diff_accuracy_matrix'].max():.2f}%")
    
    return data

def plot_improved_sgd_contour(alpha1_values, alpha2_values, accuracy_matrix, 
                             save_path=None, cmap='viridis', title=None):
    plt.figure(figsize=(3.5, 2.5))  # 稍微增大尺寸以获得更好的可读性
    
    alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values)
    vmin, vmax = np.min(accuracy_matrix), np.max(accuracy_matrix)
    levels = np.linspace(vmin, vmax, 21)
    contourf = plt.contourf(alpha1_grid, alpha2_grid, accuracy_matrix.T, 
                           levels=levels, cmap=cmap, extend='both')
    contour_lines = plt.contour(alpha1_grid, alpha2_grid, accuracy_matrix.T, 
                               levels=levels[::2], colors='black', alpha=0.4, linewidths=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(contourf, shrink=0.8, aspect=20)
    cbar.set_label('Average accuracy', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.formatter = FormatStrFormatter('%.1f')
    cbar.update_ticks()
    
    # 找到最佳准确率及其对应的参数
    max_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_alpha1 = alpha1_values[max_idx[0]]
    best_alpha2 = alpha2_values[max_idx[1]]
    best_acc = accuracy_matrix[max_idx]
    
    # 标记最佳点
    plt.plot(best_alpha1, best_alpha2, 'r*', markersize=8, 
            label=f'Best: ({best_alpha1:.2f}, {best_alpha2:.2f}) = {best_acc:.2f}%')
    
    # 设置标签和标题
    plt.xlabel(r'$\alpha_1^{\rm RGDA}$', fontsize=8)
    plt.ylabel(r'$\alpha_2^{\rm RGDA}$', fontsize=8)
    
    
    # 设置刻度
    plt.xticks(np.linspace(min(alpha1_values), max(alpha1_values), 5), fontsize=8)
    plt.yticks(np.linspace(min(alpha2_values), max(alpha2_values), 5), fontsize=8)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    # plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"改进的SGD等高线图已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, best_alpha2, best_acc

def plot_alpha1_sensitivity_comparison(alpha1_values, alpha2_values, sgd_accuracy_matrix,
                                      rgda_accuracy_matrix, fixed_alpha2=1.0, save_path=None):
    alpha2_idx = np.argmin(np.abs(alpha2_values - fixed_alpha2))
    actual_alpha2 = alpha2_values[alpha2_idx]
    sgd_accuracy_alpha2 = sgd_accuracy_matrix[:, alpha2_idx]  # 转置索引
    rgda_accuracy_alpha2 = rgda_accuracy_matrix[:, alpha2_idx]  # 转置索引
    
    lda_accuracy = 0.731329045647513 * 100 
    ncm_accuracy = 0.6967877885868847 * 100
    
    sgd_poly = np.polyfit(alpha1_values, sgd_accuracy_alpha2, 4)
    rgda_poly = np.polyfit(alpha1_values, rgda_accuracy_alpha2, 4)
    
    # 生成更密集的x值用于平滑曲线
    alpha1_dense = np.linspace(min(alpha1_values), max(alpha1_values), 100)
    sgd_fitted = np.polyval(sgd_poly, alpha1_dense)
    rgda_fitted = np.polyval(rgda_poly, alpha1_dense)
    
    # 创建图表
    plt.figure(figsize=(3.5, 2.5))
    

    plt.plot(alpha1_dense, sgd_fitted, '-.', color='royalblue', linewidth=2.0,
            label='SGD', alpha=0.9)
    plt.plot(alpha1_dense, rgda_fitted, '--', color='orangered', linewidth=2.0,
            label='RGDA', alpha=0.9)
    
    # 可选：在原始数据点位置添加标记点
    plt.scatter(alpha1_values, sgd_accuracy_alpha2, color='royalblue', s=20, alpha=0.8)
    plt.scatter(alpha1_values, rgda_accuracy_alpha2, color='orangered', s=20, alpha=0.8)
    
    plt.axhline(y=lda_accuracy, color='green', linestyle='--', linewidth=1.5,
                label=f'LDA: {lda_accuracy:.1f}%', alpha=0.7)
    plt.axhline(y=ncm_accuracy, color='orange', linestyle=':', linewidth=1.5,
                label=f'NCM: {ncm_accuracy:.1f}%', alpha=0.7)
    
    # 设置标签和标题
    plt.xlabel(r'$\alpha_1^{\rm RGDA}$', fontsize=8)
    plt.ylabel('Accuracy (%)', fontsize=8)
    
    # 设置网格和图例
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', fontsize=7, framealpha=0.9)

    # 设置坐标轴范围
    y_min = min(np.min(sgd_accuracy_alpha2), np.min(rgda_accuracy_alpha2), ncm_accuracy) - 2
    y_max = max(np.max(sgd_accuracy_alpha2), np.max(rgda_accuracy_alpha2), lda_accuracy) + 2
    plt.ylim(y_min, y_max)
    
    # 设置刻度 - y轴精确到小数点后1位
    plt.xticks(np.linspace(min(alpha1_values), max(alpha1_values), 5), fontsize=8)
    
    # 生成y轴刻度，确保包含小数位
    y_ticks = np.linspace(y_min, y_max, 5)
    plt.yticks(y_ticks, [f'{y:.1f}' for y in y_ticks], fontsize=8)
    
    # 设置y轴格式，确保显示小数位
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Alpha1敏感性比较图已保存到: {save_path}")
    plt.show()
    
    return {
        'alpha2_value': actual_alpha2,
        'alpha1_values': alpha1_values,
        'sgd_accuracy': sgd_accuracy_alpha2,
        'rgda_accuracy': rgda_accuracy_alpha2,
        'sgd_fitted_curve': (alpha1_dense, sgd_fitted),
        'rgda_fitted_curve': (alpha1_dense, rgda_fitted),
        'baselines': {'lda': lda_accuracy, 'ncm': ncm_accuracy},
    }

def create_summary_statistics(data):
    diff_matrix = data['diff_accuracy_matrix']
    
    # 统计RGDA优于SGD的比例
    total_points = diff_matrix.size
    rgda_better_points = np.sum(diff_matrix > 0)
    rgda_better_ratio = rgda_better_points / total_points * 100
    mean_diff = np.mean(diff_matrix)
    max_diff = np.max(diff_matrix)
    min_diff = np.min(diff_matrix)
    
    # SGD统计
    sgd_mean = np.mean(data['sgd_accuracy_matrix'])
    sgd_max = np.max(data['sgd_accuracy_matrix'])
    sgd_min = np.min(data['sgd_accuracy_matrix'])
    
    # RGDA统计
    rgda_mean = np.mean(data['rgda_accuracy_matrix'])
    rgda_max = np.max(data['rgda_accuracy_matrix'])
    rgda_min = np.min(data['rgda_accuracy_matrix'])
    
    print("\n" + "="*60)
    print("实验统计摘要:")
    print(f"SGD分类器统计:")
    print(f"  平均准确率: {sgd_mean:.2f}%")
    print(f"  最高准确率: {sgd_max:.2f}%")
    print(f"  最低准确率: {sgd_min:.2f}%")
    print(f"RGDA分类器统计:")
    print(f"  平均准确率: {rgda_mean:.2f}%")
    print(f"  最高准确率: {rgda_max:.2f}%")
    print(f"  最低准确率: {rgda_min:.2f}%")
    print(f"RGDA vs SGD对比:")
    print(f"  RGDA优于SGD的参数组合比例: {rgda_better_ratio:.2f}% ({rgda_better_points}/{total_points})")
    print(f"  平均精度差值(RGDA-SGD): {mean_diff:.2f}%")
    print(f"  最大精度差值(RGDA-SGD): {max_diff:.2f}%")
    print(f"  最小精度差值(RGDA-SGD): {min_diff:.2f}%")
    print("="*60)
    
    return {
        'sgd_mean': sgd_mean, 'sgd_max': sgd_max, 'sgd_min': sgd_min,
        'rgda_mean': rgda_mean, 'rgda_max': rgda_max, 'rgda_min': rgda_min,
        'rgda_better_ratio': rgda_better_ratio, 'mean_diff': mean_diff,
        'max_diff': max_diff, 'min_diff': min_diff
    }

# In[]
def main():
    """
    主函数：加载已保存的数据并重新绘制可视化效果
    """
    # 设置路径
    base_dir = "实验结果保存/分类器消融实验/vit-b-p16-clip_sgd_contour"
    sgd_results_path = os.path.join(base_dir, "vit-b-p16-clip_sgd_results.npz")
    comparison_results_path = os.path.join(base_dir, "vit-b-p16-clip_comparison_results.npz")
    
    # 创建输出目录
    improved_output_dir = os.path.join(base_dir, "improved_visualizations")
    os.makedirs(improved_output_dir, exist_ok=True)
    
    print("="*60)
    print("改进的SGD分类器参数敏感性可视化")
    print("="*60)
    
    # 加载数据
    data = load_saved_data(sgd_results_path, comparison_results_path)
    
    # 创建统计摘要
    stats = create_summary_statistics(data)
    
    # 绘制改进的SGD等高线图
    print("\n绘制改进的SGD分类器等高线图...")
    sgd_cmaps = ["viridis", "jet", "Spectral", "Blues", "cividis"]
    for cmap in sgd_cmaps:
        save_path = os.path.join(improved_output_dir, f"improved_sgd_contour_{cmap}.png")
        best_alpha1, best_alpha2, best_acc = plot_improved_sgd_contour(
            data['alpha1_values'], data['alpha2_values'], data['sgd_accuracy_matrix'],
            save_path, cmap, title=f"SGD Classifier Performance ({cmap})")
        print(f"  {cmap}: 最佳参数 ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.2f}%")
    
    # 绘制改进的RGDA-SGD差值等高线图
    print("\n绘制改进的RGDA-SGD差值等高线图...")
    diff_cmaps = ["RdBu_r", "coolwarm", "seismic", "bwr", "PiYG"]
    # diff_cmaps = ["viridis", "plasma", "inferno", "magma", "cividis"]
    for cmap in diff_cmaps:
        save_path = os.path.join(improved_output_dir, f"improved_rgda_sgd_difference_{cmap}.png")
        best_alpha1, best_alpha2, best_diff = plot_improved_difference_contour(
            data['alpha1_values'], data['alpha2_values'], data['diff_accuracy_matrix'],
            data['sgd_accuracy_matrix'], data['rgda_accuracy_matrix'],
            save_path, cmap, title=f"RGDA vs SGD Performance Difference ({cmap})")
        print(f"  {cmap}: 最大差值 ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_diff:.2f}%")
    
    # 绘制固定alpha2值时的alpha1敏感性比较图
    print("\n绘制固定alpha2值时的alpha1敏感性比较图...")
    sensitivity_save_path = os.path.join(improved_output_dir, "alpha1_sensitivity_comparison.png")
    sensitivity_results = plot_alpha1_sensitivity_comparison(
        data['alpha1_values'], data['alpha2_values'],
        data['sgd_accuracy_matrix'], data['rgda_accuracy_matrix'],
        fixed_alpha2=1.0, save_path=sensitivity_save_path)

    # 保存统计摘要
    stats_path = os.path.join(improved_output_dir, "experiment_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("SGD vs RGDA分类器对比实验统计摘要\n")
        f.write("="*50 + "\n")
        f.write(f"SGD分类器统计:\n")
        f.write(f"  平均准确率: {stats['sgd_mean']:.2f}%\n")
        f.write(f"  最高准确率: {stats['sgd_max']:.2f}%\n")
        f.write(f"  最低准确率: {stats['sgd_min']:.2f}%\n")
        f.write(f"\nRGDA分类器统计:\n")
        f.write(f"  平均准确率: {stats['rgda_mean']:.2f}%\n")
        f.write(f"  最高准确率: {stats['rgda_max']:.2f}%\n")
        f.write(f"  最低准确率: {stats['rgda_min']:.2f}%\n")
        f.write(f"\nRGDA vs SGD对比:\n")
        f.write(f"  RGDA优于SGD的参数组合比例: {stats['rgda_better_ratio']:.2f}%\n")
        f.write(f"  平均精度差值(RGDA-SGD): {stats['mean_diff']:.2f}%\n")
        f.write(f"  最大精度差值(RGDA-SGD): {stats['max_diff']:.2f}%\n")
        f.write(f"  最小精度差值(RGDA-SGD): {stats['min_diff']:.2f}%\n")
        f.write(f"\nAlpha1敏感性分析 (固定α₂={sensitivity_results['alpha2_value']:.2f}):\n")
        f.write(f"  SGD最佳参数: α₁={sensitivity_results['sgd_best']['alpha1']:.3f}, 准确率={sensitivity_results['sgd_best']['accuracy']:.2f}%\n")
        f.write(f"  RGDA最佳参数: α₁={sensitivity_results['rgda_best']['alpha1']:.3f}, 准确率={sensitivity_results['rgda_best']['accuracy']:.2f}%\n")
        f.write(f"  SGD平均准确率: {sensitivity_results['stats']['sgd_mean']:.2f}%\n")
        f.write(f"  RGDA平均准确率: {sensitivity_results['stats']['rgda_mean']:.2f}%\n")
        f.write(f"  最大性能差值: +{sensitivity_results['stats']['max_diff']:.2f}%\n")
    
    print(f"\n所有改进的可视化图表已保存到: {improved_output_dir}")
    print(f"统计摘要已保存到: {stats_path}")

if __name__ == "__main__":
    main()
# %%
