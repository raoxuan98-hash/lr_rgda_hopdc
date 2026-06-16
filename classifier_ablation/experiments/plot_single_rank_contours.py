import os
import sys
# 根据你的环境保留路径设置
sys.path.append('/home/raoxuan/projects/low_rank_rda')
try:
    os.chdir('/home/raoxuan/projects/low_rank_rda')
    print("当前工作目录:", os.getcwd())
except FileNotFoundError:
    print("注意: 目录不存在，请检查路径。当前在:", os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_rank_data(model_name, iterations, rank, base_dir="实验结果保存/分类器消融实验"):
    """
    加载指定模型、迭代次数和rank的实验数据
    """
    data_dir = os.path.join(base_dir, f"{model_name}_iter{iterations}_rank{rank}")
    data_file = os.path.join(data_dir, f"{model_name}_rank{rank}_results.npz")
    
    if not os.path.exists(data_file):
        print(f"警告: 数据文件不存在: {data_file}")
        return None, None, None
    
    data = np.load(data_file)
    alpha1_values = data['alpha1_values']
    alpha2_values = data['alpha2_values']
    accuracy_matrix = data['accuracy_matrix']
    
    print(f"已加载 {model_name} rank {rank} 的数据")
    return alpha1_values, alpha2_values, accuracy_matrix

def plot_single_rank_contour(
    model_name, iterations, target_rank, cmap='viridis',
    base_dir="实验结果保存/分类器消融实验", save_path=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. 加载数据 (只加载目标Rank)
    alpha1_values, alpha2_values, accuracy_matrix = load_rank_data(
        model_name, iterations, target_rank, base_dir
    )

    if alpha1_values is None:
        print(f"未找到 Rank {target_rank} 的数据，无法绘图。")
        return

    # 2. 创建单张图
    # figsize 设置为长宽比接近 1:1 或稍微宽一点，适合单张展示
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=500)

    # 颜色等级
    contour_levels = 15
    
    # 获取数据范围用于绘图
    vmin = accuracy_matrix.min()
    vmax = accuracy_matrix.max()

    A1, A2 = np.meshgrid(alpha1_values, alpha2_values)

    # 3. 绘制填充等高线
    cf = ax.contourf(
        A1, A2, accuracy_matrix.T,
        levels=contour_levels, cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    
    # 4. 绘制线条
    lines = ax.contour(
        A1, A2, accuracy_matrix.T,
        levels=contour_levels,
        colors='black',
        linewidths=0.4,
        alpha=0.7
    )

    # 5. 添加线上的数字标签
    ax.clabel(
        lines,
        levels=lines.levels[::1],
        inline=True,
        fontsize=6,
        fmt="%.1f" 
    )

    # 6. 标记最佳点
    midx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_a1, best_a2 = alpha1_values[midx[0]], alpha2_values[midx[1]]
    best_acc = accuracy_matrix[midx]

    ax.plot(best_a1, best_a2, 'r*', markersize=9, label='Best')

    # 7. 设置标题和坐标轴
    ax.set_title(
        f"Rank {target_rank}\nBest=({best_a1:.2f}, {best_a2:.2f}) {best_acc:.1f}%",
        fontsize=9, pad=5
    )

    ax.set_xlabel(r'$\alpha_1$', fontsize=9)
    ax.set_ylabel(r'$\alpha_2$', fontsize=9)
    
    # 刻度设置 (保持原有的刻度逻辑)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0.0, 1.0, 2.0, 3.0])

    # 美化
    ax.tick_params(labelsize=8, direction='in')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.25)

    # 8. 添加 Colorbar
    # 对单张图，fraction 和 pad 需要微调以保证美观
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='绘制指定Rank (默认为64) 的等高线图')
    parser.add_argument('--model', type=str, default='vit-b-p16',
                        help='模型名称')
    parser.add_argument('--iterations', type=int, default=0,
                        help='迭代次数')
    # 修改：默认只包含 64
    parser.add_argument('--rank', type=int, default=64,
                        help='要绘制的rank值 (默认: 64)')
    parser.add_argument('--cmap', type=str, default='Spectral',
                        help='颜色映射')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    print(f"绘制模型 {args.model} 的 Rank {args.rank} 等高线图")
    print(f"迭代次数: {args.iterations}")
    
    # 自动生成输出路径
    if args.output is None:
        base_output_dir = "实验结果保存/分类器消融实验"
        model_output_dir = os.path.join(base_output_dir, f"{args.model}_iter{args.iterations}_single_rank")
        os.makedirs(model_output_dir, exist_ok=True)
        # 文件名包含 rank 64
        args.output = os.path.join(model_output_dir, f"contour_{args.model}_iter{args.iterations}_rank{args.rank}.png")
    
    plot_single_rank_contour(
        args.model, args.iterations, args.rank, 
        cmap=args.cmap, save_path=args.output
    )

if __name__ == '__main__':
    main()