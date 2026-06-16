import os
import sys
sys.path.append('/home/raoxuan/projects/low_rank_rda')
os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())

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

def plot_multi_rank_contours(
    model_name, iterations, ranks, cmap='viridis',
    base_dir="实验结果保存/分类器消融实验", save_path=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    # 加载数据
    all_data = []
    for rank in ranks:
        alpha1_values, alpha2_values, accuracy_matrix = load_rank_data(
            model_name, iterations, rank, base_dir
        )
        if alpha1_values is not None:
            all_data.append((rank, alpha1_values, alpha2_values, accuracy_matrix))

    if len(all_data) == 0:
        print("没有找到数据")
        return

    # 全局 normalize 范围
    all_acc = np.concatenate([x[3].flatten() for x in all_data])
    global_vmin = all_acc.min()
    global_vmax = all_acc.max()

    # 创建图（美观版）：更宽松的 spacing
    fig, axes = plt.subplots(
        1, 4, figsize=(8.0, 1.5), dpi=500,
        gridspec_kw={'wspace': 0.25, 'hspace': 0.4})

    # 颜色等级减少到 10（视觉更干净）
    contour_levels = 15

    for i, (rank, a1, a2, acc) in enumerate(all_data[:4]):
        ax = axes[i]
        A1, A2 = np.meshgrid(a1, a2)
        # filled contour
        cf = ax.contourf(
            A1, A2, acc.T,
            levels=contour_levels, cmap=cmap,
            vmin=global_vmin, vmax=global_vmax)
        

        lines = ax.contour(
            A1, A2, acc.T,
            levels=contour_levels,
            colors='black',
            linewidths=0.4,
            alpha=0.7)

        ax.clabel(
            lines,
            levels=lines.levels[::1],     # 关键：隔 3 个 level
            inline=True,
            fontsize=5,
            fmt="%.1f"                    # 标签格式，例如 85.3%
        )

        # 找最佳点
        midx = np.unravel_index(np.argmax(acc), acc.shape)
        best_a1, best_a2 = a1[midx[0]], a2[midx[1]]
        best_acc = acc[midx]

        ax.plot(best_a1, best_a2, 'r*', markersize=7)

        # 小标题放在上方更紧凑
        ax.set_title(
            f"Rank {rank}\nBest=({best_a1:.2f}, {best_a2:.2f}) {best_acc:.1f}%",
            fontsize=7, pad=2
        )

        ax.set_xlabel(r'$\alpha_1$', fontsize=8)
        ax.set_xticks([0, 0.5, 1.0])  # <--- 新增这一行
        ax.set_yticks([0.0, 1.0, 2.0, 3.0])
        if i == 0:
            ax.set_ylabel(r'$\alpha_2$', fontsize=8)

        # ticks 美化
        ax.tick_params(labelsize=7, direction='in')

        # 去掉上右框线（学术期刊风）
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Grid 更淡
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.25)

    # # 主标题
    # fig.suptitle(
    #     f"{model_name} Performance Surface (iter={iterations})",
    #     fontsize=10, y=1.04
    # )

    # Colorbar 布局更紧致
    # cbar = fig.colorbar(
    #     cf, ax=axes, orientation='horizontal',
    #     fraction=0.05, pad=0.15, aspect=30
    # )

    cbar = fig.colorbar(
        cf, ax=axes,
        fraction=0.05, pad=0.04, aspect=30
    )

    cbar.set_label("Average accuracy (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='绘制同一模型不同rank的等高线图')
    parser.add_argument('--model', type=str, default='vit-b-p16-mocov3',
                        help='模型名称 (vit-b-p16, vit-b-p16-clip, vit-b-p16-mocov3, vit-b-p16-dino)')
    parser.add_argument('--iterations', type=int, default=0,
                        help='迭代次数')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 8, 32, 64],
                        help='要绘制的rank值列表')
    parser.add_argument('--cmap', type=str, default='Spectral',
                        help='颜色映射')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    print(f"绘制模型 {args.model} 的多rank等高线图")
    print(f"迭代次数: {args.iterations}")
    print(f"Rank值: {args.ranks}")
    print(f"颜色映射: {args.cmap}")
    
    # 如果没有指定输出路径，自动生成
    if args.output is None:
        base_output_dir = "实验结果保存/分类器消融实验"
        model_output_dir = os.path.join(base_output_dir, f"{args.model}_iter{args.iterations}_multi_rank")
        os.makedirs(model_output_dir, exist_ok=True)
        args.output = os.path.join(model_output_dir, f"multi_rank_contour_{args.model}_iter{args.iterations}.png")
    
    plot_multi_rank_contours(
        args.model, args.iterations, args.ranks, 
        cmap=args.cmap, save_path=args.output
    )

if __name__ == '__main__':
    main()