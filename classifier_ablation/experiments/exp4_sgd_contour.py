import os
import sys
# 根据你的环境保留路径设置
sys.path.append('/home/raoxuan/projects/low_rank_rda')
try:
    os.chdir('/home/raoxuan/projects/low_rank_rda')
    print("当前工作目录:", os.getcwd())
except FileNotFoundError:
    print("注意: 目录不存在，请检查路径。当前在:", os.getcwd())

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from contextlib import contextmanager
from sklearn.metrics import balanced_accuracy_score

# ==========================================
# 引入必要的模块 (假设这些文件在你的路径下)
# ==========================================
from classifier.da_classifier_builder import QDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from compensator.gaussian_statistics import GaussianStatistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels

# ==========================================
# 辅助函数
# ==========================================

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    return elapsed

def build_gaussian_statistics(features, labels, cholesky=True):
    features = features.cpu()
    labels = labels.cpu()
    unique_labels = torch.unique(labels)
    
    stats = {}
    for lbl in tqdm(unique_labels, desc="构建高斯统计量"):
        mask = (labels == lbl)
        feats_class = features[mask]
        
        mu = feats_class.mean(0)
        if feats_class.size(0) >= 2:
            cov = torch.cov(feats_class.T) + torch.eye(feats_class.size(1)) * 1e-4
        else:
            cov = torch.eye(feats_class.size(1)) * 1e-4
            
        stats[int(lbl.item())] = GaussianStatistics(mu, cov, cholesky=cholesky)
    
    return stats

def evaluate_classifier_class_wise(classifier, test_features, test_labels, device="cuda"):
    classifier.to(device)
    classifier.eval()
    
    batch_size = 256
    dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets)
            
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    class_wise_acc = balanced_accuracy_score(all_targets, all_preds)
    return class_wise_acc

# ==========================================
# 核心计算逻辑
# ==========================================

def grid_search_sgd_rgda(train_stats, test_features, test_labels,
                          alpha1_range, alpha2_range,
                          sgd_params, device="cuda"):
    alpha1_vals = np.linspace(alpha1_range[0], alpha1_range[1], alpha1_range[2])
    alpha2_vals = np.linspace(alpha2_range[0], alpha2_range[1], alpha2_range[2])
    
    sgd_acc_mat = np.zeros((len(alpha1_vals), len(alpha2_vals)))
    rgda_acc_mat = np.zeros((len(alpha1_vals), len(alpha2_vals)))
    
    total = len(alpha1_vals) * len(alpha2_vals)
    print(f"开始网格搜索 (Metric: Class-wise Mean Accuracy): {len(alpha1_vals)}x{len(alpha2_vals)} = {total} 个组合")
    
    with tqdm(total=total) as pbar:
        for i, a1 in enumerate(alpha1_vals):
            for j, a2 in enumerate(alpha2_vals):
                
                # 1. 评估 SGD
                sgd_builder = SGDClassifierBuilder()
                sgd_clf = sgd_builder.build(train_stats, linear=True,
                                            alpha1=a1, alpha2=a2, alpha3=0.0)
                sgd_acc = evaluate_classifier_class_wise(sgd_clf, test_features, test_labels, device)
                sgd_acc_mat[i, j] = sgd_acc * 100
                
                # 2. 评估 RGDA
                rgda_builder = QDAClassifierBuilder(
                    qda_reg_alpha1=a1, qda_reg_alpha2=a2, qda_reg_alpha3=0.5,
                    low_rank=True,
                    rank=64,
                    device=device
                )
                rgda_clf = rgda_builder.build(train_stats)
                rgda_acc = evaluate_classifier_class_wise(rgda_clf, test_features, test_labels, device)
                rgda_acc_mat[i, j] = rgda_acc * 100
                
                pbar.update(1)
                
    return alpha1_vals, alpha2_vals, sgd_acc_mat, rgda_acc_mat

# ==========================================
# 新增绘图逻辑：合并图表
# ==========================================

def plot_combined_contours(
    alpha1_values, alpha2_values, 
    sgd_acc, rgda_acc, 
    save_path, cmap='viridis'
):
    """
    绘制横向并列的两幅子图 (SGD vs RGDA)，符合IEEE单栏 (3.5 inch) 风格。
    共享 Colorbar，共用 Y 轴刻度。
    """
    # IEEE 标准单栏宽度约为 3.5 英寸 (88mm)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4.0, 1.6), dpi=300, sharey=True)
    
    # 1. 计算全局最大最小值，确保 Colorbar 统一
    vmin = min(sgd_acc.min(), rgda_acc.min())
    vmax = max(sgd_acc.max(), rgda_acc.max())
    
    contour_levels = np.linspace(vmin, vmax, 15) # 统一的等高线层级

    data_list = [
        {'data': sgd_acc, 'title': 'SGD', 'ax': axes[0]},
        {'data': rgda_acc, 'title': 'LR-RGDA (r@64)', 'ax': axes[1]}
    ]
    
    # 获取坐标网格
    A1, A2 = np.meshgrid(alpha1_values, alpha2_values)
    
    main_contour = None # 保存一个对象用于生成colorbar

    for i, item in enumerate(data_list):
        ax = item['ax']
        acc_mat = item['data']
        title = item['title']
        
        # 2. 绘制填充等高线 (使用全局 vmin/vmax)
        cf = ax.contourf(
            A1, A2, acc_mat.T,
            levels=contour_levels, 
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            extend='both' # 处理超出范围的颜色
        )
        
        # 保存第一个图的句柄给colorbar用
        if i == 0:
            main_contour = cf

        # 3. 绘制线条
        lines = ax.contour(
            A1, A2, acc_mat.T,
            levels=contour_levels,
            colors='black',
            linewidths=0.4, 
            alpha=0.7
        )
        
        # 4. 最佳点标记
        midx = np.unravel_index(np.argmax(acc_mat), acc_mat.shape)
        best_a1, best_a2 = alpha1_values[midx[0]], alpha2_values[midx[1]]
        best_acc = acc_mat[midx]
        
        ax.plot(best_a1, best_a2, 'r*', markersize=6, markeredgecolor='white', markeredgewidth=0.3, zorder=10)

        # 5. 标题和字体设置
        ax.set_title(f"{title}: {best_acc:.1f}%", fontsize=9, pad=4)

        # 设置 X 轴标签
        ax.set_xlabel(r'$\alpha_1$', fontsize=9, labelpad=1)
        
        # 刻度设置
        ax.tick_params(axis='both', which='major', labelsize=6, direction='in', length=2, width=0.4)
        
        # 坐标轴范围
        ax.set_xlim(alpha1_values.min(), alpha1_values.max())
        ax.set_ylim(alpha2_values.min(), alpha2_values.max())

        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        
        # 网格
        ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.3)

    # 6. 设置 Y 轴标签 (仅左图显示)
    axes[0].set_ylabel(r'$\alpha_2$', fontsize=9, labelpad=1)
    
    # 7. 调整子图间距
    plt.subplots_adjust(left=0.12, right=0.86, bottom=0.18, top=0.88, wspace=0.1)
    
    # 8. 添加独立的共享 Colorbar
    # 位置参数: [left, bottom, width, height] (相对于整个画布 0-1)
    cbar_ax = fig.add_axes([0.88, 0.18, 0.025, 0.7]) 
    
    # === 修改处: 添加 format='%.1f' ===
    cbar = fig.colorbar(main_contour, cax=cbar_ax, format='%.1f')
    
    # Colorbar 样式优化
    cbar.ax.tick_params(labelsize=5, direction='in', length=1.5, width=0.3)
    cbar.set_label("Accuracy (%)", fontsize=9, labelpad=2)
    cbar.outline.set_linewidth(0.4)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
        print(f"[Combined] 图像已保存至: {save_path}")
    plt.close()

# ==========================================
# 主程序
# ==========================================

def main():
    # ================= 配置区域 =================
    LOAD_FROM_EXISTING = True 
    
    # 实验设置
    model_name = "vit-b-p16" 
    num_shots = 128
    iterations = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 输出路径设置
    base_output_dir = "实验结果保存/分类器对比实验_ClassWise"
    save_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}_comparison")
    os.makedirs(save_dir, exist_ok=True)
    
    # 结果文件路径
    npz_path = os.path.join(save_dir, f"{model_name}_comparison_results.npz")
    
    print(f"运行模型: {model_name}, Iterations: {iterations}")

    # ================= 核心逻辑分支 =================
    
    # 初始化变量
    a1_vals, a2_vals = None, None
    sgd_acc, rgda_acc = None, None

    # 尝试加载逻辑
    data_loaded = False
    if LOAD_FROM_EXISTING:
        if os.path.exists(npz_path):
            try:
                print(f"正在加载保存的结果: {npz_path}")
                data = np.load(npz_path)
                a1_vals = data['alpha1_values']
                a2_vals = data['alpha2_values']
                sgd_acc = data['sgd_classwise_accuracy']
                rgda_acc = data['rgda_classwise_accuracy']
                print("数据加载成功！")
                data_loaded = True
            except Exception as e:
                print(f"加载文件出错: {e}，将重新进行计算。")
        else:
            print(f"未找到文件: {npz_path}，将重新进行计算。")

    if not data_loaded:
        # 1. 加载数据 & 特征提取
        print("正在加载数据和提取特征...")
        dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
        train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
        
        vit = get_vit(model_name)
        adapt_loader = create_adapt_loader(train_subsets)
        vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
        
        train_features, train_labels, _, test_features, test_labels, _ = extract_features_and_labels(
            vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations
        )
        
        # 2. 构建统计量
        train_stats = build_gaussian_statistics(train_features, train_labels)
        
        # 3. 运行网格搜索
        alpha1_range = (0.0, 1.0, 11) 
        alpha2_range = (0.0, 2.0, 15)
        
        sgd_params = {'epochs': 2, 'lr': 1e-3, 'linear': True}
        
        a1_vals, a2_vals, sgd_acc, rgda_acc = grid_search_sgd_rgda(
            train_stats, test_features, test_labels,
            alpha1_range, alpha2_range, sgd_params, device=device
        )
        
        # 4. 保存原始数据
        np.savez(
            npz_path,
            alpha1_values=a1_vals,
            alpha2_values=a2_vals,
            sgd_classwise_accuracy=sgd_acc,
            rgda_classwise_accuracy=rgda_acc
        )
        print(f"计算完成，数据已保存至: {npz_path}")

    # ================= 绘图逻辑 (合并在一张图) =================
    
    print("开始绘制合并等高线图...")
    
    plot_combined_contours(
        a1_vals, a2_vals, 
        sgd_acc, rgda_acc,
        save_path=os.path.join(save_dir, f"contour_Combined_{model_name}_classwise.png"),
        cmap='Spectral' # 或者使用 'viridis', 'plasma' 等
    )
    
    print("所有任务完成。")

if __name__ == '__main__':
    main()