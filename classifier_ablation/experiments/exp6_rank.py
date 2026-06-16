import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns 
import torch
import gc 
import random

# 根据你的环境保留路径设置
sys.path.append('/home/raoxuan/projects/low_rank_rda')
try:
    os.chdir('/home/raoxuan/projects/low_rank_rda')
    print("当前工作目录:", os.getcwd())
except FileNotFoundError:
    print("注意: 目录不存在，请检查路径。当前在:", os.getcwd())

# 引入项目特定的模块
from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels
from classifier.da_classifier_builder import QDAClassifierBuilder

# === 样式配置 ===
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
COLORS = {
    "QDA": "#0779DC",
    "SGD-linear": "#F50202",
    "SGD-nonlinear": "#3C8D86",
    "NCM": "#6FA8DC",
    "LDA": "#A27CCD"
}

def set_seed(seed):
    """固定随机种子以确保单次运行的可复现性，但不同run使用不同seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_classifier_fixed_alpha_varying_rank(
        rank, alpha1, alpha2, alpha3, 
        stats, features, targets, dataset_ids,
        device="cuda", batch_size=1024):
    """
    在固定Alpha、指定Rank下评估 QDA 分类器
    """
    builder = QDAClassifierBuilder(
        qda_reg_alpha1=alpha1,
        qda_reg_alpha2=alpha2,
        qda_reg_alpha3=alpha3,
        low_rank=True,
        rank=rank,
        device=device)
    
    classifier = builder.build(stats)
    classifier.to(device)
    classifier.eval()
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(features, targets, dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].cuda()
            all_targets.append(batch[1])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Class-wise Accuracy
    classes = torch.unique(all_targets)
    per_class_accuracies = []
    
    for cls in classes:
        cls_mask = (all_targets == cls)
        cls_preds = all_predictions[cls_mask]
        cls_targets = all_targets[cls_mask]
        
        if len(cls_targets) > 0:
            acc = (cls_preds == cls_targets).float().mean().item()
            per_class_accuracies.append(acc)
    
    if len(per_class_accuracies) > 0:
        accuracy = sum(per_class_accuracies) / len(per_class_accuracies)
    else:
        accuracy = 0.0
        
    return accuracy

def run_rank_ablation_experiment(
        rank_values, 
        fixed_alpha1, fixed_alpha2, fixed_alpha3,
        train_stats, test_features, test_labels, test_dataset_ids,
        device="cuda"):
    
    qda_accuracies = []
    for r in rank_values:
        acc = evaluate_classifier_fixed_alpha_varying_rank(
            r, fixed_alpha1, fixed_alpha2, fixed_alpha3,
            train_stats, test_features, test_labels, test_dataset_ids,
            device=device
        )
        qda_accuracies.append(acc)
        
    return qda_accuracies

def plot_rank_comparison_grid_pretty(results_data, rank_values, save_path, figsize=(8.5, 2.1)):
    """
    绘制 Rank 消融实验结果 (论文级美化版)
    results_data: dict, key=model_name, value=mean_accuracies (list)
    """
    
    # 确定子图数量 (根据模型数量)
    num_models = len(results_data)
    # 动态调整 figsize，如果只有1个模型
    if num_models < 2: figsize = (3.5, 2.5)

    fig, axes = plt.subplots(1, num_models, figsize=figsize)
    
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    ranks = np.array(rank_values)

    for i, (model_name, accs) in enumerate(results_data.items()):
        ax = axes[i]
        qda_pct = np.array(accs) * 100
        
        # 绘制主曲线
        ax.plot(ranks, qda_pct, 
                marker='*',          
                markersize=6,       
                linewidth=1.4, 
                label="LR-RGDA", 
                color=COLORS["QDA"], 
                linestyle='-.')
        
        # 标注最高点
        best_idx = np.argmax(qda_pct)
        best_rank = ranks[best_idx]
        best_acc = qda_pct[best_idx]
        
        # 在最高点画个红色圆圈强调
        ax.plot(best_rank, best_acc, 'o', markerfacecolor='none', markeredgecolor='red', markersize=8, markeredgewidth=1.5)

        # 设置标题 (映射常用名称)
        if "clip" in model_name:
            title = "ViT/B-CLIP"
        elif "mocov3" in model_name:
            title = "ViT/B-MoCoV3"
        elif "dino" in model_name:
            title = "ViT/B-DINO"
        else:
            title = "ViT/B-Sup21K" # 默认 vit-b-p16
            
        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_xlabel("Rank", fontsize=9)
        
        # 仅第一个子图显示 Y 轴标签
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)
        
        # --- 坐标轴调整 ---
        ax.set_xscale('log', base=2)
        ax.set_xticks(ranks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        
        # 动态调整 Y 轴范围
        y_min, y_max = min(qda_pct), max(qda_pct)
        y_range = y_max - y_min
        if y_range == 0: y_range = 1.0
        y_min_adj = y_min - 0.15 * y_range
        y_max_adj = y_max + 0.15 * y_range
        
        ax.set_ylim(y_min_adj, y_max_adj)
        
        # 设置 Y 轴刻度
        y_ticks = np.linspace(y_min_adj, y_max_adj, 5)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.68)
        ax.tick_params(axis='both', labelsize=8, width=0.6)

    # 仅在第一个图显示图例
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower right', fontsize=8, frameon=True, 
                   fancybox=True, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" 最终合并图表已保存至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行Rank消融实验 (Paper Style) - 3次平均')
    parser.add_argument('--gpu', type=str, default='0', help='GPU编号')
    parser.add_argument('--iterations', type=int, default=500, help='迭代次数')
    parser.add_argument('--num_shots', type=int, default=128, help='样本数量')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === 参数设置 ===
    # [修改处] 设置为 True 以直接加载 CSV 绘图，不进行计算
    LOAD_EXISTING_RESULTS = True
    FIXED_ALPHA1 = 0.2
    FIXED_ALPHA2 = 2.0
    FIXED_ALPHA3 = 0.5
    RANK_VALUES = [1, 2, 4, 8, 16, 32, 64]
    
    # [修改处] 添加了 'vit-b-p16-dino' 架构
    MODELS_TO_TEST = ['vit-b-p16', 'vit-b-p16-mocov3', 'vit-b-p16-clip', 'vit-b-p16-dino'] 
    
    # [修改处] 设置重复次数
    NUM_REPEATS = 1 
    
    base_output_dir = "实验结果保存/Rank影响研究_PaperStyle_Average"
    print(f"测试 Rank: {RANK_VALUES}")
    print(f"重复次数: {NUM_REPEATS}")
    
    # === 数据收集容器 ===
    # 结构: {model_name: [mean_acc_for_rank1, mean_acc_for_rank2, ...]}
    all_models_mean_results = {} 

    for model_name in MODELS_TO_TEST:
        print(f"\n>>> 正在处理架构: {model_name} ...")
        model_output_dir = os.path.join(base_output_dir, f"{model_name}_iter{args.iterations}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 存储该模型所有run的结果: shape [NUM_REPEATS, len(RANK_VALUES)]
        model_runs_accs = []
        
        for run_idx in range(NUM_REPEATS):
            csv_path = os.path.join(model_output_dir, f"{model_name}_rank_results_run{run_idx}.csv")
            
            run_accs = []
            
            if LOAD_EXISTING_RESULTS and os.path.exists(csv_path):
                print(f"  [Run {run_idx}] 正在读取: {csv_path}")
                try:
                    rank_acc_map = {}
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()[1:] # 跳过 header
                        for line in lines:
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                r_val = int(parts[0])
                                acc_val = float(parts[1])
                                rank_acc_map[r_val] = acc_val
                    
                    # 对齐 Rank 列表
                    run_accs = [rank_acc_map.get(r, 0.0) for r in RANK_VALUES]
                    model_runs_accs.append(run_accs)
                    
                except Exception as e:
                    print(f"  [ERROR] 读取 Run {run_idx} 出错: {e}")
            
            elif not LOAD_EXISTING_RESULTS:
                print(f"  [Run {run_idx}] 开始计算 (Seed={run_idx})...")
                try:
                    # 1. 设置不同的随机种子
                    current_seed = 42 + run_idx
                    set_seed(current_seed)
                    
                    # 2. 加载资源 (每次重新加载以确保 split 不同)
                    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=args.num_shots, model_name=model_name)
                    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
                    vit = get_vit(vit_name=model_name)
                    
                    if args.iterations > 0:
                        adapt_loader = create_adapt_loader(train_subsets, batch_size=48)
                        vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=args.iterations)
                    
                    # 3. 提取特征
                    train_features, train_labels, train_ids, test_features, test_labels, test_ids = extract_features_and_labels(
                        vit, dataset, train_loader, test_loader, model_name, num_shots=args.num_shots, iterations=args.iterations)
                    
                    train_ids = torch.tensor(train_ids)
                    test_ids = torch.tensor(test_ids)
                    train_stats = build_gaussian_statistics(train_features, train_labels)
                    
                    # 4. 运行 Rank 实验
                    run_accs = run_rank_ablation_experiment(
                        RANK_VALUES, FIXED_ALPHA1, FIXED_ALPHA2, FIXED_ALPHA3,
                        train_stats, test_features, test_labels, test_ids, device=device
                    )
                    
                    model_runs_accs.append(run_accs)
                    
                    # 5. 保存单次 Run 的 CSV
                    with open(csv_path, 'w') as f:
                        f.write("rank,qda_acc,model,run\n")
                        for r, acc in zip(RANK_VALUES, run_accs):
                            f.write(f"{r},{acc},{model_name},{run_idx}\n")
                    print(f"  Run {run_idx} 完成并保存。")
                    
                    # 清理显存
                    del vit, train_features, test_features, train_stats
                    torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    print(f"  [ERROR] Run {run_idx} 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                 print(f"  [Run {run_idx}] 文件缺失且模式为 Load，跳过。")

        # === 计算 3 次平均值 ===
        if len(model_runs_accs) > 0:
            # 转换为 numpy 数组: (Num_Runs, Num_Ranks)
            runs_array = np.array(model_runs_accs)
            # 沿着 Runs 维度求平均 -> (Num_Ranks,)
            mean_accs = np.mean(runs_array, axis=0).tolist()
            
            all_models_mean_results[model_name] = mean_accs
            print(f"  {model_name}: 已计算 {len(model_runs_accs)} 次运行的平均值。")
        else:
            print(f"  {model_name}: 无有效数据，跳过。")

    # === 绘图 ===
    print("\n>>> 开始绘制合并图表...")
    combined_save_dir = os.path.join(base_output_dir, "Combined_Plots")
    os.makedirs(combined_save_dir, exist_ok=True)
    
    save_img_path = os.path.join(combined_save_dir, f"rank_ablation_comparison_{args.iterations}.png")
    
    if len(all_models_mean_results) > 0:
        # [修改处] figsize 宽度设为 8.5 以适应 4 个子图
        plot_rank_comparison_grid_pretty(
            all_models_mean_results, 
            RANK_VALUES, 
            save_img_path,
            figsize=(8.5, 2.1) 
        )
    else:
        print("没有收集到有效结果，跳过绘图。")

    print("\n所有任务完成。")
