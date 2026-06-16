# -*- coding: utf-8 -*-
"""
特征奇异值分析实验
基于exp2_alpha_constraint_parallel.py，但删除分类器评估部分，专注于特征分析
"""
import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
import matplotlib as mpl
import sys
from tqdm import tqdm
os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())
sys.path.append('/home/raoxuan/projects/low_rank_rda')
import seaborn as sns

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 支持中文
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels, infer_dataset_ids_from_labels
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

def extract_full_dataset_features(model, combined_loader, dataset_manager, 
                                model_name, num_shots, iterations=None, cache_dir="cached_data/classifier_ablation"):
    """
    提取完整合并数据集的特征（训练集+测试集）
    
    Args:
        model: 特征提取模型
        combined_loader: 合并的数据加载器
        dataset_manager: 数据集管理器
        model_name: 模型名称
        num_shots: 少样本数量
        iterations: 迭代次数
        cache_dir: 缓存目录
    
    Returns:
        features: 特征张量
        labels: 标签张量
        dataset_ids: 数据集ID张量
    """
    # 创建缓存键
    if iterations is not None:
        cache_key = f"{model_name}_{num_shots}_iter{iterations}_combined_features_cache"
    else:
        cache_key = f"{model_name}_{num_shots}_combined_features_cache"
    
    cache_path = os.path.join(cache_dir, cache_key)
    cache_file = f"{cache_path}_full.pt"
    
    # 检查缓存
    if os.path.exists(cache_file):
        print(f"检测到合并特征缓存文件: {cache_file}")
        cached_data = torch.load(cache_file)
        return cached_data['features'], cached_data['labels'], cached_data['dataset_ids']
    
    print("未检测到合并特征缓存，开始提取完整数据集特征...")
    
    model.eval()
    device = "cuda"
    model.to(device)
    
    # 提取特征
    features_list = []
    labels_list = []
    
    print("从合并数据集提取特征...")
    with torch.no_grad():
        for batch in tqdm(combined_loader):
            inputs = batch[0].to(device)
            labels = batch[1]
            feats = model(inputs).cpu()
            features_list.append(feats)
            labels_list.append(labels.cpu())
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    dataset_ids = torch.tensor(infer_dataset_ids_from_labels(labels, dataset_manager))
    
    # 保存缓存
    print("保存合并特征缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    
    torch.save({
        'features': features,
        'labels': labels, 
        'dataset_ids': dataset_ids
    }, cache_file)
    
    print(f"合并特征缓存已保存到: {cache_file}")
    
    return features, labels, dataset_ids
def compute_class_wise_singular_values(features, labels, dataset_ids, device="cpu"):
    """
    计算每个类别的特征奇异值
    
    Args:
        features: 特征张量 [N, D]
        labels: 标签张量 [N]
        dataset_ids: 数据集ID张量 [N] 
        device: 计算设备
    
    Returns:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
    """
    # 移动数据到指定设备
    features = features.to(device)
    labels = labels.to(device)
    dataset_ids = dataset_ids.to(device)
    
    # 获取唯一的数据集ID和类别ID
    unique_dataset_ids = torch.unique(dataset_ids)
    all_singular_values = []
    class_ids = []
    num_samples_per_class = []
    
    print(f"发现 {len(unique_dataset_ids)} 个数据集")
    
    for dataset_id in unique_dataset_ids:
        # 筛选当前数据集的数据
        dataset_mask = (dataset_ids == dataset_id)
        dataset_features = features[dataset_mask]
        dataset_labels = labels[dataset_mask]
        
        # 获取该数据集的独特类别
        unique_labels = torch.unique(dataset_labels)
        
        for class_label in unique_labels:
            # 筛选当前类别的数据
            class_mask = (dataset_labels == class_label)
            class_features = dataset_features[class_mask]
            
            # 确保至少有2个样本才能计算协方差矩阵
            if len(class_features) < 2:
                print(f"警告: 数据集 {dataset_id}, 类别 {class_label} 只有 {len(class_features)} 个样本，跳过")
                continue
            
            # 使用torch.svd_lowrank进行奇异值分解，rank=128
            class_features_tensor = class_features  # 已经是torch tensor
            
            # 计算协方差矩阵
            # 先减去均值
            mean_features = torch.mean(class_features_tensor, dim=0, keepdim=True)
            class_features_centered = class_features_tensor - mean_features
            
            # 计算协方差矩阵 (D x D)
            cov_matrix = torch.cov(class_features_centered.T)
            
            # 使用torch.svd_lowrank进行奇异值分解，rank=128
            U, s, Vt = torch.svd_lowrank(cov_matrix, q=128)
            
            # 奇异值总是非负的，按降序排列
            singular_values = s.cpu().numpy()
            
            # 保存结果
            all_singular_values.append(singular_values)
            class_ids.append(f"D{dataset_id.item()}_C{class_label.item()}")
            num_samples_per_class.append(len(class_features))
            
            print(f"数据集 {dataset_id.item()}, 类别 {class_label.item()}: "
                  f"{len(class_features)} 个样本, {len(singular_values)} 个奇异值")
    
    return all_singular_values, class_ids, num_samples_per_class

def plot_singular_value_curves(all_singular_values, class_ids, num_samples_per_class, save_path=None, show_legend=True):
    
    plt.figure(figsize=(3.5, 2.3))  # IEEE单栏图片标准尺寸：3.5英寸宽，2.6英寸高

    max_singular_values = max(len(sv) for sv in all_singular_values)
    
    # 计算每个类别在第64维度的累积比例
    cumulative_64_ratios = []
    for singular_values in all_singular_values:
        sorted_singular_values = np.sort(singular_values)[::-1]  # 降序排列
        total_sum = np.sum(sorted_singular_values)
        
        # 计算前64个奇异值的累积比例
        k_64 = min(64, len(sorted_singular_values))
        cumulative_sum_64 = np.sum(sorted_singular_values[:k_64])
        cumulative_ratio_64 = cumulative_sum_64 / total_sum if total_sum > 0 else 0
        cumulative_64_ratios.append(cumulative_ratio_64)
    
    # 统计累积比例>0.9的类别总数
    high_quality_classes = sum(ratio > 0.9 for ratio in cumulative_64_ratios)
    print(f"64维度累积比例 > 0.9 的类别总数: {high_quality_classes}/{len(class_ids)}")
    
    # 根据累积比例的最大值和最小值进行归一化
    min_ratio = min(cumulative_64_ratios)
    max_ratio = max(cumulative_64_ratios)
    
    # 创建归一化的颜色映射 - 使用viridis，根据归一化的累积比例值赋予颜色
    normalized_ratios = [(ratio - min_ratio) / (max_ratio - min_ratio) if max_ratio > min_ratio else 0.5 
                        for ratio in cumulative_64_ratios]
    colors = [cm.get_cmap('viridis_r')(norm_ratio) for norm_ratio in normalized_ratios]
    
    # 为每个类别绘制曲线
    for i, (singular_values, class_id, n_samples) in enumerate(zip(all_singular_values, class_ids, num_samples_per_class)):
        # 计算累计比例 - 从第1个到第k个奇异值之和占所有奇异值总和的比例
        cumulative_ratios = []
        
        # 计算前k个最大奇异值的累计比例
        sorted_singular_values = np.sort(singular_values)[::-1]  # 降序排列
        cumulative_sum = 0
        total_sum = np.sum(sorted_singular_values)
        
        for k in range(min(80, len(sorted_singular_values))):  # 限制到第80维度
            cumulative_sum += sorted_singular_values[k]
            cumulative_ratio = cumulative_sum / total_sum if total_sum > 0 else 0
            cumulative_ratios.append(cumulative_ratio)
        
        # 维度编号 (从1到80)
        dimensions = np.arange(1, len(cumulative_ratios) + 1)
        
        # 绘制曲线 - 不使用对数尺度
        plt.plot(dimensions, cumulative_ratios,
                color=colors[i], linewidth=1.0, alpha=0.6,
                label=f'{class_id} (n={n_samples}, r64={cumulative_64_ratios[i]:.3f})')
    
    # 设置标签和标题
    plt.xlabel('Singular index', fontsize=7)
    plt.ylabel('Cumulative ratio of variance', fontsize=8)
    plt.ylim(0, 1.02)  # 比例范围0到1
    plt.xlim(1, 80)    # 限制到80维度
    
    # 调整x轴和y轴刻度
    plt.xticks([0, 20, 40, 60, 80], fontsize=7)
    plt.yticks(fontsize=8)  # 设置y轴刻度大小为8pt
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加颜色条显示归一化后的累积比例与颜色的对应关系
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r,
                              norm=mpl.colors.Normalize(vmin=0.88, vmax=1.0))  # 设置固定范围
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Cumulative ratio at 64-dim', fontsize=7)
    
    # 设置颜色条刻度：0.88到1.0，每0.03一个刻度，并设置刻度大小为8pt
    cbar_ticks = np.arange(0.88, 1.01, 0.03)  # 0.88, 0.91, 0.94, 0.97, 1.00
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=8)  # 设置颜色条刻度大小为8pt
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"奇异值曲线图已保存到: {save_path}")
    
    plt.show()
    
    # 打印累积比例的统计信息
    print(f"64维度累积比例范围: {min_ratio:.3f} - {max_ratio:.3f}")
    print(f"64维度累积比例平均值: {np.mean(cumulative_64_ratios):.3f}")


def analyze_singular_value_statistics(all_singular_values, class_ids, num_samples_per_class, save_path=None):
    """
    分析奇异值统计信息
    
    Args:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
        save_path: 保存路径
    """
    print("\n" + "="*60)
    print("奇异值统计分析")
    print("="*60)
    
    # 收集统计信息
    total_singular_values = []
    max_singular_values = []
    mean_singular_values = []
    std_singular_values = []
    effective_rank_values = []  # 有效秩 (奇异值平方和的平方根除以最大奇异值)
    
    for i, (singular_values, class_id, n_samples) in enumerate(zip(all_singular_values, class_ids, num_samples_per_class)):
        total_sv = len(singular_values)
        max_sv = np.max(singular_values)
        mean_sv = np.mean(singular_values)
        std_sv = np.std(singular_values)
        
        # 计算有效秩 (trace of covariance matrix / max eigenvalue)
        eigenvals_sq_sum = np.sum(singular_values**2)
        effective_rank = eigenvals_sq_sum / (max_sv**2) if max_sv > 0 else 0
        
        total_singular_values.append(total_sv)
        max_singular_values.append(max_sv)
        mean_singular_values.append(mean_sv)
        std_singular_values.append(std_sv)
        effective_rank_values.append(effective_rank)
        
        print(f"{class_id}: n={n_samples:4d}, "
              f"dims={total_sv:3d}, "
              f"max_sv={max_sv:8.4f}, "
              f"mean_sv={mean_sv:6.4f}, "
              f"std_sv={std_sv:6.4f}, "
              f"eff_rank={effective_rank:6.2f}")
    
    # 全局统计
    print(f"\n全局统计:")
    print(f"类别数量: {len(all_singular_values)}")
    print(f"平均奇异值数量: {np.mean(total_singular_values):.2f}")
    print(f"最大奇异值范围: {np.min(max_singular_values):.4f} - {np.max(max_singular_values):.4f}")
    print(f"平均有效秩: {np.mean(effective_rank_values):.2f} ± {np.std(effective_rank_values):.2f}")
    print(f"平均样本数量: {np.mean(num_samples_per_class):.2f} ± {np.std(num_samples_per_class):.2f}")
    
    # 保存统计信息
    if save_path:
        stats_path = save_path.replace('.png', '_statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("奇异值统计分析报告\n")
            f.write("="*60 + "\n\n")
            
            for i, (singular_values, class_id, n_samples) in enumerate(zip(all_singular_values, class_ids, num_samples_per_class)):
                f.write(f"{class_id}: n={n_samples:4d}, "
                       f"dims={total_singular_values[i]:3d}, "
                       f"max_sv={max_singular_values[i]:8.4f}, "
                       f"mean_sv={mean_singular_values[i]:6.4f}, "
                       f"std_sv={std_singular_values[i]:6.4f}, "
                       f"eff_rank={effective_rank_values[i]:6.2f}\n")
            
            f.write(f"\n全局统计:\n")
            f.write(f"类别数量: {len(all_singular_values)}\n")
            f.write(f"平均奇异值数量: {np.mean(total_singular_values):.2f}\n")
            f.write(f"最大奇异值范围: {np.min(max_singular_values):.4f} - {np.max(max_singular_values):.4f}\n")
            f.write(f"平均有效秩: {np.mean(effective_rank_values):.2f} ± {np.std(effective_rank_values):.2f}\n")
            f.write(f"平均样本数量: {np.mean(num_samples_per_class):.2f} ± {np.std(num_samples_per_class):.2f}\n")
        
        print(f"统计信息已保存到: {stats_path}")

def plot_violin_64dim_distribution(all_singular_values, class_ids, num_samples_per_class, save_path=None):
    """
    使用 seaborn 小提琴图 可视化不同数据集在奇异值累计占 0.85 & 0.95 比例时的秩分布（双分布）
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # 数据集映射
    dataset_name_mapping = {
        '0': 'CIFAR-100',
        '1': 'ImageNet-R',
        '2': 'Cars-196',
        '3': 'CUB-200',
        '4': 'Caltech-101',
        '5': 'Flower-102',
        '6': 'Food-101'
    }

    # 准备数据
    rows = []
    for singular_values, class_id in zip(all_singular_values, class_ids):
        dataset_id = class_id.split('_')[0][1:]  # "D0_C3" → "0"
        name = dataset_name_mapping.get(dataset_id, f"Dataset {dataset_id}")

        sv = np.sort(singular_values)[::-1]
        total = sv.sum()
        cumulative = np.cumsum(sv)

        # 计算 85% 和 95% 秩
        rank_85 = np.argmax(cumulative >= 0.85 * total) + 1
        rank_95 = np.argmax(cumulative >= 0.95 * total) + 1

        rows.append([name, rank_85, "85%"])
        rows.append([name, rank_95, "95%"])

    df = pd.DataFrame(rows, columns=["Dataset", "Rank", "Ratio"])

    # ==============================
    # 图形绘制：双小提琴图
    # ==============================
    sns.set_style("whitegrid")

    plt.figure(figsize=(4.0, 2.6))

    # 颜色：蓝色 (85%) / 红色 (95%)
    palette = {"85%": "#1f77b4", "95%": "#d62728"}

    sns.violinplot(
        x="Dataset", y="Rank", hue="Ratio",
        data=df, split=False, cut=0, linewidth=1,
        scale="width",
        inner="quart",
        palette=palette
    )

    # 叠加散点
    sns.stripplot(
        x='Dataset', y='Rank', hue="Ratio",
        data=df, dodge=True,
        color='black', size=0.8, jitter=True, alpha=0.3, legend=False
    )

    # 处理图例：避免重复 legend
    plt.legend(title="Cumulative ratio", fontsize=7, title_fontsize=7)
    # plt.xlabel("Dataset", fontsize=7)
    plt.ylabel("Rank index", fontsize=8)
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"[Saved] violin figure → {save_path}")

    plt.show()

    # ==============================
    # 打印统计信息
    # ==============================
    print("\n累计比例秩分布统计 (85% & 95%):")
    print("=" * 50)
    for ratio in ["85%", "95%"]:
        print(f"—— Ratio {ratio} ——")
        for dataset in df["Dataset"].unique():
            g = df[(df["Dataset"] == dataset) & (df["Ratio"] == ratio)]["Rank"]
            print(f"{dataset}:")
            print(f"  类别数: {len(g)}")
            print(f"  平均值: {g.mean():.2f}")
            print(f"  中位数: {g.median():.2f}")
            print(f"  标准差: {g.std():.2f}")
            print(f"  25%分位数: {g.quantile(0.25):.2f}")
            print(f"  75%分位数: {g.quantile(0.75):.2f}")
            print()



def save_singular_values_data(all_singular_values, class_ids, num_samples_per_class, save_dir):
    """
    保存奇异值数据
    
    Args:
        all_singular_values: 所有类别的奇异值列表
        class_ids: 类别ID列表
        num_samples_per_class: 每个类别的样本数量
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为npz格式
    npz_path = os.path.join(save_dir, "singular_values_data.npz")
    np.savez_compressed(npz_path, 
                       singular_values=all_singular_values,
                       class_ids=class_ids,
                       num_samples_per_class=num_samples_per_class)
    
    # 保存为CSV格式便于查看
    csv_path = os.path.join(save_dir, "singular_values_summary.csv")
    
    # 准备CSV数据
    max_dims = max(len(sv) for sv in all_singular_values)
    csv_data = []
    
    for i, (singular_values, class_id, n_samples) in enumerate(zip(all_singular_values, class_ids, num_samples_per_class)):
        row = [class_id, n_samples, len(singular_values)]
        # 填充奇异值
        for sv in singular_values:
            row.append(sv)
        # 填充空白
        for _ in range(max_dims - len(singular_values)):
            row.append('')
        csv_data.append(row)
    
    # 创建表头
    header = ['class_id', 'num_samples', 'num_singular_values']
    for i in range(max_dims):
        header.append(f'singular_value_{i+1}')
    
    # 保存CSV
    np.savetxt(csv_path, csv_data, delimiter=',', 
               header=','.join(header), comments='', fmt='%s')
    
    print(f"奇异值数据已保存:")
    print(f"  - NumPy格式: {npz_path}")
    print(f"  - CSV格式: {csv_path}")

def load_singular_values_data(data_path):
    # 首先尝试NPZ格式
    npz_path = os.path.join(data_path, "singular_values_data.npz")
    if os.path.exists(npz_path):
        # 加载NumPy格式的数据
        data = np.load(npz_path, allow_pickle=True)
        all_singular_values = data['singular_values']
        class_ids = data['class_ids']
        num_samples_per_class = data['num_samples_per_class']
        print(f"从NPZ文件加载了 {len(all_singular_values)} 个类别的奇异值数据")
        
    else:
        # 尝试CSV格式
        csv_path = os.path.join(data_path, "singular_values_summary.csv")
        if os.path.exists(csv_path):
            # 加载CSV格式的数据
            df = pd.read_csv(csv_path)
            
            all_singular_values = []
            class_ids = []
            num_samples_per_class = []
            
            for _, row in df.iterrows():
                class_id = row['class_id']
                n_samples = row['num_samples']
                n_sv = int(row['num_singular_values'])
                
                # 提取奇异值（跳过前3列）
                singular_values = []
                for i in range(n_sv):
                    sv_col = f'singular_value_{i+1}'
                    if sv_col in row and not pd.isna(row[sv_col]):
                        singular_values.append(float(row[sv_col]))
                
                if singular_values:
                    all_singular_values.append(np.array(singular_values))
                    class_ids.append(class_id)
                    num_samples_per_class.append(n_samples)
            print(f"从CSV文件加载了 {len(all_singular_values)} 个类别的奇异值数据")
        else:
            raise FileNotFoundError(f"在目录 {data_path} 中找不到奇异值数据文件 (singular_values_data.npz 或 singular_values_summary.csv)")
    
    return all_singular_values, class_ids, num_samples_per_class

# %%
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行特征奇异值分析实验')
    parser.add_argument('--model', type=str, default='vit-b-p16-clip', 
                        help='模型名称 (vit-b-p16, vit-b-p16-clip, vit-b-p16-mocov3, vit-b-p16-dino)')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU编号')
    parser.add_argument('--iterations', type=int, default=0, 
                        help='迭代次数')
    parser.add_argument('--num_shots', type=int, default=128, 
                        help='样本数量')
    parser.add_argument('--load_singular_values', type=bool, default=True)
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 实验参数设置
    model_name = args.model
    iterations = args.iterations
    num_shots = args.num_shots
    base_output_dir = "实验结果保存/特征奇异值分析"
    
    print(f"\n处理架构: {model_name}, iterations: {iterations}, GPU: {args.gpu}")
    print("="*60)
    
    # 创建输出目录
    model_output_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    if not args.load_singular_values:
        # 加载数据
        print("加载数据...")
        dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
        
        # 创建完整数据加载器（使用全部累积数据进行奇异值分解）
        print("创建完整数据加载器...")
        # 合并所有7个数据集的训练集和测试集
        print("合并所有数据集的训练集和测试集...")
        
        # 获取所有训练数据和测试数据
        train_full_subset = dataset.get_subset(len(dataset.datasets) - 1, source='train', cumulative=True, mode="test")
        # test_full_subset = dataset.get_subset(len(dataset.datasets) - 1, source='test', cumulative=True, mode="test")
        
        # 使用ConcatDataset合并训练和测试数据
        from torch.utils.data import ConcatDataset
        # combined_dataset = ConcatDataset([train_full_subset, test_full_subset])
        full_loader = DataLoader(train_full_subset, batch_size=64, shuffle=False, num_workers=8)
        
        # 创建数据加载器（用于模型适配和比较，保留原有结构）
        train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
        
        # 获取和适配模型
        print("获取和适配Vision Transformer模型...")
        vit = get_vit(vit_name=model_name)
        adapt_loader = create_adapt_loader(train_subsets)
        vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
        
        # 提取特征
        print("提取特征...")
        train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
            vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations)
        
        train_dataset_ids = torch.tensor(train_dataset_ids)
        test_dataset_ids = torch.tensor(test_dataset_ids)

        full_features = torch.cat([train_features, test_features], dim=0)
        full_labels = torch.cat([train_labels, test_labels], dim=0)
        full_dataset_ids = torch.cat([train_dataset_ids, test_dataset_ids], dim=0)
        
        # 打印数据统计信息
        print(f"\n数据统计信息:")
        print(f"合并后的数据集包含 {len(full_features)} 个样本")
        print(f"涉及 {len(torch.unique(full_labels))} 个不同类别")
        # print(f"数据集ID范围: {torch.min(full_dataset_ids)} - {torch.max(full_dataset_ids)}")
        

        # 选择用于分析的集合（训练集、测试集或合并）
        print("\n" + "="*60)
        print("开始特征奇异值分析")
        print("="*60)
        
        # 使用完整数据集进行分析（100%数据）
        print("使用完整数据集进行奇异值分析...")
        features_to_analyze = full_features
        labels_to_analyze = full_labels
        # dataset_ids_to_analyze = full_dataset_ids
        
        print(f"分析数据统计: {len(features_to_analyze)} 个样本, {len(torch.unique(labels_to_analyze))} 个类别")
        
        # 计算每个类别的奇异值
        print("计算每个类别的奇异值...")
        all_singular_values, class_ids, num_samples_per_class = compute_class_wise_singular_values(
            features_to_analyze, labels_to_analyze, full_dataset_ids, device="cpu")
        
        if not all_singular_values:
            print("错误: 没有有效的奇异值数据")
            exit(1)

        # 保存奇异值数据
        print("保存奇异值数据...")
        save_singular_values_data(all_singular_values, class_ids, num_samples_per_class, model_output_dir)
        
    else:
        all_singular_values, class_ids, num_samples_per_class = load_singular_values_data(
            model_output_dir)
    
    # 绘制奇异值曲线
    print("绘制奇异值曲线...")
    plot_path = os.path.join(model_output_dir, f"{model_name}_singular_value_curves.png")
    plot_singular_value_curves(all_singular_values, class_ids, num_samples_per_class, plot_path)
    
    # 绘制95%累积比例秩箱型图
    print("绘制95%累积比例秩箱型图...")
    boxplot_path = os.path.join(model_output_dir, f"{model_name}_95percent_rank_boxplot.png")
    plot_violin_64dim_distribution(all_singular_values, class_ids, num_samples_per_class, boxplot_path)
    
    # 分析奇异值统计信息
    print("分析奇异值统计信息...")
    analyze_singular_value_statistics(all_singular_values, class_ids, num_samples_per_class, plot_path)
    
    print("\n" + "="*50)
    print("特征奇异值分析完成!")
    print("="*50)
    print(f"分析了 {len(all_singular_values)} 个类别的特征")
    print(f"结果保存在: {model_output_dir}")
    print(f"模型: {model_name} 在GPU {args.gpu} 上的实验完成!")

# 修改说明：
# 1. 添加了完整的100%数据加载功能
# 2. 使用累积的所有数据（所有7个数据集的合并）进行奇异值分解
# 3. 替代了原来的50%随机分割策略
# 4. 现在使用完整数据集提取特征并进行分析