import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

# 尝试导入 RAPIDS cuml 的 GPU 版 TSNE，如果未安装则退回 sklearn 的 CPU 版
try:
    from cuml.manifold import TSNE as cuTSNE
    print("成功导入 cuml.manifold.TSNE，将使用 GPU 加速。")
except ImportError:
    print("警告: 未检测到 cuml 库，退回使用 sklearn 的 CPU 版 TSNE。")
    from sklearn.manifold import TSNE as cuTSNE

def select_features_across_tasks(features_dir, feature_files, class_ids):
    """
    Select features from multiple classes.
    """
    selected_features = {}
    selected_labels = {}
    for f in feature_files:
        task_id = int(f.split('_')[1])
        data = torch.load(os.path.join(features_dir, f), map_location='cpu')
        train_features = data['train_features'].numpy()
        train_labels = data['train_labels'].numpy()

        # 创建布尔掩码：选择 train_labels 在 class_ids 中的样本
        mask_indices = np.isin(train_labels, class_ids)
        
        # 应用掩码筛选特征和标签
        selected_features[task_id] = train_features[mask_indices]
        selected_labels[task_id] = train_labels[mask_indices]
        
    return selected_features, selected_labels

def cosine_similarity_pairwise(vecs_a, vecs_b):
    """
    逐对计算余弦相似度：vecs_a[i] 与 vecs_b[i]
    """
    norm_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    
    vecs_a_norm = vecs_a / (norm_a + 1e-8)
    vecs_b_norm = vecs_b / (norm_b + 1e-8)
    
    similarity = np.sum(vecs_a_norm * vecs_b_norm, axis=1)
    return similarity

def get_pastel_colors(n_labels=200):
    colors = []
    for i in range(n_labels):
        hue = i / n_labels
        colors.append(mcolors.hsv_to_rgb([hue, 0.5, 0.95]))  # 降低饱和度，提高亮度
    return colors

# 全局固定的label->color映射
LABEL_TO_COLOR = {}

def get_label_color(label, all_labels):
    """为每个label分配固定颜色"""
    global LABEL_TO_COLOR
    if not LABEL_TO_COLOR:
        unique_sorted = sorted(np.unique(all_labels))
        colors = get_pastel_colors(len(unique_sorted))
        for idx, lab in enumerate(unique_sorted):
            LABEL_TO_COLOR[lab] = colors[idx]
    return LABEL_TO_COLOR[label]

def plot_tsne_tasks(features, labels, task_ids, save_path=None, dataset=None):
    n_tasks = len(task_ids)
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks, 4))
    
    # 如果只有一个task，axes不是数组，转为列表
    if n_tasks == 1:
        axes = [axes]
    
    # 修复：动态使用 task_ids 列表里的第一个 task 获取 unique labels
    first_task = task_ids[0]
    unique_labels = np.unique(labels[first_task])
    n_classes = len(unique_labels)
    print(f"[{dataset}] 类别数量: {n_classes}")
    
    for i, task_id in enumerate(task_ids):
        ax = axes[i]
        
        # 使用 cuTSNE (GPU 加速)
        tsne_features = cuTSNE(n_components=2, random_state=42, init="pca").fit_transform(features[task_id])
        task_labels = labels[task_id]
        
        for label in sorted(unique_labels):
            mask = task_labels == label
            color = get_label_color(label, task_labels) 
            
            ax.scatter(tsne_features[mask, 0], tsne_features[mask, 1],
                      c=[color], 
                      label=f'{label}',
                      alpha=0.85,
                      s=30,
                      edgecolors='white',
                      linewidth=0.3)
        
        ax.set_title(f'After task {task_id + 1}, Classes: {min(task_labels)}-{max(task_labels)}', fontsize=12)

    plt.tight_layout()
    if dataset is not None:
        fig.suptitle(f't-SNE Visualization of {dataset}', 
                    fontsize=14, fontweight='bold', y=1.02)
        
    if save_path is not None:
        save_file = os.path.join(save_path, 'tsne_tasks.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_file}")
    
    plt.close(fig) # 批量执行时直接关闭画布，避免内存泄漏或弹出过多窗口

def plot_tsne_tasks_multiple(features_dir, class_ids, task_ids, dataset_name):
    if not os.path.exists(features_dir):
        print(f"路径不存在，跳过: {features_dir}")
        return
        
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('_features.pt')]
    if not feature_files:
        print(f"该目录下没有特征文件，跳过: {features_dir}")
        return
        
    selected_features, selected_labels = select_features_across_tasks(features_dir, feature_files, class_ids)
    plot_tsne_tasks(selected_features, selected_labels, task_ids=task_ids, save_path=features_dir, dataset=dataset_name)

# =============================================================================
# 批量执行入口
# =============================================================================
if __name__ == '__main__':
    # 集中配置要跑的实验信息。
    # CIFAR-100 通常是 10 个 Task，其他通常是 20 个 Task。你可以按需修改 task_ids 列表。
    experiments = [
        # --- CIFAR-100 ---
        {
            "dataset": "CIFAR-100",
            "class_ids": list(range(10)),
            "task_ids": [0, 4, 7, 9],
            "dirs": [
                "RGDA_WD_2025-12-19-within/DS_cifar100_224/VB16/I10_C10/r4_Basic_ht0.05_hk400_C_AS2K_I/Oada_LR0.0001_B16_IT1000/seed_1993/cached_features"
                "RGDA_WD_2025-12-19-within/DS_cifar100_224/VB16/I10_C10/r4_Full_ht0.05_hk400_kd1.0_TF_DTide_UTTru_C_AS2K_I/Oada_LR5e-06_B16_IT1500/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_cifar100_224/VB16/I10_C10/r4_Full_ht0.05_hk400_C_AS2K_I/Oada_LR5e-06_B16_IT1000/seed_1993/cached_features"
            ]
        },
        # --- Cars-196 ---
        {
            "dataset": "Cars-196",
            "class_ids": list(range(20)),
            "task_ids": [0, 4, 7, 9],
            "dirs": [
                "RGDA_WD_2025-12-19-within/DS_cars196_224/VB16/I20_C20/r4_Basic_ht0.05_hk400_C_AS2K_I/Oada_LR0.0001_B16_IT1000/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_cars196_224/VB16/I20_C20/r4_Full_ht0.05_hk400_C_AS2K_I/Oada_LR5e-06_B16_IT1000/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_cars196_224/VB16/I20_C20/r4_Full_ht0.05_hk400_kd1.0_TF_DTide_UTTru_C_AS2K_I/Oada_LR5e-06_B16_IT1500/seed_1993/cached_features"
            ]
        },
        # --- CUB-200 ---
        {
            "dataset": "CUB-200",
            "class_ids": list(range(20)),
            "task_ids": [0, 4, 7, 9],
            "dirs": [
                "RGDA_WD_2025-12-19-within/DS_cub200_224/VB16/I20_C20/r4_Basic_ht0.05_hk400_C_AS2K_I/Oada_LR0.0001_B16_IT500/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_cub200_224/VB16/I20_C20/r4_Full_ht0.05_hk400_C_AS2K_I/Oada_LR5e-06_B16_IT500/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_cub200_224/VB16/I20_C20/r4_Full_ht0.05_hk400_kd1.0_TF_DTide_UTTru_C_AS2K_I/Oada_LR5e-06_B16_IT500/seed_1993/cached_features"
            ]
        },
        # --- ImageNet-R ---
        {
            "dataset": "ImageNet-R",
            "class_ids": list(range(20)),
            "task_ids": [0, 4, 7, 9],
            "dirs": [
                "RGDA_WD_2025-12-19-within/DS_imagenet-r/VB16/I20_C20/r4_Basic_ht0.05_hk400_C_AS2K_I/Oada_LR0.0001_B16_IT1000/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_imagenet-r/VB16/I20_C20/r4_Full_ht0.05_hk400_C_AS2K_I/Oada_LR5e-06_B16_IT1000/seed_1993/cached_features",
                "RGDA_WD_2025-12-19-within/DS_imagenet-r/VB16/I20_C20/r4_Full_ht0.05_hk400_kd1.0_TF_DTide_UTTru_C_AS2K_I/Oada_LR5e-06_B16_IT1500/seed_1993/cached_features"
            ]
        }
    ]

    # 循环遍历所有实验配置并执行
    for exp in experiments:
        for d in exp["dirs"]:
            print(f"\n---> 开始处理: {exp['dataset']} - 路径: {d.split('/')[-3]}")
            # 注意重置一下颜色映射，避免跨数据集颜色混乱
            LABEL_TO_COLOR = {} 
            plot_tsne_tasks_multiple(
                features_dir=d, 
                class_ids=exp["class_ids"], 
                task_ids=exp["task_ids"],
                dataset_name=exp["dataset"]
            )
    
    print("\n所有 t-SNE 可视化任务执行完毕！")