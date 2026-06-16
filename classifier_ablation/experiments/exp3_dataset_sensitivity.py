# In[]
import os
from tqdm import tqdm
os.chdir('/home/raoxuan/projects/fancy_sgp_lora_vit')
print("当前工作目录:", os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
"""
实验3: 数据集级别参数敏感性分析
分析不同数据集对参数变化的敏感程度差异
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import seaborn as sns
import pandas as pd
from classifier.da_classifier_builder import QDAClassifierBuilder
from classifier_ablation.experiments.exp1_performance_surface import evaluate_qda_classifier, build_gaussian_statistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels, infer_dataset_ids_from_labels

def evaluate_qda_classifier_by_dataset(alpha1, alpha2, alpha3, stats, features, targets, dataset_ids,
                                       device="cuda", batch_size=512, custom_classifier=None):
    if custom_classifier is None:
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=alpha1,
            qda_reg_alpha2=alpha2,
            qda_reg_alpha3=alpha3,
            device=device)
        
        classifier = builder.build(stats)
    else:
        classifier = custom_classifier
    
    classifier.to(device)
    classifier.eval()
    classifier_device = next(classifier.parameters()).device
    
    dataset = torch.utils.data.TensorDataset(features, targets, dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_dataset_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(classifier_device)
            all_targets.append(batch[1])
            all_dataset_ids.append(batch[2])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_dataset_ids = torch.cat(all_dataset_ids)
    
    # 按数据集计算准确率
    unique_datasets = torch.unique(all_dataset_ids)
    dataset_accuracies = {}
    
    for dataset_id in unique_datasets:
        mask = (all_dataset_ids == dataset_id)
        if mask.sum() > 0:
            # 计算该数据集的总体准确率
            dataset_preds = all_predictions[mask]
            dataset_targets = all_targets[mask]
            
            total_correct = (dataset_preds == dataset_targets).float().sum().item()
            total_samples = len(dataset_targets)
            accuracy = total_correct / total_samples
            
            dataset_accuracies[int(dataset_id.item())] = accuracy * 100
    
    torch.cuda.empty_cache()
    return dataset_accuracies

# In[]
# 主要实验流程
if __name__ == '__main__':
    # 架构列表
    model_name = "vit-b-p16-clip"
    output_dir = "实验结果保存/分类器消融实验"
    
    # 基本参数
    num_shots = 128
    iterations = 0
    
    # 存储所有实验结果
    results = {}
    results[model_name] = {
            "fixed_alpha1": {
                "alpha1": [], 
                "alpha2": [], 
                "accuracy": [],
                "dataset_accuracies": []  # 新增：每个测试点的数据集级别准确率
            },
            "fixed_alpha2": {
                "alpha1": [], 
                "alpha2": [], 
                "accuracy": [],
                "dataset_accuracies": []  # 新增：每个测试点的数据集级别准确率
            }
        }

    alpha3_fixed = 0.5
    alpha2_range = (0.0, 3.0)
    alpha1_range = (0, 0.8)

    print(f"\n处理架构: {model_name}")
    print("="*60)

    # 加载数据
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)
    
    vit = get_vit(vit_name=model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
    
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations)

    # 转换为tensor
    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_dataset_ids = torch.tensor(test_dataset_ids)
    
    # 构建高斯统计量
    print("构建高斯统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)

    # 1. 固定alpha1，变动alpha2
    print(f"\n1. 固定alpha1，变动alpha2")
    fixed_alpha1 = 0.2
    alpha2_values = np.linspace(alpha2_range[0], alpha2_range[1], 5)

    for alpha2 in tqdm(alpha2_values, desc=f"{model_name} - α₁=0.1"):
        print(f"测试: alpha1={fixed_alpha1:.4f}, alpha2={alpha2:.4f}")
        
        dataset_accuracies = evaluate_qda_classifier_by_dataset(
            fixed_alpha1, alpha2, alpha3_fixed, train_stats, test_features,
            test_labels, test_dataset_ids, device="cuda")
        overall_accuracy = np.mean(list(dataset_accuracies.values()))
        results[model_name]['fixed_alpha1']['alpha1'].append(fixed_alpha1)
        results[model_name]['fixed_alpha1']['alpha2'].append(alpha2)
        results[model_name]['fixed_alpha1']['accuracy'].append(overall_accuracy)
        results[model_name]['fixed_alpha1']['dataset_accuracies'].append(dataset_accuracies)
        print(f"整体准确率: {overall_accuracy:.4f}, 数据集数量: {len(dataset_accuracies)}")

    # 2. 固定alpha2，变动alpha1
    print(f"\n2. 固定alpha2，变动alpha1")
    fixed_alpha2 = 2.0
    alpha1_values = np.linspace(alpha1_range[0], alpha1_range[1], 5)

    for alpha1 in tqdm(alpha1_values, desc=f"{model_name} - α₂=2.0"):
        print(f"测试: alpha1={alpha1:.4f}, alpha2={fixed_alpha2:.4f}")
        dataset_accuracies = evaluate_qda_classifier_by_dataset(
            alpha1, fixed_alpha2, alpha3_fixed, train_stats, test_features,
            test_labels, test_dataset_ids, device="cuda")
        
        overall_accuracy = np.mean(list(dataset_accuracies.values()))
        
        results[model_name]['fixed_alpha2']['alpha1'].append(alpha1)
        results[model_name]['fixed_alpha2']['alpha2'].append(fixed_alpha2)
        results[model_name]['fixed_alpha2']['accuracy'].append(overall_accuracy)
        results[model_name]['fixed_alpha2']['dataset_accuracies'].append(dataset_accuracies)
        print(f"整体准确率: {overall_accuracy:.4f}, 数据集数量: {len(dataset_accuracies)}")

# %%
# 引入绘图所需的库
model_name = "vit-b-p16-clip"
cross_domain_datasets_standard = [
    'CIFAR-100', 'CUB-200', 'RESISC-45', 'ImageNet-R',
    'Caltech-101', 'DTD', 'FGVC-Aircraft',
    'Food-101', 'MNIST', 'Oxford-Flower',
    'Oxford-Pets', 'Cars-196'
]

DATASET_ID_TO_NAME = {i: name for i, name in enumerate(cross_domain_datasets_standard)}


def prepare_heatmap_data(result_type, alpha_var_name, fixed_alpha_val):
    exp_data = results[model_name][result_type]
    all_dataset_ids = sorted(list(exp_data['dataset_accuracies'][0].keys()))
    df_rows = []
    alpha_var_values = exp_data[alpha_var_name]
    for i, alpha_var_val in enumerate(alpha_var_values):
        row = {'Alpha_Variable': alpha_var_val}
        current_accuracies = exp_data['dataset_accuracies'][i]
        for ds_id in all_dataset_ids:
            ds_name = DATASET_ID_TO_NAME.get(ds_id, f'DS_{ds_id}')
            row[ds_name] = current_accuracies.get(ds_id, np.nan)
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    df = df.set_index('Alpha_Variable')
    df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0, axis=0)
    df_normalized.index = [f'{v:.1f}' for v in df_normalized.index]
    return df, df_normalized, all_dataset_ids

sns.set_theme(style="whitegrid")

# --- 1. Fixed alpha1=0.2, Varying alpha2 ---
fixed_alpha1_val = 0.2
df_alpha2_raw, df_alpha2_normalized, _ = prepare_heatmap_data(
    'fixed_alpha1', 'alpha2', fixed_alpha1_val)

fixed_alpha2_val = 2.0
df_alpha1_raw, df_alpha1_normalized, _ = prepare_heatmap_data(
    'fixed_alpha2', 'alpha1', fixed_alpha2_val)

NEW_CMAP = "viridis"
ANNOT_FONTSIZE = 7 
LINE_COLOR = 'black'

FIG_WIDTH = 4.5
FIG_HEIGHT = 2.5
# --- 绘制并保存图 1: 固定 alpha1, 变动 alpha2 ---
fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
sns.heatmap(
    df_alpha2_normalized,
    annot=df_alpha2_raw.applymap(lambda x: f'{x:.1f}'),
    fmt='s',
    cmap=NEW_CMAP,
    linewidths=.5,
    linecolor=LINE_COLOR, # 应用黑色边框
    cbar=False,
    ax=ax1,
    vmin=0, vmax=1,
    annot_kws={"fontsize": ANNOT_FONTSIZE})

fig1_filename = f"QDA_Sensitivity_Alpha2_Varying_Alpha1_Fixed_{fixed_alpha1_val}.pdf"
ax1.set_ylabel(r'$\alpha_2^{\rm RGDA}$', fontsize=11)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
plt.setp(ax1.get_yticklabels(), rotation=45, ha="right", fontsize=8) 
plt.tight_layout()
plt.savefig(f"实验结果保存/分类器消融实验/exp3_dataset_sensitivity_{model_name}_alpha2.png", dpi=500)


fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
sns.heatmap(
    df_alpha1_normalized,
    annot=df_alpha1_raw.applymap(lambda x: f'{x:.1f}'),
    fmt='s',
    cmap=NEW_CMAP,
    linewidths=.5,
    linecolor=LINE_COLOR, # 应用黑色边框
    cbar=False, # 取消颜色条
    ax=ax2,
    vmin=0, vmax=1,
    annot_kws={"fontsize": ANNOT_FONTSIZE})

fig2_filename = f"QDA_Sensitivity_Alpha1_Varying_Alpha2_Fixed_{fixed_alpha2_val}.pdf"
ax2.set_ylabel(r'$\alpha_1^{\rm RGDA}$', fontsize=11)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=45, ha="right", fontsize=8) 
plt.tight_layout()
plt.savefig(f"实验结果保存/分类器消融实验/exp3_dataset_sensitivity_{model_name}_alpha1.png", dpi=500)
# %%
