# In[]
import os
import sys
sys.path.append('/home/raoxuan/projects/low_rank_rda')
os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())

# 设置GPU设备必须在导入torch之前
# 只有在没有设置CUDA_VISIBLE_DEVICES时才使用默认值
# if 'CUDA_VISIBLE_DEVICES' not in os.environ:
#     os.environ['CUDA_VISIBLfE_DEVICES'] = '0'
#     print("使用默认GPU: 0")
# else:
#     print(f"使用环境变量设置的GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from classifier.da_classifier_builder import QDAClassifierBuilder
from compensator.gaussian_statistics import GaussianStatistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels, infer_dataset_ids_from_labels

def build_gaussian_statistics(features, labels):
    import tqdm
    
    features = features.cpu()
    labels = labels.cpu()
    unique_labels = torch.unique(labels)
    
    stats = {}
    for lbl in tqdm.tqdm(unique_labels):
        mask = (labels == lbl)
        feats_class = features[mask]
        
        mu = feats_class.mean(0)
        if feats_class.size(0) >= 2:
            cov = torch.cov(feats_class.T) + torch.eye(feats_class.size(1)) * 1e-4
        else:
            cov = torch.eye(feats_class.size(1)) * 1e-4
        stats[int(lbl.item())] = GaussianStatistics(mu, cov)
    
    return stats

def evaluate_qda_classifier(alpha1, alpha2, alpha3, stats, features, targets, dataset_ids,
                           device="cuda", batch_size=2048, custom_classifier=None, return_class_wise=True, rank=64):
    if custom_classifier is None:
        try:
            builder = QDAClassifierBuilder(
                qda_reg_alpha1=alpha1,
                qda_reg_alpha2=alpha2,
                qda_reg_alpha3=alpha3,
                low_rank=True,
                rank=rank,
                device=device)
            
            classifier = builder.build(stats)
        except Exception as e:
            print(f"构建QDA分类器失败: {e}")
            print(f"参数: alpha1={alpha1}, alpha2={alpha2}, alpha3={alpha3}, rank={rank}")
            return 0.0
    else:
        classifier = custom_classifier
    
    
    classifier.to(device)
    classifier.eval()

    dataset = torch.utils.data.TensorDataset(features, targets, dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_dataset_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            all_targets.append(batch[1])
            all_dataset_ids.append(batch[2])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_dataset_ids = torch.cat(all_dataset_ids)
    
    # 计算准确率
    if return_class_wise:
        # 计算每个类别的准确率
        unique_classes = torch.unique(all_targets)
        class_accuracies = []
        
        for class_id in unique_classes:
            mask = (all_targets == class_id)
            if mask.sum() > 0:
                class_correct = (all_predictions[mask] == all_targets[mask]).float().sum().item()
                class_total = mask.sum().item()
                class_acc = class_correct / class_total
                class_accuracies.append(class_acc)
        
        accuracy = np.mean(class_accuracies) if class_accuracies else 0.0
        print(f"Class-wise平均准确率: {accuracy:.4f}")
    else:
        # 计算所有样本的总体准确度
        total_correct = (all_predictions == all_targets).float().sum().item()
        total_samples = len(all_targets)
        accuracy = total_correct / total_samples
        print(f"全部样本总体准确度: {accuracy:.4f} (正确: {total_correct}/{total_samples})")
    
    torch.cuda.empty_cache()
    return accuracy

def plot_alpha1_alpha2_contour(alpha1_values, alpha2_values, accuracy_matrix, save_path=None, cmap='viridis'):
    plt.figure(figsize=(3.5, 2.5))  # IEEE单栏标准尺寸
    
    # 创建网格
    alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values)
    
    # 绘制等高线图
    contour = plt.contourf(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=15, cmap=cmap)
    cbar = plt.colorbar(contour, shrink=0.8)
    cbar.set_label('Average accuracy', fontsize=8)
    
    # 设置colorbar刻度为5个
    vmin = np.min(accuracy_matrix)
    vmax = np.max(accuracy_matrix)
    tick_positions = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_positions)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticklabels([f'{tick:.1f}' for tick in tick_positions])
    
    # 添加等高线
    contour_lines = plt.contour(alpha1_grid, alpha2_grid, accuracy_matrix.T, levels=15, colors='black', alpha=0.4)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    
    # 找到最佳准确率及其对应的参数
    max_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_alpha1 = alpha1_values[max_idx[0]]
    best_alpha2 = alpha2_values[max_idx[1]]
    best_acc = accuracy_matrix[max_idx]
    plt.plot(best_alpha1, best_alpha2, 'r*', markersize=4, label=f'Best: ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.4f}')

    plt.xlabel(r'$\alpha_1$', fontsize=8)
    plt.ylabel(r'$\alpha_2$', fontsize=8)
    
    # 设置x轴和y轴刻度为4个
    plt.xticks(np.linspace(min(alpha1_values), max(alpha1_values), 5), fontsize=8)
    plt.yticks(np.linspace(min(alpha2_values), max(alpha2_values), 6), fontsize=8)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"等高线图已保存到: {save_path}")
    
    plt.show()
    return best_alpha1, best_alpha2, best_acc

def grid_search_alpha1_alpha2(train_stats, test_features, test_labels, test_dataset_ids,
                            alpha1_min=0, alpha1_max=1.0, alpha2_min=0, alpha2_max=1.0,
                            alpha1_points=10, alpha2_points=10, return_class_wise=True,
                            device="cuda", use_log_sampling=False, rank=64):

    if use_log_sampling:
        # 对数采样: 使用logspace在10^alpha_min到10^alpha_max之间采样
        # 处理0值：用很小的正数代替
        alpha1_min_log = np.log10(max(alpha1_min, 1e-6))
        alpha1_max_log = np.log10(max(alpha1_max, 1e-6))
        alpha2_min_log = np.log10(max(alpha2_min, 1e-6))
        alpha2_max_log = np.log10(max(alpha2_max, 1e-6))
        
        alpha1_values = np.logspace(alpha1_min_log, alpha1_max_log, alpha1_points)
        alpha2_values = np.logspace(alpha2_min_log, alpha2_max_log, alpha2_points)
        
        # 如果原始最小值为0，将第一个值设为0
        if alpha1_min == 0:
            alpha1_values[0] = 0.0
        if alpha2_min == 0:
            alpha2_values[0] = 0.0
            
        print(f"Alpha1对数采样值: {alpha1_values}")
        print(f"Alpha2对数采样值: {alpha2_values}")
    else:
        alpha1_values = np.linspace(alpha1_min, alpha1_max, alpha1_points)
        alpha2_values = np.linspace(alpha2_min, alpha2_max, alpha2_points)
        
    alpha3_fixed = 0.5
    
    # 初始化准确率矩阵
    accuracy_matrix = np.zeros((alpha1_points, alpha2_points))
    print(f"开始二维网格搜索: {alpha1_points} x {alpha2_points} = {alpha1_points * alpha2_points} 个组合")
    print(f"Alpha3固定为: {alpha3_fixed}")
    print(f"Rank: {rank}")
    total_tests = alpha1_points * alpha2_points
    
    completed = 0
    # 创建tqdm进度条
    with tqdm(total=total_tests, desc="测试进度") as pbar:
        for i, alpha1 in enumerate(alpha1_values):
            for j, alpha2 in enumerate(alpha2_values):
                # 直接调用评估函数
                acc = evaluate_qda_classifier(alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                                test_labels, test_dataset_ids, device=device,
                                                return_class_wise=return_class_wise, rank=rank)
                accuracy_matrix[i, j] = acc * 100
                completed += 1
                pbar.update(1)
                                
    
    return alpha1_values, alpha2_values, accuracy_matrix

def save_results(alpha1_values, alpha2_values, accuracy_matrix, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_results.npz")
    np.savez(save_path,
             alpha1_values=alpha1_values,
             alpha2_values=alpha2_values,
             accuracy_matrix=accuracy_matrix)
    print(f"结果已保存到: {save_path}")
    return save_path

def run_experiment1_performance_surface(train_stats, test_features, test_labels, test_dataset_ids,
                                      alpha1_min=0, alpha1_max=5.0, alpha2_min=0, alpha2_max=5.0,
                                      alpha1_points=11, alpha2_points=11, save_path=None, device="cuda", use_log_sampling=False, rank=64):
    print("\n" + "="*50)
    print("实验1: 性能曲面等高线图")
    print("="*50)
    
    # 执行二维网格搜索
    alpha1_values, alpha2_values, accuracy_matrix = grid_search_alpha1_alpha2(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_min=alpha1_min, alpha1_max=alpha1_max,
        alpha2_min=alpha2_min, alpha2_max=alpha2_max,
        alpha1_points=alpha1_points, alpha2_points=alpha2_points,
        return_class_wise=False, device=device, use_log_sampling=use_log_sampling, rank=rank
    )
    
    return alpha1_values, alpha2_values, accuracy_matrix

# In[]
if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行exp1_performance_surface实验')
    parser.add_argument('--model', type=str, default='vit-b-p16',
                        help='模型名称 (vit-b-p16, vit-b-p16-clip, vit-b-p16-mocov3, vit-b-p16-dino)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU编号')
    parser.add_argument('--rank', type=int, default=32,
                        help='低秩分解的rank值')
    parser.add_argument('--iterations', type=int, default=0,
                        help='迭代次数')
    parser.add_argument('--num_shots', type=int, default=128,
                        help='样本数量')
    args = parser.parse_args()
    
    # 注意：GPU环境变量已在文件开头导入torch之前设置
    # 此处不能重新设置，因为torch已经导入
    print(f"使用GPU: {args.gpu} (已在启动时设置)")
    
    # 实验参数设置
    model_name = args.model
    rank = args.rank
    iterations = args.iterations
    num_shots = args.num_shots
    base_output_dir = "实验结果保存/分类器消融实验"
    
    # 所有可用的模型列表
    # model_names = ["vit-b-p16-clip", "vit-b-p16", "vit-b-p16-dino", "vit-b-p16-mocov3"]
    model_names = ["vit-b-p16-mocov3"]
    
    print(f"\n处理架构: {model_name}, iterations: {iterations}, rank: {rank}, GPU: {args.gpu}")
    print("="*60)
    
    # 创建架构和iterations特定的输出目录
    model_output_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}_rank{rank}")
    os.makedirs(model_output_dir, exist_ok=True)

    # 加载数据
    dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_subsets, test_subsets)

    vit = get_vit(vit_name=model_name)
    adapt_loader = create_adapt_loader(train_subsets)
    vit = adapt_backbone(vit, adapt_loader, dataset.total_classes, iterations=iterations)
    train_features, train_labels, train_dataset_ids, test_features, test_labels, test_dataset_ids = extract_features_and_labels(
        vit, dataset, train_loader, test_loader, model_name, num_shots=num_shots, iterations=iterations)

    train_dataset_ids = torch.tensor(train_dataset_ids)
    test_dataset_ids = torch.tensor(test_dataset_ids)

    # 构建高斯统计量
    print("\n构建高斯统计量...")
    train_stats = build_gaussian_statistics(train_features, train_labels)

    # 实验1: 性能曲面等高线图
    print("\n" + "="*60)
    print("运行实验1: 性能曲面等高线图")
    print("="*60)
    alpha1_values, alpha2_values, accuracy_matrix = run_experiment1_performance_surface(
        train_stats, test_features, test_labels, test_dataset_ids,
        alpha1_min=0, alpha1_max=1.0, alpha2_min=0, alpha2_max=3.0,
        alpha1_points=11, alpha2_points=16,
        save_path=f"{model_output_dir}/exp1_contour_{model_name}_alpha3_zero_rank{rank}.png",
        device="cuda", use_log_sampling=False, rank=rank)

    # 保存计算结果
    save_results(alpha1_values, alpha2_values, accuracy_matrix, f"{model_name}_rank{rank}", model_output_dir)
    
    # 绘制不同cmap的等高线图
    cmaps = ["viridis", "jet", "Blues", "Spectral"]
    for cmap in cmaps:
        save_path = os.path.join(model_output_dir, f"exp1_contour_{cmap}_alpha3_zero_rank{rank}.png")
        best_alpha1, best_alpha2, best_acc = plot_alpha1_alpha2_contour(
            alpha1_values, alpha2_values, accuracy_matrix, save_path, cmap)
        print(f"使用 {cmap} cmap 的等高线图已保存，最佳参数: ({best_alpha1:.3f}, {best_alpha2:.3f}) = {best_acc:.4f}")

    print(f"\n{model_name} rank-{rank} 在GPU {args.gpu} 上的实验完成!")

# %%
