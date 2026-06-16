# In[]
import os
import argparse
from tqdm import tqdm
import sys
os.chdir('/home/raoxuan/projects/low_rank_rda')
print("当前工作目录:", os.getcwd())
sys.path.append('/home/raoxuan/projects/low_rank_rda')
import seaborn as sns
# 解析命令行参数（提前解析以获取GPU设置）
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Alpha约束实验')
    parser.add_argument('--model_name', type=str, default="vit-b-p16-dino",
                        help='模型名称 (默认: vit-b-p16)')
    parser.add_argument('--gpu', type=str, default="2",
                        help='使用的GPU设备 (默认: 0)')
    parser.add_argument('--iterations', type=int, default=0,
                        help='迭代次数 (默认: 0)')
    parser.add_argument('--num_shots', type=int, default=128,
                        help='每个类的样本数 (默认: 128)')
    parser.add_argument('--base_output_dir', type=str, default="实验结果保存/分类器消融实验",
                        help='基础输出目录 (默认: 实验结果保存/分类器消融实验)')
    parser.add_argument('--load_results', type=bool, default=True,
                        help='如果设置，则加载已保存的结果而不是重新运行实验')
    
    return parser.parse_args(args=[])
    # return parser.parse_args()


# 提前解析命令行参数以获取GPU设置
args = parse_args()

# 设置GPU（在导入torch之前）
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(f"使用GPU: {args.gpu}")

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from classifier.da_classifier_builder import QDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.ncm_classifier import NCMClassifier
from classifier.gaussian_classifier import LinearLDAClassifier
from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics
from classifier_ablation.data.data_loader import load_cross_domain_data, create_data_loaders, create_adapt_loader
from classifier_ablation.features.feature_extractor import get_vit, adapt_backbone, extract_features_and_labels
import time

def evaluate_classifier_with_alpha_constraint(alpha1, stats, features, targets, dataset_ids,
                                              classifier_type="qda", alpha3=0.01,
                                              device="cuda", batch_size=512):
    """
    在约束条件α1 + α2 = 1.0下评估分类器性能
    Args:
        alpha1: 第一正则化参数
        stats: 高斯统计量
        features: 测试特征
        targets: 测试标签
        dataset_ids: 数据集ID
        classifier_type: 分类器类型 ("qda", "sgd_linear", "ncm", "lda")
        alpha3: 第三正则化参数（固定）
        device: 计算设备
        batch_size: 批次大小
    Returns:
        accuracy: 分类准确度
    """
    alpha2 = 1.0 - alpha1
    if classifier_type == "qda":
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=alpha1,
            qda_reg_alpha2=alpha2,
            qda_reg_alpha3=alpha3,
            low_rank=True,
            rank=128,
            device=device)
        classifier = builder.build(stats)
        
    elif classifier_type == "sgd_linear":
        builder = SGDClassifierBuilder(device=device)
        classifier = builder.build(stats,
                                 linear=True,
                                 alpha1=alpha1,
                                 alpha2=alpha2,
                                 alpha3=alpha3)
    elif classifier_type == "ncm":
        # NCM分类器不需要alpha参数，直接使用stats构建
        classifier = NCMClassifier(stats, device=device)

    elif classifier_type == "lda":
        # LDA分类器使用alpha1作为正则化参数
        classifier = LinearLDAClassifier(stats, lda_reg_alpha=alpha1, device=device)
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")
    
    classifier.to(device)
    classifier.eval()
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(features, targets, dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 评估分类器
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            all_targets.append(batch[1])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # 计算class-wise平均准确率
    num_classes = len(torch.unique(all_targets))
    class_accuracies = []
    
    for class_id in range(num_classes):
        class_mask = (all_targets == class_id)
        if class_mask.sum() > 0:  # 确保该类有样本
            class_correct = ((all_predictions == all_targets) & class_mask).float().sum().item()
            class_total = class_mask.sum().item()
            class_accuracy = class_correct / class_total
            class_accuracies.append(class_accuracy)
    
    accuracy = np.mean(class_accuracies) if class_accuracies else 0.0
    
    torch.cuda.empty_cache()
    return accuracy

def evaluate_classifiers_under_constraint(
        alpha1_values, train_stats, test_features, test_labels, test_dataset_ids,
        alpha3=0.5, device="cuda"):
    """
    在约束条件下评估多个分类器的性能
    
    Args:
        alpha1_values: α1值的数组
        train_stats: 训练数据的统计量
        test_features: 测试特征
        test_labels: 测试标签
        test_dataset_ids: 测试数据集ID
        alpha3: 第三正则化参数
        device: 计算设备
    
    Returns:
        qda_accuracies: QDA分类器的准确度数组
        sgd_linear_accuracies: 线性SGD分类器的准确度数组
        ncm_accuracies: NCM分类器的准确度数组（单一值复制）
        lda_accuracies: LDA分类器的准确度数组（单一值复制）
    """
    qda_accuracies = []
    sgd_linear_accuracies = []
    
    # 先评估NCM和LDA分类器（只需要评估一次）
    print(f"评估NCM分类器...")
    start_time = time.time()
    ncm_acc = evaluate_classifier_with_alpha_constraint(
        alpha1_values[0], train_stats, test_features, test_labels, test_dataset_ids,
        classifier_type="ncm", alpha3=alpha3, device=device
    )
    ncm_time = time.time() - start_time
    print(f"NCM准确度: {ncm_acc:.4f} (耗时: {ncm_time:.2f}s)")
    # 为所有alpha1值复制相同的NCM准确度
    ncm_accuracies = [ncm_acc] * len(alpha1_values)
    
    print(f"评估LDA分类器...")
    start_time = time.time()
    lda_acc = evaluate_classifier_with_alpha_constraint(
        alpha1_values[0], train_stats, test_features, test_labels, test_dataset_ids,
        classifier_type="lda", alpha3=alpha3, device=device
    )
    lda_time = time.time() - start_time
    print(f"LDA准确度: {lda_acc:.4f} (耗时: {lda_time:.2f}s)")
    # 为所有alpha1值复制相同的LDA准确度
    lda_accuracies = [lda_acc] * len(alpha1_values)
    
    print(f"\n开始评估 {len(alpha1_values)} 个α1值点的性能...")
    
    for i, alpha1 in enumerate(alpha1_values):
        alpha2 = 1.0 - alpha1
        print(f"\n评估点 {i+1}/{len(alpha1_values)}: α1={alpha1:.3f}, α2={alpha2:.3f}")
        print(f"  评估QDA分类器...")
        start_time = time.time()
        qda_acc = evaluate_classifier_with_alpha_constraint(
            alpha1, train_stats, test_features, test_labels, test_dataset_ids,
            classifier_type="qda", alpha3=0.5, device=device
        )
        qda_time = time.time() - start_time
        qda_accuracies.append(qda_acc)
        print(f"    QDA准确度: {qda_acc:.4f} (耗时: {qda_time:.2f}s)")
        
        # 评估线性SGD分类器
        print(f"  评估线性SGD分类器...")
        start_time = time.time()
        sgd_linear_acc = evaluate_classifier_with_alpha_constraint(
            alpha1, train_stats, test_features, test_labels, test_dataset_ids,
            classifier_type="sgd_linear", alpha3=0.0, device=device
        )
        sgd_linear_time = time.time() - start_time
        sgd_linear_accuracies.append(sgd_linear_acc)
        print(f"    线性SGD准确度: {sgd_linear_acc:.4f} (耗时: {sgd_linear_time:.2f}s)")
        
    return qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies

def plot_alpha_constraint_performance(alpha1_values, qda_accuracies, sgd_linear_accuracies,
                                    ncm_accuracies, lda_accuracies, save_path=None, show_best_points=True):

    # 设置seaborn风格
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # IEEE单栏图片尺寸
    plt.figure(figsize=(3.5, 2.5))
    
    # 转换准确度为百分比
    qda_acc_pct = np.array(qda_accuracies) * 100
    sgd_linear_acc_pct = np.array(sgd_linear_accuracies) * 100
    ncm_acc_pct = np.array(ncm_accuracies) * 100
    lda_acc_pct = np.array(lda_accuracies) * 100
    
    # 定义颜色和样式 - 不使用marker
    colors = sns.color_palette("deep", 4)
    linestyles = ['-', '--', '-.', ':']
    linewidths = [1.5, 1.5, 1.2, 1.2]  # 主要方法用粗线，常量方法用细线
    
    # 绘制性能曲线（所有数据点，不使用marker）
    plt.plot(alpha1_values, qda_acc_pct, color=colors[0], linestyle=linestyles[0], 
             label='QDA', linewidth=linewidths[0])
    plt.plot(alpha1_values, sgd_linear_acc_pct, color=colors[1], linestyle=linestyles[1],
             label='SGD-linear', linewidth=linewidths[1])
    
    # 绘制NCM和LDA的水平线（常量）
    ncm_mean = float(np.mean(ncm_acc_pct))
    lda_mean = float(np.mean(lda_acc_pct))
    
    plt.axhline(y=ncm_mean, color=colors[2], linestyle=linestyles[2],
                label=f'NCM: {ncm_mean:.1f}%', linewidth=linewidths[2])
    plt.axhline(y=lda_mean, color=colors[3], linestyle=linestyles[3],
                label=f'LDA: {lda_mean:.1f}%', linewidth=linewidths[3])
    
    # 初始化最佳点变量
    qda_best_idx = None
    sgd_linear_best_idx = None
    
    # 标记最佳性能点
    if show_best_points:
        qda_best_idx = np.argmax(qda_accuracies)
        sgd_linear_best_idx = np.argmax(sgd_linear_accuracies)
        
        # 在最佳点位置添加星形标记（只标记最佳点）
        plt.plot(alpha1_values[qda_best_idx], qda_acc_pct[qda_best_idx],
                marker='*', color=colors[0], markersize=8, linestyle='None',
                markeredgecolor='black', markeredgewidth=0.5)
        plt.plot(alpha1_values[sgd_linear_best_idx], sgd_linear_acc_pct[sgd_linear_best_idx],
                marker='*', color=colors[1], markersize=8, linestyle='None',
                markeredgecolor='black', markeredgewidth=0.5)
        
        # 在图中添加最佳值文本标注
        plt.annotate(f'{qda_acc_pct[qda_best_idx]:.1f}%', 
                    xy=(alpha1_values[qda_best_idx], qda_acc_pct[qda_best_idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=6,
                    color=colors[0], ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='none'))
        plt.annotate(f'{sgd_linear_acc_pct[sgd_linear_best_idx]:.1f}%',
                    xy=(alpha1_values[sgd_linear_best_idx], sgd_linear_acc_pct[sgd_linear_best_idx]),
                    xytext=(5, 5), textcoords='offset points', fontsize=6,
                    color=colors[1], ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 设置标签和标题
    plt.xlabel(r'$\alpha_1$', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    
    # 设置坐标轴刻度
    plt.xticks(np.linspace(min(alpha1_values), max(alpha1_values), 6), fontsize=8)
    plt.yticks(fontsize=8)
    
    # 优化图例
    plt.legend(fontsize=6, loc='best', frameon=True, fancybox=False, 
               framealpha=0.8, edgecolor='black')
    
    # 网格设置
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    
    # 移除顶部和右侧边框
    sns.despine()
    
    # 布局调整
    plt.tight_layout(pad=0.5)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
        print(f"性能曲线图已保存到: {save_path}")
    
    plt.show()
    
    # 返回最佳性能信息
    if show_best_points and qda_best_idx is not None:
        return (alpha1_values[qda_best_idx], alpha1_values[sgd_linear_best_idx],
                qda_acc_pct[qda_best_idx], sgd_linear_acc_pct[sgd_linear_best_idx],
                ncm_mean, lda_mean)
    else:
        return None, None, None, None, None, None


def save_constraint_results(alpha1_values, qda_accuracies, sgd_linear_accuracies,
                            ncm_accuracies, lda_accuracies, model_name, save_dir):
    """
    保存约束条件实验结果
    
    Args:
        alpha1_values: α1值数组
        qda_accuracies: QDA准确度数组
        sgd_linear_accuracies: 线性SGD准确度数组
        ncm_accuracies: NCM准确度数组
        lda_accuracies: LDA准确度数组
        model_name: 模型名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存原始数据
    save_path = os.path.join(save_dir, f"{model_name}_constraint_results.npz")
    np.savez(save_path,
             alpha1_values=alpha1_values,
             qda_accuracies=qda_accuracies,
             sgd_linear_accuracies=sgd_linear_accuracies,
             ncm_accuracies=ncm_accuracies,
             lda_accuracies=lda_accuracies)
    
    print(f"约束条件实验结果已保存到: {save_path}")
    
    # 保存CSV格式便于查看
    csv_path = os.path.join(save_dir, f"{model_name}_constraint_results.csv")
    results_df = np.column_stack([alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies])
    np.savetxt(csv_path, results_df, delimiter=',',
               header='alpha1,qda_accuracy,sgd_linear_accuracy,ncm_accuracy,lda_accuracy', comments='')
    
    print(f"CSV格式结果已保存到: {csv_path}")
    
    return save_path

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
# In[]
if __name__ == '__main__':
    
    # 实验参数设置
    model_name = args.model_name
    iterations = args.iterations
    num_shots = args.num_shots
    base_output_dir = args.base_output_dir
    
    print(f"\n处理架构: {model_name}, iterations: {iterations}")
    print("="*60)
    
    # 创建输出目录
    model_output_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 初始化变量，确保在所有路径中都定义
    alpha1_values = None
    qda_accuracies = None
    sgd_linear_accuracies = None
    ncm_accuracies = None
    lda_accuracies = None
    
    # 检查是否需要加载已保存的结果
    if args.load_results:
        try:
            print("尝试加载已保存的实验结果...")
            alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies = load_constraint_results(
                model_name, model_output_dir)
            print("成功加载已保存的实验结果，跳过实验运行。")
        except FileNotFoundError as e:
            print(f"无法加载已保存的结果: {e}")
            print("将运行新的实验...")
            args.load_results = False
    
    if not args.load_results:
        # 加载数据
        print("加载数据...")
        dataset, train_subsets, test_subsets = load_cross_domain_data(num_shots=num_shots, model_name=model_name)
        
        # 打印数据集信息
        print("\n数据集信息:")
        print(f"总类别数: {dataset.total_classes}")
        print(f"class_names 数量: {len(dataset.global_class_names)}")
        print(f"标签范围: 0 到 {dataset.total_classes - 1}")
        
        # 创建数据加载器
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
        
        # 构建高斯统计量
        print("\n构建高斯统计量...")
        train_stats = build_gaussian_statistics(train_features, train_labels)
        
        print("\n" + "="*60)
        print("运行约束条件下的α1-α2性能曲线对比实验")
        print("约束条件: α1 + α2 = 1.0")
        print("="*60)
        
        # 生成α1采样点
        alpha1_values = np.linspace(0, 1.0, 11)
        print(f"α1采样点: {alpha1_values}")
        print(f"模型: {model_name}")
        # 检查CUDA是否可用，并设置设备
        if torch.cuda.is_available():
            device = "cuda:0"  # 使用第一个可见的GPU
            print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA不可用，使用CPU")
        
        qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies = evaluate_classifiers_under_constraint(
            alpha1_values, train_stats, test_features, test_labels, test_dataset_ids,
            alpha3=0.01, device=device)
        
        # 保存实验结果
        save_constraint_results(alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies,
                            model_name, model_output_dir)
    
    save_dir = model_output_dir
    plot_path = os.path.join(save_dir, f"{model_name}_constraint_performance.png")
    best_alpha1_qda, best_alpha1_sgd_linear, best_acc_qda, best_acc_sgd_linear, best_acc_ncm, best_acc_lda = plot_alpha_constraint_performance(
        alpha1_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracies, lda_accuracies, plot_path
    )
        
    print("\n" + "="*50)
    print("实验总结")
    print("="*50)
    print(f"QDA最佳性能: α1={best_alpha1_qda:.3f}, 准确度={best_acc_qda:.2f}%")
    print(f"线性SGD最佳性能: α1={best_alpha1_sgd_linear:.3f}, 准确度={best_acc_sgd_linear:.2f}%")
    print(f"NCM性能: 准确度={best_acc_ncm:.2f}%")
    print(f"LDA性能: 准确度={best_acc_lda:.2f}%")
    
    # 计算平均性能
    avg_qda = np.mean(np.array(qda_accuracies)) * 100
    avg_sgd_linear = np.mean(np.array(sgd_linear_accuracies)) * 100
    avg_ncm = np.mean(np.array(ncm_accuracies)) * 100
    avg_lda = np.mean(np.array(lda_accuracies)) * 100
    print(f"QDA平均性能: {avg_qda:.2f}%")
    print(f"线性SGD平均性能: {avg_sgd_linear:.2f}%")
    print(f"NCM平均性能: {avg_ncm:.2f}%")
    print(f"LDA平均性能: {avg_lda:.2f}%")
    
    # 找到性能差异最大的点
    if alpha1_values is not None and qda_accuracies is not None and sgd_linear_accuracies is not None:
        acc_diff_qda_linear = np.abs(np.array(qda_accuracies) - np.array(sgd_linear_accuracies))
        
        max_diff_idx_qda_linear = np.argmax(acc_diff_qda_linear)
        
        print(f"QDA与线性SGD最大性能差异: {acc_diff_qda_linear[max_diff_idx_qda_linear]*100:.2f}% at α1={alpha1_values[max_diff_idx_qda_linear]:.3f}")
# %%