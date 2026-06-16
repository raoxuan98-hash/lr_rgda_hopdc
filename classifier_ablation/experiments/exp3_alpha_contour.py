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
    parser = argparse.ArgumentParser(description='Alpha等高线实验')
    parser.add_argument('--model_name', type=str, default="vit-b-p16-mocov3",
                        help='模型名称 (默认: vit-b-p16)')
    parser.add_argument('--gpu', type=str, default="0",
                        help='使用的GPU设备 (默认: 0)')
    parser.add_argument('--iterations', type=int, default=0,
                        help='迭代次数 (默认: 0)')
    parser.add_argument('--num_shots', type=int, default=128,
                        help='每个类的样本数 (默认: 128)')
    parser.add_argument('--base_output_dir', type=str, default="实验结果保存/分类器消融实验",
                        help='基础输出目录 (默认: 实验结果保存/分类器消融实验)')
    parser.add_argument('--load_results', type=bool, default=True,
                        help='如果设置，则加载已保存的结果而不是重新运行实验')
    parser.add_argument('--alpha1_min', type=float, default=0.0,
                        help='α1的最小值 (默认: 0.0)')
    parser.add_argument('--alpha1_max', type=float, default=1.0,
                        help='α1的最大值 (默认: 1.0)')
    parser.add_argument('--alpha1_steps', type=int, default=11,
                        help='α1的采样点数 (默认: 10)')
    parser.add_argument('--alpha2_min', type=float, default=0.0,
                        help='α2的最小值 (默认: 0.0)')
    parser.add_argument('--alpha2_max', type=float, default=2.5,
                        help='α2的最大值 (默认: 2.0)')
    parser.add_argument('--alpha2_steps', type=int, default=11,
                        help='α2的采样点数 (默认: 10)')
    
    return parser.parse_args()


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

def evaluate_classifier_with_alpha(alpha1, alpha2, stats, features, targets, dataset_ids,
                                  classifier_type="qda", alpha3=0.01,
                                  device="cuda", batch_size=1024):
    """
    评估分类器性能（无约束条件）
    Args:
        alpha1: 第一正则化参数
        alpha2: 第二正则化参数
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
    if classifier_type == "qda":
        builder = QDAClassifierBuilder(
            qda_reg_alpha1=alpha1,
            qda_reg_alpha2=alpha2,
            qda_reg_alpha3=alpha3,
            low_rank=True,
            rank=64,
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
    classifier_device = next(classifier.parameters()).device
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(features, targets, dataset_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 评估分类器
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(classifier_device)
            all_targets.append(batch[1])
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # 计算准确率
    total_correct = (all_predictions == all_targets).float().sum().item()
    total_samples = len(all_targets)
    accuracy = total_correct / total_samples
    
    torch.cuda.empty_cache()
    return accuracy

def evaluate_classifiers_on_grid(
        alpha1_values, alpha2_values, train_stats, test_features, test_labels, test_dataset_ids, device="cuda"):
    """
    在二维网格上评估多个分类器的性能
    
    Args:
        alpha1_values: α1值的数组
        alpha2_values: α2值的数组
        train_stats: 训练数据的统计量
        test_features: 测试特征
        test_labels: 测试标签
        test_dataset_ids: 测试数据集ID
        alpha3: 第三正则化参数
        device: 计算设备
    
    Returns:
        qda_accuracies: QDA分类器的准确度矩阵 (alpha1_steps x alpha2_steps)
        sgd_linear_accuracies: 线性SGD分类器的准确度矩阵
        ncm_accuracy: NCM分类器的准确度（单一值）
        lda_accuracy: LDA分类器的准确度（单一值）
    """
    # 初始化准确度矩阵
    qda_accuracies = np.zeros((len(alpha1_values), len(alpha2_values)))
    sgd_linear_accuracies = np.zeros((len(alpha1_values), len(alpha2_values)))
    
    # 先评估NCM和LDA分类器（只需要评估一次）
    print(f"评估NCM分类器...")
    start_time = time.time()
    ncm_acc = evaluate_classifier_with_alpha(
        alpha1_values[0], alpha2_values[0], train_stats, test_features, test_labels, test_dataset_ids,
        classifier_type="ncm", alpha3=0.0, device=device
    )

    ncm_time = time.time() - start_time
    print(f"NCM准确度: {ncm_acc:.4f} (耗时: {ncm_time:.2f}s)")
    
    print(f"评估LDA分类器...")
    start_time = time.time()
    lda_acc = evaluate_classifier_with_alpha(
        alpha1_values[0], alpha2_values[0], train_stats, test_features, test_labels, test_dataset_ids,
        classifier_type="lda", alpha3=0.0, device=device
    )
    lda_time = time.time() - start_time
    print(f"LDA准确度: {lda_acc:.4f} (耗时: {lda_time:.2f}s)")
    
    total_evaluations = len(alpha1_values) * len(alpha2_values)
    print(f"\n开始评估 {total_evaluations} 个(α1, α2)组合点的性能...")
    
    evaluation_count = 0
    for i, alpha1 in enumerate(alpha1_values):
        for j, alpha2 in enumerate(alpha2_values):
            evaluation_count += 1
            print(f"\n评估点 {evaluation_count}/{total_evaluations}: α1={alpha1:.3f}, α2={alpha2:.3f}")
            
            # 评估QDA分类器
            print(f"  评估QDA分类器...")
            start_time = time.time()
            qda_acc = evaluate_classifier_with_alpha(
                alpha1, alpha2, train_stats, test_features, test_labels, test_dataset_ids,
                classifier_type="qda", alpha3=0.5, device=device
            )
            qda_time = time.time() - start_time
            qda_accuracies[i, j] = qda_acc
            print(f"    QDA准确度: {qda_acc:.4f} (耗时: {qda_time:.2f}s)")
            
            # 评估线性SGD分类器
            print(f"  评估线性SGD分类器...")
            start_time = time.time()
            sgd_linear_acc = evaluate_classifier_with_alpha(
                alpha1, alpha2, train_stats, test_features, test_labels, test_dataset_ids,
                classifier_type="sgd_linear", alpha3=0.0, device=device
            )
            sgd_linear_time = time.time() - start_time
            sgd_linear_accuracies[i, j] = sgd_linear_acc
            print(f"    线性SGD准确度: {sgd_linear_acc:.4f} (耗时: {sgd_linear_time:.2f}s)")
            
    return qda_accuracies, sgd_linear_accuracies, ncm_acc, lda_acc

def plot_alpha_contour(alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies,
                      ncm_accuracy, lda_accuracy, save_path=None, classifier_type="qda"):
    """
    绘制α1-α2等高线图
    
    Args:
        alpha1_values: α1值数组
        alpha2_values: α2值数组
        qda_accuracies: QDA分类器的准确度矩阵
        sgd_linear_accuracies: 线性SGD分类器的准确度矩阵
        ncm_accuracy: NCM分类器的准确度
        lda_accuracy: LDA分类器的准确度
        save_path: 保存路径
        classifier_type: 要绘制的分类器类型 ("qda", "sgd_linear", "all")
    """
    # 设置seaborn风格
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # 根据分类器类型选择准确度矩阵
    accuracies = None
    title = ""
    if classifier_type == "qda":
        accuracies = qda_accuracies
        title = "QDA Classifier Performance"
    elif classifier_type == "sgd_linear":
        accuracies = sgd_linear_accuracies
        title = "SGD-Linear Classifier Performance"
    elif classifier_type == "all":
        # 创建子图显示所有分类器
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Classifier Performance Contours", fontsize=14)
        
        classifiers = [
            ("QDA", qda_accuracies, axes[0, 0]),
            ("SGD-Linear", sgd_linear_accuracies, axes[0, 1])
        ]
        
        for name, acc_matrix, ax in classifiers:
            # 转换为百分比
            acc_pct = acc_matrix * 100
            
            # 创建网格
            alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values, indexing='ij')
            
            # 绘制等高线图
            contour = ax.contourf(alpha1_grid, alpha2_grid, acc_pct, levels=20, cmap='viridis')
            ax.contour(alpha1_grid, alpha2_grid, acc_pct, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # 添加颜色条
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Accuracy (%)', fontsize=10)
            
            # 设置标题和标签
            ax.set_title(f"{name} Performance", fontsize=12)
            ax.set_xlabel(r'$\alpha_1$', fontsize=10)
            ax.set_ylabel(r'$\alpha_2$', fontsize=10)
            
            # 标记最佳点
            max_idx = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
            ax.plot(alpha1_values[max_idx[0]], alpha2_values[max_idx[1]], 
                   'r*', markersize=10, markeredgecolor='black', markeredgewidth=0.5)
            ax.annotate(f'{acc_pct[max_idx]:.1f}%', 
                        xy=(alpha1_values[max_idx[0]], alpha2_values[max_idx[1]]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        color='white', ha='left', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7, edgecolor='none'))
        
        # 在最后一个子图中显示NCM和LDA的基准线
        ax = axes[1, 0]
        ax.axis('off')
        ax.text(0.5, 0.7, f'NCM Baseline: {ncm_accuracy*100:.2f}%', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0.5, 0.3, f'LDA Baseline: {lda_accuracy*100:.2f}%', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.set_title("Baseline Performance", fontsize=12)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            all_save_path = save_path.replace('.png', '_all_classifiers.png')
            plt.savefig(all_save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
            print(f"所有分类器等高线图已保存到: {all_save_path}")
        
        plt.show()
        return
    
    # 单个分类器的等高线图
    # IEEE单栏图片尺寸
    plt.figure(figsize=(3.5, 3.0))
    
    # 转换为百分比
    acc_pct = accuracies * 100
    
    # 创建网格
    alpha1_grid, alpha2_grid = np.meshgrid(alpha1_values, alpha2_values, indexing='ij')
    
    # 绘制等高线图
    contour = plt.contourf(alpha1_grid, alpha2_grid, acc_pct, levels=20, cmap='viridis')
    plt.contour(alpha1_grid, alpha2_grid, acc_pct, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(contour)
    cbar.set_label('Accuracy (%)', fontsize=10)
    
    # 标记最佳点
    max_idx = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    plt.plot(alpha1_values[max_idx[0]], alpha2_values[max_idx[1]], 
            'r*', markersize=10, markeredgecolor='black', markeredgewidth=0.5)
    plt.annotate(f'{acc_pct[max_idx]:.1f}%', 
                xy=(alpha1_values[max_idx[0]], alpha2_values[max_idx[1]]),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                color='white', ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7, edgecolor='none'))
    
    # 设置标签和标题
    plt.xlabel(r'$\alpha_1$', fontsize=10)
    plt.ylabel(r'$\alpha_2$', fontsize=10)
    plt.title(title, fontsize=12)
    
    # 设置坐标轴刻度
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    # 网格设置
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 移除顶部和右侧边框
    sns.despine()
    
    # 布局调整
    plt.tight_layout(pad=0.5)
    
    # 保存图像
    if save_path:
        classifier_save_path = save_path.replace('.png', f'_{classifier_type}.png')
        plt.savefig(classifier_save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
        print(f"{classifier_type}分类器等高线图已保存到: {classifier_save_path}")
    
    plt.show()

def save_contour_results(alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies,
                         ncm_accuracy, lda_accuracy, model_name, save_dir):
    """
    保存等高线实验结果
    
    Args:
        alpha1_values: α1值数组
        alpha2_values: α2值数组
        qda_accuracies: QDA准确度矩阵
        sgd_linear_accuracies: 线性SGD准确度矩阵
        ncm_accuracy: NCM准确度
        lda_accuracy: LDA准确度
        model_name: 模型名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存原始数据
    save_path = os.path.join(save_dir, f"{model_name}_contour_results.npz")
    np.savez(save_path,
             alpha1_values=alpha1_values,
             alpha2_values=alpha2_values,
             qda_accuracies=qda_accuracies,
             sgd_linear_accuracies=sgd_linear_accuracies,
             ncm_accuracy=ncm_accuracy,
             lda_accuracy=lda_accuracy)
    
    print(f"等高线实验结果已保存到: {save_path}")
    
    # 保存CSV格式便于查看（将矩阵展平）
    csv_path = os.path.join(save_dir, f"{model_name}_contour_results.csv")
    
    # 创建CSV数据
    csv_data = []
    for i, alpha1 in enumerate(alpha1_values):
        for j, alpha2 in enumerate(alpha2_values):
            csv_data.append([alpha1, alpha2,
                           qda_accuracies[i, j],
                           sgd_linear_accuracies[i, j]])
    
    # 添加标题行
    header = 'alpha1,alpha2,qda_accuracy,sgd_linear_accuracy'
    np.savetxt(csv_path, csv_data, delimiter=',', header=header, comments='')
    
    print(f"CSV格式结果已保存到: {csv_path}")
    
    return save_path

def load_contour_results(model_name, save_dir):
    """
    加载已保存的等高线实验结果
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
        
    Returns:
        alpha1_values: α1值数组
        alpha2_values: α2值数组
        qda_accuracies: QDA准确度矩阵
        sgd_linear_accuracies: 线性SGD准确度矩阵
        ncm_accuracy: NCM准确度
        lda_accuracy: LDA准确度
    """
    load_path = os.path.join(save_dir, f"{model_name}_contour_results.npz")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到已保存的结果文件: {load_path}")
    
    data = np.load(load_path)
    alpha1_values = data['alpha1_values']
    alpha2_values = data['alpha2_values']
    qda_accuracies = data['qda_accuracies']
    sgd_linear_accuracies = data['sgd_linear_accuracies']
    ncm_accuracy = data['ncm_accuracy']
    lda_accuracy = data['lda_accuracy']
    
    print(f"已加载等高线实验结果: {load_path}")
    
    return alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracy, lda_accuracy

# In[]
if __name__ == '__main__':
    
    # 实验参数设置
    model_name = args.model_name
    iterations = args.iterations
    num_shots = args.num_shots
    base_output_dir = args.base_output_dir
    
    # Alpha参数范围设置
    alpha1_min = args.alpha1_min
    alpha1_max = args.alpha1_max
    alpha1_steps = args.alpha1_steps
    alpha2_min = args.alpha2_min
    alpha2_max = args.alpha2_max
    alpha2_steps = args.alpha2_steps
    
    print(f"\n处理架构: {model_name}, iterations: {iterations}")
    print(f"α1范围: [{alpha1_min}, {alpha1_max}], 采样点数: {alpha1_steps}")
    print(f"α2范围: [{alpha2_min}, {alpha2_max}], 采样点数: {alpha2_steps}")
    print("="*60)
    
    # 创建输出目录
    model_output_dir = os.path.join(base_output_dir, f"{model_name}_iter{iterations}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 生成α1和α2的采样点
    alpha1_values = np.linspace(alpha1_min, alpha1_max, alpha1_steps)
    alpha2_values = np.linspace(alpha2_min, alpha2_max, alpha2_steps)
    print(f"α1采样点: {alpha1_values}")
    print(f"α2采样点: {alpha2_values}")
    
    # 初始化变量
    qda_accuracies = None
    sgd_linear_accuracies = None
    ncm_accuracy = None
    lda_accuracy = None
    
    # 检查是否需要加载已保存的结果
    if args.load_results:
        try:
            print("尝试加载已保存的实验结果...")
            alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies, ncm_accuracy, lda_accuracy = load_contour_results(
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
        print("运行α1-α2网格性能等高线实验")
        print("无约束条件，使用凸组合")
        print("="*60)
        
        # 检查CUDA是否可用，并设置设备
        if torch.cuda.is_available():
            device = "cuda:0"  # 使用第一个可见的GPU
            print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDA不可用，使用CPU")
        
        qda_accuracies, sgd_linear_accuracies, ncm_accuracy, lda_accuracy = evaluate_classifiers_on_grid(
            alpha1_values, alpha2_values, train_stats, test_features, test_labels, test_dataset_ids, device=device)
        
        # 保存实验结果
        save_contour_results(alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies,
                            ncm_accuracy, lda_accuracy, model_name, model_output_dir)
    
    # 绘制等高线图
    save_dir = model_output_dir
    plot_path = os.path.join(save_dir, f"{model_name}_alpha_contour.png")
    
    # 绘制所有分类器的等高线图
    plot_alpha_contour(alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies,
                      ncm_accuracy, lda_accuracy, plot_path, classifier_type="all")
    
    # 绘制单个分类器的等高线图
    for classifier_type in ["qda", "sgd_linear"]:
        plot_alpha_contour(alpha1_values, alpha2_values, qda_accuracies, sgd_linear_accuracies,
                          ncm_accuracy, lda_accuracy, plot_path, classifier_type=classifier_type)
        
    print("\n" + "="*50)
    print("实验总结")
    print("="*50)
    
    # 找到每个分类器的最佳性能点
    qda_max_idx = np.unravel_index(np.argmax(qda_accuracies), qda_accuracies.shape)
    sgd_linear_max_idx = np.unravel_index(np.argmax(sgd_linear_accuracies), sgd_linear_accuracies.shape)
    
    print(f"QDA最佳性能: α1={alpha1_values[qda_max_idx[0]]:.3f}, α2={alpha2_values[qda_max_idx[1]]:.3f}, 准确度={qda_accuracies[qda_max_idx]*100:.2f}%")
    print(f"线性SGD最佳性能: α1={alpha1_values[sgd_linear_max_idx[0]]:.3f}, α2={alpha2_values[sgd_linear_max_idx[1]]:.3f}, 准确度={sgd_linear_accuracies[sgd_linear_max_idx]*100:.2f}%")
    print(f"NCM性能: 准确度={ncm_accuracy*100:.2f}%")
    print(f"LDA性能: 准确度={lda_accuracy*100:.2f}%")
    
    # 计算平均性能
    avg_qda = np.mean(qda_accuracies) * 100
    avg_sgd_linear = np.mean(sgd_linear_accuracies) * 100
    print(f"QDA平均性能: {avg_qda:.2f}%")
    print(f"线性SGD平均性能: {avg_sgd_linear:.2f}%")
# %%