"""
实验3: 分类器对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from classifier.da_classifier_builder import QDAClassifierBuilder
from classifier.sgd_classifier_builder import SGDClassifierBuilder
from classifier.ncm_classifier import NCMClassifier
from classifier.da_classifier_builder import LDAClassifierBuilder

def evaluate_qda_classifier(alpha1, alpha2, alpha3, stats, features, targets, dataset_ids,
                           device="cuda", batch_size=512, custom_classifier=None, return_dataset_wise=False):
    """
    评估QDA分类器
    
    Args:
        alpha1, alpha2, alpha3: QDA正则化参数
        stats: 高斯统计量
        features: 特征张量
        targets: 目标标签
        dataset_ids: 数据集ID
        device: 设备
        batch_size: 批次大小
        custom_classifier: 自定义分类器
        return_dataset_wise: 是否返回数据集级别的准确率
    
    Returns:
        accuracy: 准确率
    """
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
    
    # 计算每个数据集的准确率
    unique_datasets = torch.unique(all_dataset_ids)
    dataset_accuracies = []
    
    for dataset_id in unique_datasets:
        mask = (all_dataset_ids == dataset_id)
        if mask.sum() > 0:
            dataset_correct = (all_predictions[mask] == all_targets[mask]).float().sum().item()
            dataset_total = mask.sum().item()
            dataset_acc = dataset_correct / dataset_total
            dataset_accuracies.append(dataset_acc)
    
    # 计算准确率
    if return_dataset_wise:
        # 计算所有数据集准确率的平均值
        accuracy = np.mean(dataset_accuracies) if dataset_accuracies else 0.0
        print(f"Dataset-wise平均准确率: {accuracy:.4f}")
    else:
        # 计算所有样本的总体准确度
        total_correct = (all_predictions == all_targets).float().sum().item()
        total_samples = len(all_targets)
        accuracy = total_correct / total_samples
        print(f"全部样本总体准确度: {accuracy:.4f} (正确: {total_correct}/{total_samples})")
    
    torch.cuda.empty_cache()
    return accuracy

def plot_experiment3_comparison(results, save_path=None):
    """
    绘制实验3对比图
    
    Args:
        results: 实验结果
        save_path: 保存路径
    """
    plt.figure(figsize=(3.5, 2.5))  # IEEE单栏标准尺寸
    
    classifiers = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    bars = plt.bar(classifiers, accuracies, color=colors[:len(classifiers)], alpha=0.7, width=0.6)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=7)
    
    plt.xlabel('Classifier Type', fontsize=9)
    plt.ylabel('Dataset-wise Avg-Acc', fontsize=9)
    plt.title('Classifier Performance Comparison', fontsize=9)
    plt.xticks(rotation=45, fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分类器对比图已保存到: {save_path}")
    
    plt.show()

def run_experiment3_classifier_comparison(train_stats, test_features, test_labels, test_dataset_ids,
                                        sgd_epochs=5, sgd_lr=0.01, save_path=None):
    """
    运行实验3: 分类器对比
    
    Args:
        train_stats: 训练数据统计量
        test_features: 测试特征
        test_labels: 测试标签
        test_dataset_ids: 测试数据集ID
        sgd_epochs: SGD训练轮数
        sgd_lr: SGD学习率
        save_path: 保存路径
    
    Returns:
        results: 实验结果
    """
    print("\n" + "="*50)
    print("实验3: SGD分类器对比及baseline集成")
    print("="*50)
    
    results = {}
    alpha3_fixed = 0.5
    
    # 1. RGDA分类器（不同参数）
    print("\n1. 测试RGDA分类器...")
    rgda_params = [
        (0.5, 0.5, "RGDA(0.5,0.5)"),
        (1.0, 2.0, "RGDA(1.0,2.0)"),
        (2.0, 3.0, "RGDA(2.0,3.0)"),
    ]
    
    for alpha1, alpha2, name in rgda_params:
        print(f"测试: {name}")
        
        acc = evaluate_qda_classifier(alpha1, alpha2, alpha3_fixed, train_stats, test_features,
                                     test_labels, test_dataset_ids, device="cuda", return_dataset_wise=True)
        
        results[name] = acc
        print(f"准确率: {acc:.4f}")
    
    # 2. SGD分类器
    print("\n2. 测试SGD分类器...")
    
    # 生成随机样本用于SGD训练
    cached_Z = torch.randn(1024, test_features.size(1))
    
    # 构建SGD分类器
    sgd_builder = SGDClassifierBuilder(
        cached_Z=cached_Z,
        device="cuda",
        epochs=sgd_epochs,
        lr=sgd_lr
    )
    
    classifier = sgd_builder.build(train_stats)
    acc = evaluate_qda_classifier(0, 0, 0, train_stats, test_features,
                                 test_labels, test_dataset_ids, device="cuda",
                                 custom_classifier=classifier, return_dataset_wise=True)
    
    results['SGD'] = acc
    print(f"SGD准确率: {acc:.4f}")
    
    # 3. NCM baseline
    print("\n3. 测试NCM baseline...")
    
    ncm_classifier = NCMClassifier(train_stats).to("cuda")
    acc = evaluate_qda_classifier(0, 0, 0, train_stats, test_features,
                                 test_labels, test_dataset_ids, device="cuda",
                                 custom_classifier=ncm_classifier, return_dataset_wise=True)
    
    results['NCM'] = acc
    print(f"NCM准确率: {acc:.4f}")
    
    # 4. LDA baseline (alpha1=0的RGDA变体)
    print("\n4. 测试LDA baseline...")
    
    lda_builder = LDAClassifierBuilder(
        reg_alpha=0.3,
        device="cuda"
    )
    
    classifier = lda_builder.build(train_stats)
    acc = evaluate_qda_classifier(0, 0, 0, train_stats, test_features,
                                 test_labels, test_dataset_ids, device="cuda",
                                 custom_classifier=classifier, return_dataset_wise=True)
    
    results['LDA'] = acc
    print(f"LDA准确率: {acc:.4f}")
    
    # 绘制对比图
    plot_experiment3_comparison(results, save_path)
    
    return results