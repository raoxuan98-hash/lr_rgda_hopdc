"""
数据加载和预处理模块
"""
from torch.utils.data import random_split, Dataset, Subset, DataLoader
from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore, create_balanced_data_manager

def load_cross_domain_data(num_shots=128, model_name="vit-b-p16-clip", seed=0):
    """
    加载跨域数据集
    
    Args:
        num_shots: 每个数据集的样本数
        model_name: 模型名称
        seed: 随机种子
    
    Returns:
        dataset: 数据集管理器
        train_subsets: 训练数据子集
utils        test_subsets: 测试数据子集
    """
    cross_domain_datasets = [
        'cifar100_224', 'imagenet-r', 'cars196_224', 'cub200_224',
        'caltech-101', 'oxford-flower-102', 'food-101']
    
    dataset = create_balanced_data_manager(
        dataset_names=cross_domain_datasets,
        balanced_datasets_root="balanced_datasets",
        shuffle=False,
        seed=seed,
        num_shots=num_shots,
        use_balanced_datasets=True
    )
    subsets = dataset.get_subset(len(cross_domain_datasets) - 1, source='train', cumulative=True, mode="test")
    train_subsets, test_subsets = random_split(subsets, [0.5, 0.5])
    return dataset, train_subsets, test_subsets

def create_data_loaders(train_subsets, test_subsets, batch_size=64, num_workers=8):

    train_loader = DataLoader(train_subsets, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subsets, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def create_adapt_loader(train_subsets, batch_size=24, num_workers=6):
    """
    创建适应训练的数据加载器
    
    Args:
        train_subsets: 训练数据子集
        batch_size: 批次大小
        num_workers: 工作进程数
    
    Returns:
        adapt_loader: 适应训练数据加载器
    """
    adapt_loader = DataLoader(train_subsets, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return adapt_loader