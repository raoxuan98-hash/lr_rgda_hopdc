"""
数据加载和预处理模块 - Within Domain 版本
"""
from torch.utils.data import random_split, Dataset, Subset, DataLoader
from utils.data_manager import WithinDomainDataManager
import torchvision.transforms as transforms

def load_within_domain_data(dataset_name="cifar100_224", init_cls=50, increment=10, model_name="vit-b-p16-clip", seed=0):
    """
    加载域内数据集
    
    Args:
        dataset_name: 数据集名称
        init_cls: 初始类别数
        increment: 增量类别数
        model_name: 模型名称
        seed: 随机种子
    
    Returns:
        dataset: 数据集管理器
        train_subsets: 训练数据子集
        test_subsets: 测试数据子集
    """
    
    # 创建域内数据管理器
    dataset = WithinDomainDataManager(
        dataset_name=dataset_name,
        seed=seed,
        init_cls=init_cls,
        increment=increment,
        args={}
    )
    
    # 获取最后一个任务的数据（包含所有类别）
    last_task_id = dataset.nb_tasks - 1
    train_subset = dataset.get_subset(last_task_id, source='train', cumulative=True, mode='train')
    test_subset = dataset.get_subset(last_task_id, source='test', cumulative=True, mode='test')
    
    return dataset, train_subset, test_subset

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