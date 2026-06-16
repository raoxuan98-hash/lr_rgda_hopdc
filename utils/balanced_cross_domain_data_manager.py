import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from pathlib import Path

from utils.data1 import get_dataset, SimpleDataset, pil_loader
from utils.cross_domain_data_manager import CrossDomainDataManagerCore, CrossDomainSimpleDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BalancedCrossDomainDataManagerCore(CrossDomainDataManagerCore):
    """
    平衡后的Cross-domain class-incremental data manager.
    继承自CrossDomainDataManagerCore，支持加载平衡后的数据集
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        balanced_datasets_root: str = "balanced_datasets",
        seed: int = 0,
        num_shots: int = 0,
        log_level: int = logging.INFO,
        use_balanced_datasets: bool = True,
        enable_incremental_split: bool = False,
        num_incremental_splits: int = 2,
        incremental_split_seed: int = 42,
        shuffle: bool = False
    ) -> None:
        
        self.balanced_datasets_root = balanced_datasets_root
        self.use_balanced_datasets = use_balanced_datasets
        self.enable_incremental_split = enable_incremental_split
        self.num_incremental_splits = num_incremental_splits
        self.incremental_split_seed = incremental_split_seed
        
        if use_balanced_datasets:
            self._init_balanced_datasets(dataset_names, shuffle, seed, num_shots, log_level)
        else:
            # 使用原始数据集
            super().__init__(dataset_names, shuffle, seed, num_shots, log_level)
    
    def _init_balanced_datasets(self, dataset_names: List[str], shuffle: bool, seed: int, num_shots: int, log_level: int):
        """初始化平衡后的数据集"""
        
        logging.basicConfig(level=log_level)
        self.dataset_names = dataset_names
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_shots = int(num_shots)
        
        # Load all balanced datasets
        self.datasets = []
        self.global_label_offset = []
        self.total_classes = 0
        self.global_class_names = []
        
        for i, dataset_name in enumerate(dataset_names):
            logging.info(f"[BCDM] Loading balanced dataset {i+1}/{len(dataset_names)}: {dataset_name}")
            dataset_info = self._load_balanced_dataset(dataset_name)
            
            # Apply few-shot sampling if num_shots > 0
            if self.num_shots > 0:
                logging.info(f"[BCDM] Applying few-shot sampling: {self.num_shots} shots per class")
                dataset_info = self._apply_few_shot_sampling(dataset_info, self.seed + i)
            
            offset = self.total_classes
            # 保持原始标签为 0-based，偏移将在 get_subset 中应用
            # dataset_info['train_targets'] = dataset_info['train_targets'] + offset
            # dataset_info['test_targets'] = dataset_info['test_targets'] + offset
            
            self.datasets.append(dataset_info)
            self.global_label_offset.append(offset)
            self.total_classes += dataset_info['num_classes']
            self.global_class_names.extend(dataset_info['class_names'])
        
        logging.info(f"[BCDM] Total datasets: {len(self.datasets)}")
        logging.info(f"[BCDM] Total classes: {self.total_classes}")
        logging.info(f"[BCDM] Total tasks: {len(self.datasets)}")
        
        # 如果启用了增量拆分，则进行增量拆分
        if self.enable_incremental_split:
            logging.info(f"[BCDM] Creating incremental splits: {self.num_incremental_splits} splits per dataset")
            self._create_incremental_splits()

    def _split_classes_randomly(self, num_classes: int, num_splits: int, seed: int) -> List[List[int]]:
        """
        随机均匀拆分类别
        
        Args:
            num_classes: 总类别数
            num_splits: 拆分数
            seed: 随机种子
            
        Returns:
            每个子集的类别索引列表（从0开始的连续索引）
        """
        np.random.seed(seed)
        
        # 创建类别索引列表并随机打乱
        class_indices = list(range(num_classes))
        np.random.shuffle(class_indices)
        
        # 计算每个子集的类别数
        base_classes_per_split = num_classes // num_splits
        remainder = num_classes % num_splits
        
        # 分配类别到各个子集
        splits = []
        start_idx = 0
        
        for i in range(num_splits):
            # 前remainder个子集多分配一个类别
            num_classes_in_split = base_classes_per_split + (1 if i < remainder else 0)
            end_idx = start_idx + num_classes_in_split
            
            split_classes = class_indices[start_idx:end_idx]
            splits.append(sorted(split_classes))  # 保持类别索引有序
            
            start_idx = end_idx
        
        return splits
    
    def _create_incremental_splits(self) -> None:
        """创建增量拆分，将每个数据集拆分为多个增量子集"""
        if not self.enable_incremental_split or self.num_incremental_splits <= 1:
            return
        
        incremental_datasets = []
        incremental_global_label_offset = []
        
        # 首先，我们需要保存原始数据集的全局偏移量
        original_datasets_global_offsets = []
        current_offset = 0
        for dataset in self.datasets:
            original_datasets_global_offsets.append(current_offset)
            current_offset += dataset['num_classes']
        
        # 现在进行拆分
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_name = dataset['name']
            num_classes = dataset['num_classes']
            original_global_offset = original_datasets_global_offsets[dataset_idx]
            
            logging.info(f"[BCDM] Creating incremental splits for dataset {dataset_name}: "
                        f"{num_classes} classes -> {self.num_incremental_splits} splits, "
                        f"original global offset: {original_global_offset}")
            
            # 随机拆分类别
            class_splits = self._split_classes_randomly(num_classes, self.num_incremental_splits,
                                                       self.incremental_split_seed + dataset_idx)
            
            # 为每个拆分创建数据集
            for split_idx, class_indices in enumerate(class_splits):
                split_dataset = self._create_incremental_dataset(
                    dataset, 
                    class_indices, 
                    split_idx,
                    original_global_offset
                )
                
                incremental_datasets.append(split_dataset)
                
                # 计算新的全局标签偏移（基于当前累积的类别数）
                offset = sum(d['num_classes'] for d in incremental_datasets[:-1])
                incremental_global_label_offset.append(offset)
                
                logging.info(f"[BCDM]   Split {split_idx + 1}: {len(class_indices)} classes, "
                            f"original classes: {class_indices}, "
                            f"new global offset: {offset}, "
                            f"{len(split_dataset['train_data'])} train samples, "
                            f"{len(split_dataset['test_data'])} test samples")
        
        # 更新数据集和偏移
        self.datasets = incremental_datasets
        self.global_label_offset = incremental_global_label_offset
        self.total_classes = sum(d['num_classes'] for d in self.datasets)
        
        logging.info(f"[BCDM] After incremental split: {len(self.datasets)} total tasks, "
                    f"{self.total_classes} total classes")
        for i, offset in enumerate(self.global_label_offset):
            logging.info(f"[BCDM] Task {i}: global offset = {offset}, num_classes = {self.datasets[i]['num_classes']}")

    def _create_incremental_dataset(self, original_dataset: Dict, class_indices: List[int],
                                   split_idx: int, original_global_offset: int) -> Dict:
        """
        创建增量子集数据集
        
        Args:
            original_dataset: 原始数据集字典
            class_indices: 当前拆分的类别索引（在原始数据集中的索引）
            split_idx: 拆分索引
            original_global_offset: 原始数据集的全局偏移量
            
        Returns:
            拆分后的数据集字典
        """
        # 将原始数据集中的本地类别索引转换为全局索引（如果原始数据集已经是全局的）
        # 或者直接使用本地索引（如果原始数据集是本地的）
        # 由于我们在 _init_balanced_datasets 中注释掉了全局偏移，这里的 original_dataset 应该是本地标签 (0-based)
        
        # 筛选训练数据
        train_mask = np.isin(original_dataset['train_targets'], class_indices)
        train_data = original_dataset['train_data'][train_mask]
        train_targets_original = original_dataset['train_targets'][train_mask]
        
        # 筛选测试数据
        test_mask = np.isin(original_dataset['test_targets'], class_indices)
        test_data = original_dataset['test_data'][test_mask]
        test_targets_original = original_dataset['test_targets'][test_mask]
        
        # 创建标签映射：原始本地标签 -> 拆分后的新本地标签 (从0开始)
        sorted_local_classes = sorted(class_indices)
        local_label_mapping = {orig_label: new_local_label for new_local_label, orig_label in enumerate(sorted_local_classes)}
        
        # 将筛选后的标签映射到新的本地标签
        train_targets_local = np.array([local_label_mapping[t] for t in train_targets_original])
        test_targets_local = np.array([local_label_mapping[t] for t in test_targets_original])
        
        # 筛选类别名称
        class_names = [original_dataset['class_names'][idx] for idx in class_indices]
        
        # 创建子集数据集
        split_dataset = {
            'name': f"{original_dataset['name']}_split_{split_idx}",
            'train_data': train_data,
            'test_data': test_data,
            'train_targets': train_targets_local,  # 新的本地标签（0开始）
            'test_targets': test_targets_local,    # 新的本地标签（0开始）
            'num_classes': len(class_indices),
            'use_path': original_dataset['use_path'],
            'class_names': class_names,
            'templates': original_dataset['templates'],
            'original_dataset_name': original_dataset['name'],
            'split_index': split_index if 'split_index' in locals() else split_idx,
            'original_class_indices': class_indices,
            'original_global_offset': original_global_offset,
            'global_class_indices': [idx + original_global_offset for idx in class_indices]
        }
        
        return split_dataset

    def _load_balanced_dataset(self, dataset_name: str) -> Optional[Dict]:
        """加载平衡后的数据集"""
        dataset_path = Path(self.balanced_datasets_root) / dataset_name
        
        if not dataset_path.exists():
            logging.warning(f"[BCDM] Balanced dataset not found: {dataset_path}")
            return None
        
        try:
            # 读取标签文件
            label_file = dataset_path / "label.txt"
            if not label_file.exists():
                logging.error(f"[BCDM] Label file not found: {label_file}")
                return None
            
            with open(label_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            
            # 读取训练和测试数据
            train_data, train_targets = self._load_data_from_directory(dataset_path / "train")
            test_data, test_targets = self._load_data_from_directory(dataset_path / "test")
            
            # 判断是否使用路径
            use_path = all(isinstance(x, str) for x in train_data[:10]) if len(train_data) > 0 else False
            
            # 获取模板（从原始数据集）
            try:
                original_dataset = get_dataset(dataset_name)
                templates = getattr(original_dataset, 'templates', [])
            except:
                templates = []
            
            dataset_info = {
                'name': dataset_name,
                'train_data': np.array(train_data),
                'test_data': np.array(test_data),
                'train_targets': np.array(train_targets, dtype=np.int64),
                'test_targets': np.array(test_targets, dtype=np.int64),
                'num_classes': len(class_names),
                'use_path': use_path,
                'class_names': class_names,
                'templates': templates
            }
            
            logging.info(f"[BCDM] Loaded balanced dataset {dataset_name}: "
                        f"{len(train_data)} train samples, {len(test_data)} test samples, "
                        f"{len(class_names)} classes")
            
            return dataset_info
            
        except Exception as e:
            logging.error(f"[BCDM] Error loading balanced dataset {dataset_name}: {str(e)}")
            return None
    
    def _load_data_from_directory(self, dir_path: Path) -> Tuple[List, List[int]]:
        """从目录结构加载数据"""
        data = []
        targets = []
        
        if not dir_path.exists():
            return data, targets
        
        # 遍历所有类别目录
        for class_dir in sorted(dir_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue
            
            # 遍历类别中的所有文件
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    data.append(str(file_path))
                    targets.append(class_id)
        
        return data, targets
    
    def get_balanced_statistics(self) -> Dict[str, Dict]:
        """获取平衡后数据集的统计信息"""
        stats = {}
        
        for i, dataset in enumerate(self.datasets):
            dataset_name = dataset['name']
            
            # 统计每个类别的样本数
            train_counts = {}
            test_counts = {}
            
            for label in dataset['train_targets']:
                train_counts[label] = train_counts.get(label, 0) + 1
            
            for label in dataset['test_targets']:
                test_counts[label] = test_counts.get(label, 0) + 1
            
            # 计算统计指标
            train_values = list(train_counts.values())
            test_values = list(test_counts.values())
            
            stats[dataset_name] = {
                'num_classes': dataset['num_classes'],
                'total_train_samples': len(dataset['train_data']),
                'total_test_samples': len(dataset['test_data']),
                'train_per_class': {
                    'min': min(train_values) if train_values else 0,
                    'max': max(train_values) if train_values else 0,
                    'mean': np.mean(train_values) if train_values else 0,
                    'std': np.std(train_values) if train_values else 0
                },
                'test_per_class': {
                    'min': min(test_values) if test_values else 0,
                    'max': max(test_values) if test_values else 0,
                    'mean': np.mean(test_values) if test_values else 0,
                    'std': np.std(test_values) if test_values else 0
                }
            }
        
        return stats
    
    def compare_with_original(self) -> Dict[str, Dict]:
        """与原始数据集进行比较"""
        comparison = {}
        
        for dataset_name in self.dataset_names:
            try:
                # 加载原始数据集
                original_dataset = get_dataset(dataset_name)
                
                # 找到对应的平衡数据集
                balanced_dataset = None
                for dataset in self.datasets:
                    if dataset['name'] == dataset_name:
                        balanced_dataset = dataset
                        break
                
                if balanced_dataset is None:
                    continue
                
                # 获取原始数据
                original_train_targets = getattr(original_dataset, 'train_targets', [])
                original_test_targets = getattr(original_dataset, 'test_targets', [])
                original_train_data = getattr(original_dataset, 'train_data', [])
                original_test_data = getattr(original_dataset, 'test_data', [])
                
                # 计算原始统计
                original_train_counts = {}
                original_test_counts = {}
                
                for label in original_train_targets:
                    original_train_counts[int(label)] = original_train_counts.get(int(label), 0) + 1
                
                for label in original_test_targets:
                    original_test_counts[int(label)] = original_test_counts.get(int(label), 0) + 1
                
                # 计算平衡后统计
                balanced_train_counts = {}
                balanced_test_counts = {}
                
                for label in balanced_dataset['train_targets']:
                    balanced_train_counts[int(label)] = balanced_train_counts.get(int(label), 0) + 1
                
                for label in balanced_dataset['test_targets']:
                    balanced_test_counts[int(label)] = balanced_test_counts.get(int(label), 0) + 1
                
                comparison[dataset_name] = {
                    'original': {
                        'total_train_samples': len(original_train_data),
                        'total_test_samples': len(original_test_data),
                        'train_per_class_stats': {
                            'min': min(original_train_counts.values()) if original_train_counts else 0,
                            'max': max(original_train_counts.values()) if original_train_counts else 0,
                            'mean': np.mean(list(original_train_counts.values())) if original_train_counts else 0,
                            'std': np.std(list(original_train_counts.values())) if original_train_counts else 0
                        },
                        'test_per_class_stats': {
                            'min': min(original_test_counts.values()) if original_test_counts else 0,
                            'max': max(original_test_counts.values()) if original_test_counts else 0,
                            'mean': np.mean(list(original_test_counts.values())) if original_test_counts else 0,
                            'std': np.std(list(original_test_counts.values())) if original_test_counts else 0
                        }
                    },
                    'balanced': {
                        'total_train_samples': len(balanced_dataset['train_data']),
                        'total_test_samples': len(balanced_dataset['test_data']),
                        'train_per_class_stats': {
                            'min': min(balanced_train_counts.values()) if balanced_train_counts else 0,
                            'max': max(balanced_train_counts.values()) if balanced_train_counts else 0,
                            'mean': np.mean(list(balanced_train_counts.values())) if balanced_train_counts else 0,
                            'std': np.std(list(balanced_train_counts.values())) if balanced_train_counts else 0
                        },
                        'test_per_class_stats': {
                            'min': min(balanced_test_counts.values()) if balanced_test_counts else 0,
                            'max': max(balanced_test_counts.values()) if balanced_test_counts else 0,
                            'mean': np.mean(list(balanced_test_counts.values())) if balanced_test_counts else 0,
                            'std': np.std(list(balanced_test_counts.values())) if balanced_test_counts else 0
                        }
                    }
                }
                
            except Exception as e:
                logging.error(f"[BCDM] Error comparing dataset {dataset_name}: {str(e)}")
        
        return comparison
    
    def get_incremental_subset(
        self,
        task: int,
        source: str = "train",
        cumulative: bool = False,
        mode = None,
        transform = None
    ) -> Dataset:
        """
        获取增量子集
        
        当 enable_incremental_split=True 时，专门用于获取增量拆分后的子集数据；
        当 enable_incremental_split=False 时，自动回退到使用 get_subset 方法。
        这样可以统一使用该方法，无需根据配置动态选择。
        
        Args:
            task: 任务索引
            source: 数据源（"train"或"test"）
            cumulative: 是否返回累积数据（包含之前所有任务的数据）
            mode: 模式（"train"或"test"）
            transform: 数据转换
            
        Returns:
            数据集对象
        """
        # 如果启用了增量拆分，直接使用 get_subset（此时已经是拆分后的格式）
        # 如果未启用增量拆分，回退到使用 get_subset
        return self.get_subset(task, source, cumulative=cumulative, mode=mode, transform=transform)
    
    def get_original_dataset_splits(self, original_dataset_name: str) -> List[int]:
        """
        获取原始数据集的所有拆分索引
        
        Args:
            original_dataset_name: 原始数据集名称
            
        Returns:
            拆分索引列表
        """
        if not self.enable_incremental_split:
            return []
        
        split_indices = []
        for i, dataset in enumerate(self.datasets):
            if dataset.get('original_dataset_name') == original_dataset_name:
                split_indices.append(i)
        
        return split_indices
    
    def get_incremental_statistics(self) -> Dict[str, Dict]:
        """
        获取增量拆分的统计信息
        
        Returns:
            统计信息字典
        """
        if not self.enable_incremental_split:
            logging.warning("Incremental split is not enabled")
            return {}
        
        stats = {}
        original_datasets = {}
        
        # 按原始数据集分组
        for i, dataset in enumerate(self.datasets):
            original_name = dataset.get('original_dataset_name', dataset['name'])
            if original_name not in original_datasets:
                original_datasets[original_name] = []
            original_datasets[original_name].append((i, dataset))
        
        # 计算每个原始数据集的统计
        for original_name, datasets in original_datasets.items():
            stats[original_name] = {
                'num_splits': len(datasets),
                'total_classes': sum(d['num_classes'] for _, d in datasets),
                'total_train_samples': sum(len(d['train_data']) for _, d in datasets),
                'total_test_samples': sum(len(d['test_data']) for _, d in datasets),
                'splits': []
            }
            
            for task_id, dataset in datasets:
                split_info = {
                    'task_id': task_id,
                    'split_index': dataset.get('split_index', 0),
                    'num_classes': dataset['num_classes'],
                    'train_samples': len(dataset['train_data']),
                    'test_samples': len(dataset['test_data']),
                    'class_names': dataset['class_names']
                }
                stats[original_name]['splits'].append(split_info)
        
        return stats


def create_balanced_data_manager(dataset_names: List[str],
                               balanced_datasets_root: str = "balanced_datasets",
                               **kwargs) -> BalancedCrossDomainDataManagerCore:
    """
    创建平衡后的数据管理器
    
    Args:
        dataset_names: 数据集名称列表
        balanced_datasets_root: 平衡数据集根目录
        **kwargs: 其他参数传递给BalancedCrossDomainDataManagerCore
        
    Returns:
        BalancedCrossDomainDataManagerCore实例
    """
    return BalancedCrossDomainDataManagerCore(
        dataset_names=dataset_names,
        balanced_datasets_root=balanced_datasets_root,
        **kwargs
    )

def main():
    """测试函数"""
    # 默认数据集列表
    default_datasets = [
        'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101',
        'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist',
        'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
    ]
    
    # 创建平衡数据管理器（启用增量拆分）
    manager = create_balanced_data_manager(
        dataset_names=default_datasets[:2],  # 只测试前2个数据集
        balanced_datasets_root="balanced_datasets",
        use_balanced_datasets=True,
        enable_incremental_split=True,
        num_incremental_splits=3,
        incremental_split_seed=42
    )
    
    # 获取统计信息
    stats = manager.get_balanced_statistics()
    print("平衡后数据集统计信息:")
    for dataset_name, stat in stats.items():
        print(f"{dataset_name}:")
        print(f"  训练样本: {stat['total_train_samples']}, 测试样本: {stat['total_test_samples']}")
        print(f"  训练每类: min={stat['train_per_class']['min']}, max={stat['train_per_class']['max']}")
        print(f"  测试每类: min={stat['test_per_class']['min']}, max={stat['test_per_class']['max']}")
    
    # 获取增量拆分统计
    incremental_stats = manager.get_incremental_statistics()
    print("\n增量拆分统计信息:")
    for original_name, stat in incremental_stats.items():
        print(f"\n原始数据集: {original_name}")
        print(f"  拆分数: {stat['num_splits']}")
        print(f"  总类别: {stat['total_classes']}")
        print(f"  总训练样本: {stat['total_train_samples']}")
        print(f"  总测试样本: {stat['total_test_samples']}")
        for split in stat['splits']:
            print(f"    拆分 {split['split_index']} (任务 {split['task_id']}): "
                  f"{split['num_classes']} 类别, {split['train_samples']} 训练样本, {split['test_samples']} 测试样本")


if __name__ == "__main__":
    main()