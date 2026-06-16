#!/usr/bin/env python3
"""
Cross-Domain数据集重新划分工具
将每个类别的样本数量限制到128，确保数据集平衡
"""

import os
import sys
import json
import shutil
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data1 import get_dataset
from utils.cross_domain_data_manager import CrossDomainDataManagerCore


class MetadataManager:
    """元数据管理器，记录原始分布和采样过程"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.original_distribution = {}
        self.balanced_distribution = {}
        self.sampling_config = {}
        self.dataset_statistics = {}
        
    def record_original_distribution(self, dataset_name: str, 
                                   train_counts: Dict[int, int], 
                                   test_counts: Dict[int, int],
                                   class_names: List[str]):
        """记录原始数据分布"""
        self.original_distribution[dataset_name] = {
            'train_counts': train_counts,
            'test_counts': test_counts,
            'class_names': class_names,
            'total_train_samples': sum(train_counts.values()),
            'total_test_samples': sum(test_counts.values()),
            'num_classes': len(class_names)
        }
    
    def record_balanced_distribution(self, dataset_name: str,
                                  train_counts: Dict[int, int],
                                  test_counts: Dict[int, int],
                                  sampling_info: Dict[str, Any]):
        """记录平衡后的数据分布"""
        self.balanced_distribution[dataset_name] = {
            'train_counts': train_counts,
            'test_counts': test_counts,
            'total_train_samples': sum(train_counts.values()),
            'total_test_samples': sum(test_counts.values()),
            'sampling_info': sampling_info
        }
    
    def record_sampling_config(self, config: Dict[str, Any]):
        """记录采样配置"""
        self.sampling_config = config
    
    def record_dataset_statistics(self, stats: Dict[str, Any]):
        """记录数据集统计信息"""
        self.dataset_statistics = stats
    
    def save_all_metadata(self):
        """保存所有元数据到文件"""
        # 保存原始分布
        with open(self.metadata_dir / "original_distribution.json", 'w') as f:
            json.dump(self.original_distribution, f, indent=2)
        
        # 保存平衡后分布
        with open(self.metadata_dir / "balanced_distribution.json", 'w') as f:
            json.dump(self.balanced_distribution, f, indent=2)
        
        # 保存采样配置
        with open(self.metadata_dir / "sampling_config.json", 'w') as f:
            json.dump(self.sampling_config, f, indent=2)
        
        # 保存统计信息
        with open(self.metadata_dir / "dataset_statistics.json", 'w') as f:
            json.dump(self.dataset_statistics, f, indent=2)
        
        logging.info(f"元数据已保存到 {self.metadata_dir}")


class DatasetResplitter:
    """数据集重新划分器"""
    
    def __init__(self, 
                 max_samples_per_class: int = 128,
                 seed: int = 42,
                 output_dir: str = "balanced_datasets"):
        """
        初始化数据集重新划分器
        
        Args:
            max_samples_per_class: 每个类别的最大样本数
            seed: 随机种子
            output_dir: 输出目录
        """
        self.max_samples_per_class = max_samples_per_class
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化元数据管理器
        self.metadata_manager = MetadataManager(str(self.output_dir))
        
        # 记录采样配置
        self.metadata_manager.record_sampling_config({
            'max_samples_per_class': max_samples_per_class,
            'seed': seed,
            'output_dir': str(output_dir)
        })
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def resplit_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        重新划分单个数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            采样信息字典
        """
        logging.info(f"开始处理数据集: {dataset_name}")
        
        # 获取数据集
        try:
            dataset = get_dataset(dataset_name)
        except Exception as e:
            logging.error(f"无法加载数据集 {dataset_name}: {str(e)}")
            return {}
        
        # 获取原始数据
        train_data = getattr(dataset, 'train_data', None)
        train_targets = getattr(dataset, 'train_targets', None)
        test_data = getattr(dataset, 'test_data', None)
        test_targets = getattr(dataset, 'test_targets', None)
        class_names = getattr(dataset, 'class_names', [])
        use_path = getattr(dataset, 'use_path', False)
        
        if any(x is None for x in [train_data, train_targets, test_data, test_targets]):
            logging.error(f"数据集 {dataset_name} 缺少必要属性")
            return {}
        
        # 确保数据是numpy数组
        train_data = np.asarray(train_data)
        train_targets = np.asarray(train_targets)
        test_data = np.asarray(test_data)
        test_targets = np.asarray(test_targets)
        
        # 统计原始分布
        train_counts = self._count_samples_by_class(train_targets)
        test_counts = self._count_samples_by_class(test_targets)
        
        # 记录原始分布
        self.metadata_manager.record_original_distribution(
            dataset_name, train_counts, test_counts, class_names
        )
        
        # 执行平衡采样
        balanced_train_data, balanced_train_targets, balanced_test_data, balanced_test_targets, sampling_info = self._balance_dataset(
            train_data, train_targets, test_data, test_targets, len(class_names)
        )
        
        # 统计平衡后分布
        balanced_train_counts = self._count_samples_by_class(balanced_train_targets)
        balanced_test_counts = self._count_samples_by_class(balanced_test_targets)
        
        # 记录平衡后分布
        self.metadata_manager.record_balanced_distribution(
            dataset_name, balanced_train_counts, balanced_test_counts, sampling_info
        )
        
        # 保存平衡后的数据集
        self._save_balanced_dataset(
            dataset_name,
            balanced_train_data, balanced_train_targets,
            balanced_test_data, balanced_test_targets,
            class_names, use_path
        )
        
        logging.info(f"数据集 {dataset_name} 处理完成")
        return sampling_info
    
    def _count_samples_by_class(self, targets: np.ndarray) -> Dict[int, int]:
        """统计每个类别的样本数量"""
        counts = Counter(targets.tolist())
        # 确保键是int类型，避免JSON序列化问题
        return {int(k): int(v) for k, v in counts.items()}
    
    def _balance_dataset(self,
                        train_data: np.ndarray, train_targets: np.ndarray,
                        test_data: np.ndarray, test_targets: np.ndarray,
                        num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        平衡数据集 - 修复测试集为空的问题
        
        策略：
        1. 总样本数 >= 256：训练集128 + 测试集128
        2. 128 <= 总样本数 < 256：训练集64 + 测试集64
        3. 总样本数 < 128：按比例分配，确保训练集和测试集都至少1个样本
        
        Returns:
            (balanced_train_data, balanced_train_targets, balanced_test_data, balanced_test_targets, sampling_info)
        """
        balanced_train_data = []
        balanced_train_targets = []
        balanced_test_data = []
        balanced_test_targets = []
        
        sampling_info = {
            'classes_processed': 0,
            'classes_with_insufficient_samples': [],
            'samples_moved_from_train_to_test': {},
            'final_train_counts': {},
            'final_test_counts': {}
        }
        
        for class_id in range(num_classes):
            # 获取当前类别的训练和测试样本索引
            train_indices = np.where(train_targets == class_id)[0]
            test_indices = np.where(test_targets == class_id)[0]
            
            train_count = len(train_indices)
            test_count = len(test_indices)
            total_count = train_count + test_count
            
            logging.debug(f"类别 {class_id}: 训练样本={train_count}, 测试样本={test_count}, 总计={total_count}")
            
            # 根据总样本数决定分配策略
            if total_count >= self.max_samples_per_class * 2:
                # 情况1：总样本充足（>=256），可以分配128+128
                self._process_sufficient_samples(
                    class_id, train_data, train_indices, test_data, test_indices,
                    balanced_train_data, balanced_train_targets,
                    balanced_test_data, balanced_test_targets,
                    sampling_info
                )
            elif total_count >= self.max_samples_per_class:
                # 情况2：总样本足够（128-256），分配64+64
                self._process_moderate_samples(
                    class_id, train_data, train_indices, test_data, test_indices,
                    balanced_train_data, balanced_train_targets,
                    balanced_test_data, balanced_test_targets,
                    sampling_info
                )
            else:
                # 情况3：总样本不足（<128），按比例分配
                self._process_insufficient_samples(
                    class_id, train_data, train_indices, test_data, test_indices,
                    balanced_train_data, balanced_train_targets,
                    balanced_test_data, balanced_test_targets,
                    sampling_info
                )
            
            sampling_info['classes_processed'] += 1
        
        # 转换为numpy数组
        balanced_train_data = np.array(balanced_train_data)
        balanced_train_targets = np.array(balanced_train_targets)
        balanced_test_data = np.array(balanced_test_data)
        balanced_test_targets = np.array(balanced_test_targets)
        
        # 记录最终分布
        sampling_info['final_train_counts'] = self._count_samples_by_class(balanced_train_targets)
        sampling_info['final_test_counts'] = self._count_samples_by_class(balanced_test_targets)
        
        return balanced_train_data, balanced_train_targets, balanced_test_data, balanced_test_targets, sampling_info
    
    def _process_sufficient_samples(self, class_id: int,
                                   train_data: np.ndarray, train_indices: np.ndarray,
                                   test_data: np.ndarray, test_indices: np.ndarray,
                                   balanced_train_data: List, balanced_train_targets: List,
                                   balanced_test_data: List, balanced_test_targets: List,
                                   sampling_info: Dict[str, Any]):
        """处理样本充足的类别（总样本数 >= 256）"""
        train_count = len(train_indices)
        test_count = len(test_indices)
        
        # 测试集：优先从原始测试集采样128个
        if test_count >= self.max_samples_per_class:
            # 测试集足够，直接采样128个
            sampled_test_indices = np.random.choice(
                test_indices, size=self.max_samples_per_class, replace=False
            )
            balanced_test_data.extend(test_data[sampled_test_indices])
            balanced_test_targets.extend([class_id] * self.max_samples_per_class)
        else:
            # 测试集不足，先从测试集取所有，再从训练集补充
            balanced_test_data.extend(test_data[test_indices])
            needed_from_train = self.max_samples_per_class - test_count
            
            if len(train_indices) >= needed_from_train:
                # 训练集足够补充
                move_indices = np.random.choice(train_indices, size=needed_from_train, replace=False)
                balanced_test_data.extend(train_data[move_indices])
                sampling_info['samples_moved_from_train_to_test'][int(class_id)] = int(needed_from_train)
            else:
                # 训练集也不够，取所有训练样本
                balanced_test_data.extend(train_data[train_indices])
                if len(train_indices) > 0:
                    sampling_info['samples_moved_from_train_to_test'][int(class_id)] = len(train_indices)
            
            balanced_test_targets.extend([class_id] * self.max_samples_per_class)
        
        # 训练集：从剩余的训练样本中采样128个
        if len(train_indices) >= self.max_samples_per_class:
            # 如果之前移动了部分训练样本到测试集，需要排除这些
            moved_count = sampling_info['samples_moved_from_train_to_test'].get(int(class_id), 0)
            if moved_count > 0:
                # 重新获取未被移动的样本索引
                remaining_indices = np.setdiff1d(train_indices, train_indices[:moved_count])
                if len(remaining_indices) >= self.max_samples_per_class:
                    sampled_train = np.random.choice(remaining_indices, size=self.max_samples_per_class, replace=False)
                else:
                    sampled_train = remaining_indices
            else:
                sampled_train = np.random.choice(train_indices, size=self.max_samples_per_class, replace=False)
            
            balanced_train_data.extend(train_data[sampled_train])
            balanced_train_targets.extend([class_id] * len(sampled_train))
        else:
            # 训练集不足128，取所有剩余样本
            balanced_train_data.extend(train_data[train_indices])
            balanced_train_targets.extend([class_id] * train_count)
    
    def _process_moderate_samples(self, class_id: int,
                                 train_data: np.ndarray, train_indices: np.ndarray,
                                 test_data: np.ndarray, test_indices: np.ndarray,
                                 balanced_train_data: List, balanced_train_targets: List,
                                 balanced_test_data: List, balanced_test_targets: List,
                                 sampling_info: Dict[str, Any]):
        """处理样本适中的类别（128 <= 总样本数 < 256）"""
        train_count = len(train_indices)
        test_count = len(test_indices)
        target_samples = self.max_samples_per_class // 2  # 64个
        
        # 测试集：优先从原始测试集采样64个
        if test_count >= target_samples:
            # 测试集足够，采样64个
            sampled_test_indices = np.random.choice(test_indices, size=target_samples, replace=False)
            balanced_test_data.extend(test_data[sampled_test_indices])
        else:
            # 测试集不足，先取所有测试样本，再从训练集补充
            balanced_test_data.extend(test_data[test_indices])
            needed_from_train = target_samples - test_count
            
            if train_count >= needed_from_train:
                move_indices = np.random.choice(train_indices, size=needed_from_train, replace=False)
                balanced_test_data.extend(train_data[move_indices])
                sampling_info['samples_moved_from_train_to_test'][int(class_id)] = int(needed_from_train)
            else:
                # 训练集也不够，取所有训练样本
                balanced_test_data.extend(train_data[train_indices])
                if train_count > 0:
                    sampling_info['samples_moved_from_train_to_test'][int(class_id)] = train_count
        
        balanced_test_targets.extend([class_id] * target_samples)
        
        # 训练集：从剩余的训练样本中采样64个
        moved_count = sampling_info['samples_moved_from_train_to_test'].get(int(class_id), 0)
        if train_count > moved_count:
            remaining_indices = train_indices[moved_count:] if moved_count > 0 else train_indices
            if len(remaining_indices) >= target_samples:
                sampled_train = np.random.choice(remaining_indices, size=target_samples, replace=False)
            else:
                sampled_train = remaining_indices
            
            balanced_train_data.extend(train_data[sampled_train])
            balanced_train_targets.extend([class_id] * len(sampled_train))
        else:
            # 没有剩余训练样本
            pass
    
    def _process_insufficient_samples(self, class_id: int,
                                     train_data: np.ndarray, train_indices: np.ndarray,
                                     test_data: np.ndarray, test_indices: np.ndarray,
                                     balanced_train_data: List, balanced_train_targets: List,
                                     balanced_test_data: List, balanced_test_targets: List,
                                     sampling_info: Dict[str, Any]):
        """处理样本不足的类别（总样本数 < 128）"""
        train_count = len(train_indices)
        test_count = len(test_indices)
        total_count = train_count + test_count
        
        # 记录样本不足的类别
        sampling_info['classes_with_insufficient_samples'].append({
            'class_id': int(class_id),
            'total_samples': int(total_count),
            'desired_samples': int(self.max_samples_per_class)
        })
        
        if total_count == 0:
            # 没有样本，跳过
            return
        
        # 确保至少1个训练样本和1个测试样本（如果可能）
        if total_count == 1:
            # 只有一个样本，分配给训练集
            balanced_train_data.extend(train_data[train_indices] if train_count > 0 else test_data[test_indices])
            balanced_train_targets.extend([class_id])
        else:
            # 按比例分配，但确保两边都至少1个
            if train_count == 0:
                # 只有测试样本，分配1个给训练，其余给测试
                balanced_train_data.extend(test_data[test_indices[0:1]])
                balanced_train_targets.extend([class_id])
                balanced_test_data.extend(test_data[test_indices[1:]])
                balanced_test_targets.extend([class_id] * (test_count - 1))
            elif test_count == 0:
                # 只有训练样本，分配1个给测试，其余给训练
                balanced_test_data.extend(train_data[train_indices[0:1]])
                balanced_test_targets.extend([class_id])
                balanced_train_data.extend(train_data[train_indices[1:]])
                balanced_train_targets.extend([class_id] * (train_count - 1))
            else:
                # 两边都有样本，保持原始比例
                balanced_train_data.extend(train_data[train_indices])
                balanced_train_targets.extend([class_id] * train_count)
                balanced_test_data.extend(test_data[test_indices])
                balanced_test_targets.extend([class_id] * test_count)
    
    def _save_balanced_dataset(self, 
                             dataset_name: str,
                             train_data: np.ndarray, train_targets: np.ndarray,
                             test_data: np.ndarray, test_targets: np.ndarray,
                             class_names: List[str], use_path: bool):
        """保存平衡后的数据集"""
        dataset_dir = self.output_dir / dataset_name
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"
        
        # 创建目录结构
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建类别目录
        for i, class_name in enumerate(class_names):
            (train_dir / str(i)).mkdir(exist_ok=True)
            (test_dir / str(i)).mkdir(exist_ok=True)
        
        # 保存训练数据
        self._save_data_to_directory(train_data, train_targets, train_dir, use_path)
        
        # 保存测试数据
        self._save_data_to_directory(test_data, test_targets, test_dir, use_path)
        
        # 保存类别标签文件
        label_file = dataset_dir / "label.txt"
        with open(label_file, 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        logging.info(f"平衡后的数据集已保存到 {dataset_dir}")
    
    def _save_data_to_directory(self, data: np.ndarray, targets: np.ndarray,
                               base_dir: Path, use_path: bool):
        """将数据保存到目录结构中"""
        if len(data) == 0:
            logging.warning(f"没有数据需要保存到 {base_dir}")
            return
            
        if use_path:
            # 如果数据是路径，直接复制文件
            for i, (sample, label) in enumerate(zip(data, targets)):
                src_path = Path(sample)
                dst_dir = base_dir / str(int(label))  # 确保标签是int
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / f"{i}_{src_path.name}"
                shutil.copy2(src_path, dst_path)
        else:
            # 如果数据是数组，保存为图像文件
            from PIL import Image
            for i, (sample, label) in enumerate(zip(data, targets)):
                dst_dir = base_dir / str(int(label))  # 确保标签是int
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / f"{i:06d}.png"
                if isinstance(sample, np.ndarray):
                    img = Image.fromarray(sample)
                else:
                    # 假设是PIL图像
                    img = sample
                img.save(dst_path)
    
    def resplit_all_datasets(self, dataset_names: List[str]) -> Dict[str, Any]:
        """重新划分所有数据集"""
        logging.info(f"开始处理 {len(dataset_names)} 个数据集")
        
        all_sampling_info = {}
        
        for dataset_name in dataset_names:
            try:
                sampling_info = self.resplit_single_dataset(dataset_name)
                all_sampling_info[dataset_name] = sampling_info
            except Exception as e:
                logging.error(f"处理数据集 {dataset_name} 时出错: {str(e)}")
                all_sampling_info[dataset_name] = {'error': str(e)}
        
        # 生成统计信息
        statistics = self._generate_statistics(all_sampling_info)
        self.metadata_manager.record_dataset_statistics(statistics)
        
        # 保存所有元数据
        self.metadata_manager.save_all_metadata()
        
        logging.info(f"所有数据集处理完成，结果保存在 {self.output_dir}")
        return all_sampling_info
    
    def _generate_statistics(self, all_sampling_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {
            'total_datasets_processed': len(all_sampling_info),
            'datasets_with_errors': 0,
            'classes_with_insufficient_samples': 0,
            'total_samples_moved': 0,
            'balance_improvement': {}
        }
        
        for dataset_name, info in all_sampling_info.items():
            if 'error' in info:
                stats['datasets_with_errors'] += 1
                continue
            
            if 'classes_with_insufficient_samples' in info:
                stats['classes_with_insufficient_samples'] += len(info['classes_with_insufficient_samples'])
            
            if 'samples_moved_from_train_to_test' in info:
                moved_dict = info['samples_moved_from_train_to_test']
                if moved_dict:  # 确保字典不为空
                    stats['total_samples_moved'] += sum(moved_dict.values())
        
        return stats


def main():
    """主函数"""
    # 默认数据集列表
    default_datasets = [
        'cifar100_224', 'cub200_224', 'resisc45', 'imagenet-r', 'caltech-101', 
        'dtd', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 
        'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'
    ]
    
    # 创建重新划分器
    resplitter = DatasetResplitter(
        max_samples_per_class=128,
        seed=42,
        output_dir="balanced_datasets"
    )
    
    # 处理所有数据集
    results = resplitter.resplit_all_datasets(default_datasets)
    
    print("数据集重新划分完成！")
    print(f"结果保存在: balanced_datasets/")
    print(f"元数据保存在: balanced_datasets/metadata/")


if __name__ == "__main__":
    main()