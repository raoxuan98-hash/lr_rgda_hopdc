import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data1 import get_dataset, SimpleDataset, pil_loader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CrossDomainSimpleDataset(Dataset):
    """专门用于跨域场景的Dataset类，正确处理全局偏移标签和类名映射"""
    
    def __init__(self,
                 images,
                 labels,
                 use_path=False,
                 class_names=None,
                 templates=None,
                 transform=None,
                 label_offset=0):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.use_path = use_path
        self.class_names = class_names
        self.templates = templates
        self.transform = transform
        self.label_offset = label_offset
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 检查当前图像是路径还是数组
        if isinstance(self.images[idx], str):
            # 如果是路径，使用pil_loader
            image = pil_loader(self.images[idx])
        else:
            # 如果是数组，直接从数组创建图像
            image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])

        class_name = self.class_names[label] if self.class_names is not None else None
        return image, label, class_name


class CrossDomainDataManagerCore:
    """
    Cross-domain class-incremental data manager.
    Each dataset is treated as a separate task.
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        shuffle: bool = False,
        seed: int = 0,
        num_shots: int = 0,
        num_samples_per_task_for_evaluation: int = 0,
        log_level: int = logging.INFO
    ) -> None:
        
        logging.basicConfig(level=log_level)
        self.dataset_names = dataset_names
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_shots = int(num_shots)
        self.num_samples_per_task_for_evaluation = int(num_samples_per_task_for_evaluation)
        
        # Load all datasets
        self.datasets = []
        self.global_label_offset = []
        self.total_classes = 0
        self.global_class_names = []
        
        for i, dataset_name in enumerate(dataset_names):
            logging.info(f"[CDM] Loading dataset {i+1}/{len(dataset_names)}: {dataset_name}")
            dataset = get_dataset(dataset_name)
            
            # Store dataset info
            dataset_info = {
                'name': dataset_name,
                'train_data': np.asarray(dataset.train_data),
                'test_data': np.asarray(dataset.test_data),
                'train_targets': np.asarray(dataset.train_targets, dtype=np.int64),
                'test_targets': np.asarray(dataset.test_targets, dtype=np.int64),
                'num_classes': len(np.unique(dataset.train_targets)),
                'use_path': bool(getattr(dataset, "use_path", False)),
                'class_names': list(getattr(dataset, "class_names", []) or []),
                'templates': list(getattr(dataset, "templates", []) or [])
            }
            
            # Apply few-shot sampling if num_shots > 0
            if self.num_shots > 0:
                logging.info(f"[CDM] Applying few-shot sampling: {self.num_shots} shots per class")
                dataset_info = self._apply_few_shot_sampling(dataset_info, self.seed + i)
            
            # Apply global label offset
            offset = self.total_classes
            
            # 保持原始标签为 0-based，偏移将在 get_subset 中应用
            # dataset_info['train_targets'] = dataset_info['train_targets'] + offset
            # dataset_info['test_targets'] = dataset_info['test_targets'] + offset
            
            self.datasets.append(dataset_info)
            self.global_label_offset.append(offset)
            self.total_classes += dataset_info['num_classes']
            self.global_class_names.extend(dataset_info['class_names'])
        
        logging.info(f"[CDM] Total datasets: {len(self.datasets)}")
        logging.info(f"[CDM] Total classes: {self.total_classes}")
        logging.info(f"[CDM] Total tasks: {len(self.datasets)}")
    
    @property
    def nb_tasks(self) -> int:
        """Number of tasks (equal to number of datasets)"""
        return len(self.datasets)
    
    @property
    def num_classes(self) -> int:
        """Total number of classes across all datasets"""
        return self.total_classes
    
    def get_task_size(self, task_id: int) -> int:
        """Get number of classes in a specific task"""
        assert 0 <= task_id < self.nb_tasks
        return self.datasets[task_id]['num_classes']
    
    def get_task_classes(self, task_id: int, cumulative: bool = False) -> List[int]:
        """
        Return classes for a given task.
        cumulative=False: classes in this task only.
        cumulative=True: all classes up to this task.
        """
        assert 0 <= task_id < self.nb_tasks
        
        if cumulative:
            end = sum(self.datasets[i]['num_classes'] for i in range(task_id + 1))
            return list(range(0, end))
        else:
            offset = self.global_label_offset[task_id]
            num_classes = self.datasets[task_id]['num_classes']
            return list(range(offset, offset + num_classes))
    
    def get_subset(
        self,
        task: int,
        source: str = "train",
        cumulative: bool = False,
        mode = None,
        transform = None
    ) -> Dataset:
        assert 0 <= task < self.nb_tasks
        assert source in ["train", "test"]

        if mode is not None:
            transform = self._build_transform(mode)
        
        # 计算到当前任务的总类别数（用于class_names）
        total_classes_up_to_task = sum(self.datasets[i]['num_classes'] for i in range(task + 1))
        cumulative_class_names = self.global_class_names[:total_classes_up_to_task]
        
        if cumulative:
            # 累积模式：返回当前任务与先前数据集的拼接数据集
            all_data = []
            all_targets = []
            use_path = False
            templates = []
            
            for i in range(task + 1):
                dataset = self.datasets[i]
                if source == "train":
                    data = dataset['train_data']
                    targets = dataset['train_targets']
                else:
                    data = dataset['test_data']
                    targets = dataset['test_targets']
                

                targets = targets + self.global_label_offset[i]
                
                all_data.extend(data)
                all_targets.extend(targets)
                
                use_path = use_path or dataset['use_path']
                if dataset['templates']:
                    templates.extend(dataset['templates'])

            dataset = CrossDomainSimpleDataset(
                images=all_data,
                labels=np.array(all_targets),
                use_path=use_path,
                class_names=cumulative_class_names,
                templates=templates if templates else None,
                transform=transform,
                label_offset=0
            )

        else:
            # 非累积模式：返回当前任务的数据集
            dataset = self.datasets[task]
            if source == "train":
                data = dataset['train_data']
                targets = dataset['train_targets']
            else:
                data = dataset['test_data']
                targets = dataset['test_targets']
            
            # 标签已经在初始化时添加了全局偏移，不需要再次添加
            targets = targets + self.global_label_offset[task]

            # DEBUG: 记录非累积模式下的数据信息
            targets_array = np.array(targets)
            logging.info(f"[CDM] Non-cumulative mode for task {task}, source {source}:")
            logging.info(f"[CDM]   Total samples: {len(data)}")
            if len(targets_array) > 0:
                logging.info(f"[CDM]   Global label range: min={np.min(targets_array)}, max={np.max(targets_array)}")
            else:
                logging.info(f"[CDM]   Global label range: (empty)")
            logging.info(f"[CDM]   Class names count: {len(cumulative_class_names)}")
            logging.info(f"[CDM]   Expected max valid label: {len(cumulative_class_names) - 1}")
            logging.info(f"[CDM]   Global offset: {self.global_label_offset[task]}")
            
            # 检查标签是否在有效范围内
            if len(cumulative_class_names) > 0:
                max_label = np.max(targets_array)
                if max_label >= len(cumulative_class_names):
                    logging.error(f"[CDM] ERROR: Max label {max_label} exceeds class_names length {len(cumulative_class_names)}")
                    logging.error(f"[CDM] Task {task} ({dataset['name']}): expected range 0-{len(cumulative_class_names)-1}")
                    logging.error(f"[CDM] Actual label range: {np.min(targets_array)}-{max_label}")
                    raise ValueError(f"Label {max_label} exceeds class_names length {len(cumulative_class_names)}")
            
            dataset = CrossDomainSimpleDataset(
                images=data,
                labels=targets,  # 使用加上全局偏移的标签
                use_path=dataset['use_path'],
                class_names=cumulative_class_names,
                templates=dataset['templates'] if dataset['templates'] else None,
                transform=transform,
                label_offset=0
            )
        
        # Apply sampling for evaluation if needed
        # if source == "test" and self.num_samples_per_task_for_evaluation > 0:
        #     dataset = self._sample_subset(dataset, self.num_samples_per_task_for_evaluation)
            
        return dataset
    
    def _sample_subset(self, dataset: Dataset, num_samples: int) -> Dataset:
        """Randomly sample a subset from dataset"""
        import random
        import numpy as np
        from torch.utils.data import Subset
        
        # Set seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        total_size = len(dataset)  # type: ignore
        if total_size <= num_samples:
            return dataset
        
        # Randomly sample indices
        indices = np.random.choice(total_size, size=num_samples, replace=False).tolist()
        return Subset(dataset, indices)
    
    def _build_transform(self, mode: str) -> transforms.Compose:
        """Build data transformation for train or test mode"""
        if mode == "train":
            ops = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]
        elif mode == "test":
            ops = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]
        else:
            raise ValueError(f"Unknown mode {mode}.")
        
        return transforms.Compose(ops)
    
    def _apply_few_shot_sampling(self, dataset_info: Dict, seed: int) -> Dict:
        """
        Apply few-shot sampling to the dataset.
        For each class, randomly sample num_shots samples from the training set.
        
        Args:
            dataset_info: Dictionary containing dataset information
            seed: Random seed for reproducible sampling
            
        Returns:
            Updated dataset_info with sampled training data
        """
        np.random.seed(seed)
        
        train_data = dataset_info['train_data']
        train_targets = dataset_info['train_targets']
        num_classes = dataset_info['num_classes']
        
        sampled_data = []
        sampled_targets = []
        
        for class_id in range(num_classes):
            # Find all samples for this class (before global offset)
            class_mask = train_targets == class_id
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                logging.warning(f"[CDM] No samples found for class {class_id} in dataset {dataset_info['name']}")
                continue
            
            # Randomly sample num_shots samples (or all if not enough)
            num_samples = min(self.num_shots, len(class_indices))
            sampled_indices = np.random.choice(class_indices, size=num_samples, replace=False)
            
            sampled_data.extend(train_data[sampled_indices])
            sampled_targets.extend(train_targets[sampled_indices])
            
            # logging.info(f"[CDM] Class {class_id}: {len(class_indices)} total samples, sampled {num_samples}")
        
        # Update dataset_info with sampled data
        dataset_info['train_data'] = np.array(sampled_data)
        dataset_info['train_targets'] = np.array(sampled_targets)
        
        logging.info(f"[CDM] Few-shot sampling complete: {len(sampled_data)} total samples from {num_classes} classes")
        
        return dataset_info