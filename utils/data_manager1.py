import logging
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data1 import SimpleDataset, get_dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class IncrementalDataManager:
    def __init__(
        self,
        dataset_name: str,
        initial_classes: int,
        increment_classes: int,
        shuffle: bool = True,
        seed: int = 0,
        log_level: int = logging.INFO) -> None:
        
        logging.basicConfig(level=log_level)
        
        self.dataset_name = dataset_name
        self.initial_classes = int(initial_classes)
        self.increment_classes = int(increment_classes)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self._load_idata()
        self._build_class_order()
        self._remap_all_labels()
        self._increment_classess = self._build_task_schedule()
        assert self.initial_classes <= self.num_classes, "initial_classes is larger than total classes"

    @property
    def num_classes(self) -> int:
        return len(self._class_order)

    @property
    def nb_tasks(self) -> int:
        return len(self._increment_classess)

    @property
    def class_order(self) -> List[int]:
        """Original class ids order mapped to new labels [0..C-1]."""
        return list(self._class_order)

    def get_task_size(self, task_id: int) -> int:
        return int(self._increment_classess[task_id])

    # ------------------------- Core functions -------------------------
    def _load_idata(self) -> None:
        idata = get_dataset(self.dataset_name)

        # Load arrays
        self._train_data = np.asarray(idata.train_data)
        self._test_data = np.asarray(idata.test_data)
        self._train_targets = np.asarray(idata.train_targets, dtype=np.int64)
        self._test_targets = np.asarray(idata.test_targets, dtype=np.int64)

        self.use_path = bool(getattr(idata, "use_path", False))
        self.class_names = list(getattr(idata, "class_names", []) or []) or None
        self.templates = list(getattr(idata, "templates", []) or []) or None

        if getattr(idata, "class_order", None) is not None:
            self._orig_class_order = list(idata.class_order)
        else:
            uniq = np.unique(self._train_targets)
            self._orig_class_order = [int(x) for x in sorted(uniq.tolist())]

        logging.info(
            "[IDM] load dataset %s | train=%d, test=%d, classes=%d",
            self.dataset_name,
            len(self._train_targets),
            len(self._test_targets),
            len(self._orig_class_order))

    def _build_class_order(self) -> None:
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            self._class_order = rng.permutation(self._orig_class_order).tolist()
        else:
            self._class_order = list(self._orig_class_order)
        logging.info("[IDM] class_order: %s", self._class_order)

    def _remap_all_labels(self) -> None:
        mapping: Dict[int, int] = {orig: new for new, orig in enumerate(self._class_order)}
        self._train_targets = np.asarray([mapping[int(y)] for y in self._train_targets], dtype=np.int64)
        self._test_targets = np.asarray([mapping[int(y)] for y in self._test_targets], dtype=np.int64)

    def _build_task_schedule(self) -> List[int]:
        total = len(self._class_order)

        # Special case: no incremental classes requested.
        if self.increment_classes <= 0:
            if self.initial_classes < total:
                logging.warning(
                    "[IDM] increment_classes=%d is non-positive; "
                    "only the initial %d classes will be used out of %d.",
                    self.increment_classes,
                    self.initial_classes,
                    total,
                )
            incs = [self.initial_classes if self.initial_classes > 0 else total]
            logging.info("[IDM] increment_classess=%s (nb_tasks=%d)", incs, len(incs))
            return incs

        incs = [self.initial_classes]
        remain = total - self.initial_classes
        while remain > 0:
            step = min(self.increment_classes, remain)
            if step <= 0:
                break
            incs.append(step)
            remain -= step
        logging.info("[IDM] increment_classess=%s (nb_tasks=%d)", incs, len(incs))
        return incs

    def get_task_classes(self, task_id: int, cumulative: bool = False) -> List[int]:
        """
        Return classes under new label space for a given task.
        cumulative=False: classes newly introduced at this task.
        cumulative=True : all classes up to this task.
        """
        assert 0 <= task_id < self.nb_tasks
        if cumulative:
            end = sum(self._increment_classess[: task_id + 1])
            return list(range(0, end))
        else:
            start = sum(self._increment_classess[:task_id])
            end = start + self._increment_classess[task_id]
            return list(range(start, end))

    def get_subset(
        self,
        task_id: int,
        source: str = "train",
        cumulative: bool = False,
        transform = None) -> Dataset:

        cls_indices = self.get_task_classes(task_id, cumulative=cumulative)
        data, targets = self._select_by_classes(source, cls_indices)
        
        return SimpleDataset(
            images=data,
            labels=targets,
            use_path=self.use_path,
            class_names=self.class_names,
            templates=self.templates,
            transform=transform)
    
    def _select_by_classes(self, source: str, cls_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        if source == "train":
            data, targets = self._train_data, self._train_targets
        elif source == "test":
            data, targets = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown source: {source}")
        mask = np.isin(targets, np.asarray(cls_indices, dtype=np.int64))
        return data[mask], targets[mask]

    # --------- DataManager-compatible API ---------
    def _build_transform(self, mode: str) -> transforms.Compose:
        # CIFAR100_224使用CIFAR专用归一化参数
        if self.dataset_name == "cifar100_224":
            return self._build_cifar100_224_transform(mode)
        # CUB200_224和Cars196_224使用Resize+Crop策略
        # elif self.dataset_name in ["cub200_224", "cars196_224"]:
        #     return self._build_resize_crop_transform(mode)
        # ImageFolder benchmarks use ImageNet-size preprocessing.
        elif self.dataset_name in ["cub200_224", "cars196_224", "imagenet-r", "imagenet-a", "vtab"]:
            return self._build_imagenet_r_transform(mode)
            # return self._build_resize_crop_transform(mode)
        # 其他数据集使用默认ImageNet标准预处理
        else:
            return self._build_default_transform(mode)
    
    def _build_cifar100_224_transform(self, mode: str) -> transforms.Compose:
        """CIFAR100_224专用预处理，匹配SLCA"""
        cifar_mean = (0.5071, 0.4867, 0.4408)
        cifar_std = (0.2675, 0.2565, 0.2761)
        
        if mode == "train":
            ops = [
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),  # BICUBIC插值
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),        # 保留ColorJitter
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std)
            ]
        elif mode == "test":
            ops = [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),              # BICUBIC插值
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(cifar_mean, cifar_std)
            ]
        else:
            raise ValueError(f"Unknown mode {mode}.")
        
        return transforms.Compose(ops)
    
    def _build_resize_crop_transform(self, mode: str) -> transforms.Compose:
        """CUB200_224和Cars196_224专用预处理，匹配SLCA的Resize+Crop策略"""
        if mode == "train":
            ops = [
                transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC),      # 先resize到300x300
                transforms.RandomCrop((224, 224)),                 # 再随机裁剪
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]
        elif mode == "test":
            ops = [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),              # BICUBIC插值
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]
        else:
            raise ValueError(f"Unknown mode {mode}.")
        
        return transforms.Compose(ops)
    
    def _build_imagenet_r_transform(self, mode: str) -> transforms.Compose:
        """ImageNet-R专用预处理，匹配SLCA（无ColorJitter）"""
        if mode == "train":
            ops = [
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),  # BICUBIC插值
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]
        elif mode == "test":
            ops = [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),              # BICUBIC插值
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]
        else:
            raise ValueError(f"Unknown mode {mode}.")
        
        return transforms.Compose(ops)
    
    def _build_default_transform(self, mode: str) -> transforms.Compose:
        """默认ImageNet标准预处理，用于其他数据集"""
        if mode == "train":
            ops = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
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


if __name__ == "__main__":
    dm = IncrementalDataManager(
        dataset_name="mnist",
        initial_classes=5,
        increment_classes=1,
        shuffle=True,
        seed=1993)

    train_t0 = dm.get_subset(task_id=0, source="train", cumulative=False)
    test_t0 = dm.get_subset(task_id=0, source="test", cumulative=False)
    print("Task0:", len(train_t0), len(test_t0))

    train_t1_cum = dm.get_subset(task_id=1 if dm.nb_tasks > 1 else 0, source="train", cumulative=True)
    print("Task1 cumulative:", len(train_t1_cum))
