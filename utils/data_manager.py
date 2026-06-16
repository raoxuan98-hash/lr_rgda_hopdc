from typing import Iterable, Optional

from torch.utils.data import Dataset

from utils.data_manager1 import IncrementalDataManager
from utils.cross_domain_data_manager import CrossDomainDataManagerCore


from typing import Optional, List
from torch.utils.data import Dataset

class WithinDomainDataManager:
    def __init__(self, dataset_name: str, seed: int, init_cls: int, increment: int, args: dict = None, shuffle: bool = True):
        self.args = args if args is not None else {}
        self._idm = IncrementalDataManager(
            dataset_name=dataset_name,
            initial_classes=init_cls,
            increment_classes=increment,
            shuffle=shuffle,
            seed=seed)
        self.dataset_name = dataset_name
        self.num_samples_per_task_for_evaluation = self.args.get('num_samples_per_task_for_evaluation', 0)

    @property
    def nb_tasks(self) -> int:
        return self._idm.nb_tasks

    def get_task_size(self, task: int) -> int:
        return self._idm.get_task_size(task)

    def get_task_classes(self, task_id: int, cumulative: bool = False):
        return self._idm.get_task_classes(task_id, cumulative=cumulative)

    def _prepare_subset(self, subset: Dataset, mode: Optional[str]) -> Dataset:
        if mode is not None:
            transform = self._idm._build_transform(mode)
            setattr(subset, "transform", transform)
        return subset

    def get_subset(self, task: int, source: str, cumulative: bool = False, mode: Optional[str] = None) -> Dataset:
        transform = self._idm._build_transform(mode) if mode is not None else None
        subset = self._idm.get_subset(task, source=source, cumulative=cumulative, transform=transform)
        
        # Apply sampling for evaluation if needed
        if source == "test" and self.num_samples_per_task_for_evaluation > 0:
            subset = self._sample_subset(subset, self.num_samples_per_task_for_evaluation)
            
        return self._prepare_subset(subset, mode)
    
    def get_incremental_subset(self, task: int, source: str, cumulative: bool = False, mode: Optional[str] = None) -> Dataset:
        return self.get_subset(task, source, cumulative, mode)
    
    def _sample_subset(self, dataset: Dataset, num_samples: int) -> Dataset:
        """Randomly sample a subset from the dataset"""
        import random
        import numpy as np
        
        # Set seed for reproducibility
        random.seed(self._idm.seed)
        np.random.seed(self._idm.seed)
        
        total_size = len(dataset)
        if total_size <= num_samples:
            return dataset
        
        # Randomly sample indices
        indices = np.random.choice(total_size, size=num_samples, replace=False)
        from torch.utils.data import Subset
        return Subset(dataset, indices)


class CrossDomainDataManager:
    def __init__(self, dataset_name: str, shuffle: bool, seed: int, args: dict = None):
        self.args = args if args is not None else {}
        dataset_names = self.args.get('cross_domain_datasets', [
            'caltech-101', 'dtd', 'eurosat_clip', 'fgvc-aircraft-2013b-variants102',
            'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets',
            'stanford-cars', 'imagenet-r'])
        
        self._cdm = CrossDomainDataManagerCore(
            dataset_names=dataset_names,
            shuffle=shuffle,
            seed=seed,
            num_samples_per_task_for_evaluation=self.args.get('num_samples_per_task_for_evaluation', 0))
        self.dataset_name = dataset_name
        self.num_samples_per_task_for_evaluation = self.args.get('num_samples_per_task_for_evaluation', 0)

    @property
    def nb_tasks(self) -> int:
        return self._cdm.nb_tasks

    def get_task_size(self, task: int) -> int:
        return self._cdm.get_task_size(task)

    def get_task_classes(self, task_id: int, cumulative: bool = False):
        return self._cdm.get_task_classes(task_id, cumulative=cumulative)

    def _prepare_subset(self, subset: Dataset, mode: Optional[str]) -> Dataset:
        if mode is not None:
            transform = self._cdm._build_transform(mode)
            setattr(subset, "transform", transform)
        return subset

    def get_subset(self, task: int, source: str, cumulative: bool = False, mode: Optional[str] = None) -> Dataset:
        transform = self._cdm._build_transform(mode) if mode is not None else None
        subset = self._cdm.get_subset(task, source=source, cumulative=cumulative, transform=transform)
        
        # Apply sampling for evaluation if needed
        if source == "test" and self.num_samples_per_task_for_evaluation > 0:
            subset = self._sample_subset(subset, self.num_samples_per_task_for_evaluation)
            
        return self._prepare_subset(subset, mode)
    
    def _sample_subset(self, dataset: Dataset, num_samples: int) -> Dataset:
        """Randomly sample a subset from the dataset"""
        import random
        import numpy as np
        
        # Set seed for reproducibility
        random.seed(self._cdm.seed)
        np.random.seed(self._cdm.seed)
        
        total_size = len(dataset)
        if total_size <= num_samples:
            return dataset
        
        # Randomly sample indices
        indices = np.random.choice(total_size, size=num_samples, replace=False)
        from torch.utils.data import Subset
        return Subset(dataset, indices)