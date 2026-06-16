"""Incremental dataset manager for CLIP fine-tuning built on utils.data1 datasets."""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from utils.data1 import SimpleDataset, get_dataset

__all__ = ["ClipIncrementalDataManager"]


@dataclass(frozen=True)
class _ClassInfo:
    dataset_name: str
    class_name: str
    templates: Tuple[Callable, ...]
    local_label: int


@dataclass
class _DatasetEntry:
    name: str
    dataset: object
    label_map: Dict[int, int]
    local_labels: List[int]
    class_names: List[str]
    templates: Tuple[Callable, ...]
    train_size: int
    test_size: int

    def global_labels(self) -> List[int]:
        return [
            global_label
            for _, global_label in sorted(self.label_map.items(), key=lambda kv: kv[0])
        ]


class _RemappedDataset(Dataset):
    def __init__(
        self,
        base_dataset: SimpleDataset,
        label_map: Dict[int, int],
        class_infos: Sequence[_ClassInfo],
        dataset_name: str,
    ) -> None:
        self._base = base_dataset
        self._label_map = label_map
        self._class_infos = class_infos
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        image, local_label, _ = self._base[idx]
        global_label = self._label_map[int(local_label)]
        class_info = self._class_infos[global_label]
        return image, global_label, class_info.class_name


class _ConcatRemappedDatasets(Dataset):
    def __init__(self, datasets: Sequence[_RemappedDataset]) -> None:
        if not datasets:
            raise ValueError("datasets must not be empty")
        self._datasets = list(datasets)
        total = 0
        self._prefix: List[int] = []
        for ds in self._datasets:
            total += len(ds)
            self._prefix.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("index out of range")
        dataset_idx = bisect.bisect_right(self._prefix, idx)
        prev_total = self._prefix[dataset_idx - 1] if dataset_idx > 0 else 0
        return self._datasets[dataset_idx][idx - prev_total]


TransformSpec = Union[
    None,
    Callable,
    Dict[Union[str, int], Callable],
]


class ClipIncrementalDataManager:
    """Incremental dataset manager where each dataset forms one CLIP fine-tuning task."""

    def __init__(
        self,
        dataset_names: Sequence[str],
        shuffle: bool = False,
        seed: int = 0,
        log_level: int = logging.INFO,
    ) -> None:
        if not dataset_names:
            raise ValueError("dataset_names must not be empty")

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(log_level)

        self._requested_names = list(dataset_names)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        order = list(range(len(self._requested_names)))
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(order)

        self._ordered_indices = order
        self._ordered_names = [self._requested_names[i] for i in order]
        self._entries: List[_DatasetEntry] = []
        self._class_infos: List[_ClassInfo] = []

        self._build_entries()
        self._name_to_index = {name: idx for idx, name in enumerate(self._ordered_names)}

    def __len__(self) -> int:
        return self.nb_tasks

    @property
    def nb_tasks(self) -> int:
        return len(self._entries)

    @property
    def num_classes(self) -> int:
        return len(self._class_infos)

    @property
    def task_names(self) -> List[str]:
        return list(self._ordered_names)

    @property
    def class_names(self) -> List[str]:
        return [info.class_name for info in self._class_infos]

    @property
    def class_templates(self) -> List[Tuple[Callable, ...]]:
        return [info.templates for info in self._class_infos]

    def get_task_index(self, dataset_name: str) -> int:
        try:
            return self._name_to_index[dataset_name]
        except KeyError as exc:
            raise KeyError(
                f"Dataset {dataset_name!r} is not managed by ClipIncrementalDataManager"
            ) from exc

    def get_task_metadata(self, task_id: int) -> Dict[str, Union[str, int, List[int]]]:
        idx = self._normalize_task_index(task_id)
        entry = self._entries[idx]
        return {
            "dataset_name": entry.name,
            "train_size": entry.train_size,
            "test_size": entry.test_size,
            "num_classes": len(entry.local_labels),
            "global_labels": entry.global_labels(),
        }

    def get_task_labels(self, task_id: int, cumulative: bool = False) -> List[int]:
        indices = self._resolve_task_indices(task_id, cumulative)
        labels: List[int] = []
        for idx in indices:
            labels.extend(self._entries[idx].global_labels())
        return labels

    def get_task_class_names(
        self,
        task_id: int,
        cumulative: bool = False,
    ) -> List[str]:
        return [self._class_infos[label].class_name for label in self.get_task_labels(task_id, cumulative)]

    def get_task_templates(
        self,
        task_id: int,
        cumulative: bool = False,
    ) -> List[Tuple[Callable, ...]]:
        return [self._class_infos[label].templates for label in self.get_task_labels(task_id, cumulative)]

    def get_dataset_templates(self, task_id: int) -> Tuple[Callable, ...]:
        entry = self._entries[self._normalize_task_index(task_id)]
        return entry.templates

    def get_class_info(self, global_label: int) -> _ClassInfo:
        if global_label < 0 or global_label >= len(self._class_infos):
            raise IndexError(f"global_label {global_label} out of range")
        return self._class_infos[global_label]

    def get_task_dataset(
        self,
        task_id: int,
        source: str = "train",
        cumulative: bool = False,
        transform: TransformSpec = None,
    ) -> Dataset:
        if source not in {"train", "test"}:
            raise ValueError("source must be 'train' or 'test'")

        indices = self._resolve_task_indices(task_id, cumulative)
        segments: List[_RemappedDataset] = []

        for idx in indices:
            entry = self._entries[idx]
            base_transform = self._select_transform(transform, entry.name, idx)
            base_dataset = entry.dataset.build_dataset(
                train=(source == "train"),
                transform=base_transform,
            )
            segments.append(
                _RemappedDataset(
                    base_dataset=base_dataset,
                    label_map=entry.label_map,
                    class_infos=self._class_infos,
                    dataset_name=entry.name,
                )
            )

        if not segments:
            raise RuntimeError("no datasets selected for the provided task parameters")

        if len(segments) == 1:
            return segments[0]

        return _ConcatRemappedDatasets(segments)

    def _build_entries(self) -> None:
        global_label = 0

        for name in self._ordered_names:
            dataset = get_dataset(name)
            train_targets = getattr(dataset, "train_targets", None)
            test_targets = getattr(dataset, "test_targets", None)

            unique_labels = self._collect_unique_labels(train_targets, test_targets)
            if not unique_labels:
                class_names_source = getattr(dataset, "class_names", None)
                if class_names_source:
                    unique_labels = list(range(len(class_names_source)))

            if not unique_labels:
                raise ValueError(f"Dataset {name!r} does not expose any class labels")

            class_names = self._collect_class_names(dataset, name, unique_labels)
            if len(class_names) != len(unique_labels):
                raise ValueError(
                    f"Dataset {name!r} class_names length mismatch with labels "
                    f"({len(class_names)} vs {len(unique_labels)})"
                )

            templates = tuple(getattr(dataset, "templates", None) or ())
            label_map: Dict[int, int] = {}
            local_labels: List[int] = []

            for offset, local_label in enumerate(unique_labels):
                local_label_int = int(local_label)
                label_map[local_label_int] = global_label
                local_labels.append(local_label_int)

                self._class_infos.append(
                    _ClassInfo(
                        dataset_name=name,
                        class_name=class_names[offset],
                        templates=templates,
                        local_label=local_label_int,
                    )
                )
                global_label += 1

            train_size = int(len(train_targets)) if train_targets is not None else 0
            test_size = int(len(test_targets)) if test_targets is not None else 0

            entry = _DatasetEntry(
                name=name,
                dataset=dataset,
                label_map=label_map,
                local_labels=local_labels,
                class_names=class_names,
                templates=templates,
                train_size=train_size,
                test_size=test_size,
            )
            self._entries.append(entry)
            self._logger.info(
                "[ClipIDM] dataset=%s | train=%d | test=%d | classes=%d",
                name,
                train_size,
                test_size,
                len(local_labels),
            )

    @staticmethod
    def _collect_unique_labels(
        train_targets,
        test_targets,
    ) -> List[int]:
        labels = set()

        if train_targets is not None:
            labels.update(int(x) for x in np.unique(train_targets))
        if test_targets is not None:
            labels.update(int(x) for x in np.unique(test_targets))

        return sorted(labels)

    @staticmethod
    def _collect_class_names(
        dataset,
        dataset_name: str,
        unique_labels: Sequence[int],
    ) -> List[str]:
        class_names_source = getattr(dataset, "class_names", None)

        if class_names_source:
            names = list(class_names_source)
            mapped: List[str] = []
            for label in unique_labels:
                idx = int(label)
                if 0 <= idx < len(names):
                    mapped.append(names[idx])
                else:
                    mapped.append(f"{dataset_name}:{idx}")
            return mapped

        return [f"{dataset_name}:{int(label)}" for label in unique_labels]

    def _normalize_task_index(self, task_id: int) -> int:
        if task_id < 0:
            task_id += len(self._entries)
        if task_id < 0 or task_id >= len(self._entries):
            raise IndexError(f"task_id {task_id} out of range")
        return task_id

    def _resolve_task_indices(self, task_id: int, cumulative: bool) -> List[int]:
        idx = self._normalize_task_index(task_id)
        if cumulative:
            return list(range(0, idx + 1))
        return [idx]

    @staticmethod
    def _select_transform(
        transform: TransformSpec,
        dataset_name: str,
        task_idx: int,
    ):
        if transform is None:
            return None
        if callable(transform):
            return transform
        if isinstance(transform, dict):
            if dataset_name in transform:
                return transform[dataset_name]
            if task_idx in transform:
                return transform[task_idx]
            return None
        raise TypeError("transform must be a callable, dict, or None")
