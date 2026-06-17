# -*- coding: utf-8 -*-
import copy
import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compensator.gaussian_statistics import GaussianStatistics
from compensator.statistics_collector import DistributionStatisticsCollector
from compensator.sldc_linear import LinearCompensator
from compensator.sldc_weaknonlinear import WeakNonlinearCompensator
from compensator.sldc_attention import HopfieldDistributionCompensator
from compensator.base_compensator import RFFDriftCompensator
from compensator.rff_hopdc import RFFLinearAttentionHopDC


def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")


class DistributionCompensator:
    """
    负责从特征对 (features_before, features_after) 构建多种补偿变体。
    输出 variants: Dict[str, Dict[int, GaussianStatistics]]
    """
    def __init__(
        self,
        device: str = "cuda",
        auxiliary_data_size: int = 1024,
        compensator_types = None,
        feature_combination_type: str = "combined",
        hopfield_temp: float = 0.1,
        hopfield_topk: int = 1000,
        rgda_num_centers: int = 1,
        rgda_gmm_k: int = 4,
        rgda_gmm_backend: str = "sklearn_spherical",
        rff_hopdc_dim: int = 1024,
        rff_hopdc_gamma: float = 5.0,
        rff_hopdc_feature_mode: str = "cos_positive",
        rff_hopdc_compensate_cov: bool = False,
        rff_hopdc_den_eps: float = 1e-6,
        rff_hopdc_drift_clip: float = 0.0,
        rff_hopdc_cov_samples: int = 128,
        rff_hopdc_seed: int = 42,
    ):
        self.device = device
        self.auxiliary_data_size = auxiliary_data_size
        self.feature_combination_type = feature_combination_type
        self.hopfield_temp = hopfield_temp
        self.hopfield_topk = hopfield_topk
        self.rgda_num_centers = max(1, int(rgda_num_centers))
        self.rgda_gmm_k = max(0, int(rgda_gmm_k))
        self.rgda_gmm_backend = str(rgda_gmm_backend).lower()
        self.rff_hopdc_dim = int(rff_hopdc_dim)
        self.rff_hopdc_gamma = float(rff_hopdc_gamma)
        self.rff_hopdc_feature_mode = str(rff_hopdc_feature_mode)
        self.rff_hopdc_compensate_cov = bool(rff_hopdc_compensate_cov)
        self.rff_hopdc_den_eps = float(rff_hopdc_den_eps)
        self.rff_hopdc_drift_clip = float(rff_hopdc_drift_clip)
        self.rff_hopdc_cov_samples = int(rff_hopdc_cov_samples)
        self.rff_hopdc_seed = int(rff_hopdc_seed)
        
        # 补偿器类型控制
        if compensator_types is None:
            self.compensator_types = ["SeqFT", "SeqFT + HopDC"]
        else:
            self.compensator_types = self._normalize_compensator_types(compensator_types)

        # 特征和缓存相关
        self.feature_dim = None
        self.cached_Z = None
        self.aux_loader = None
        self.statistics_collector = DistributionStatisticsCollector(
            device=device,
            feature_combination_type=feature_combination_type,
            rgda_num_centers=self.rgda_num_centers,
            rgda_gmm_k=self.rgda_gmm_k,
            rgda_gmm_backend=self.rgda_gmm_backend,
        )

        # 变体存储
        self.variants = self._initialize_variants()

    def _normalize_compensator_types(self, compensator_types):
        aliases = {
            "SeqFT": "SeqFT",
            "SeqFT + linear": "SeqFT + linear",
            "SeqFT + weaknonlinear": "SeqFT + weaknonlinear",
            "SeqFT + Hopfield": "SeqFT + HopDC",
            "SeqFT + HopDC": "SeqFT + HopDC",
            "SeqFT + rff": "SeqFT + rff",
            "SeqFT + RFF-HopDC": "SeqFT + RFF-HopDC",
            "SeqFT + RFFHopDC": "SeqFT + RFF-HopDC",
            "SeqFT + LinearHopDC": "SeqFT + RFF-HopDC",
        }
        normalized = []
        for comp_type in compensator_types:
            canonical = aliases.get(comp_type)
            if canonical is None:
                raise ValueError(f"Unsupported compensator type: {comp_type}")
            if canonical not in normalized:
                normalized.append(canonical)

        if "SeqFT" not in normalized:
            normalized.insert(0, "SeqFT")
        return normalized

    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """获取当前GPU显存信息"""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
        }

    def _log_memory_usage(self, operation_name: str, start_memory: Dict[str, float], end_memory: Dict[str, float]):
        """记录显存使用情况"""
        allocated_diff = end_memory["allocated"] - start_memory["allocated"]
        reserved_diff = end_memory["reserved"] - start_memory["reserved"]
        
        logging.info(
            f"[GPU Memory] {operation_name}: "
            f"Allocated={end_memory['allocated']:.2f}GB "
            f"(+{allocated_diff:.2f}GB), "
            f"Reserved={end_memory['reserved']:.2f}GB "
            f"(+{reserved_diff:.2f}GB), "
            f"Max={end_memory['max_allocated']:.2f}GB"
        )

    def _initialize_variants(self) -> Dict[str, Dict]:
        """根据compensator_types初始化指定的补偿变体结构"""
        variants = {}
        for comp_type in self.compensator_types:
            variants[comp_type] = {}
        return variants

    # ============================================================
    #                  特征抽取与统计构建
    # ============================================================
    
    @torch.no_grad()
    def extract_features_before_after(
        self, model_before,  model_after, data_loader, return_features_before= True,
    ):
        return self.statistics_collector.extract_features_before_after(
            model_before,
            model_after,
            data_loader,
            return_features_before=return_features_before,
        )

    @torch.no_grad()
    def _extract_combined_features(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        current_loader: DataLoader):
        return self.statistics_collector.extract_combined_features(
            model_before,
            model_after,
            current_loader,
        )

    def _build_gaussian_statistics(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        low_rank = False,
    ):
        return self.statistics_collector.build_gaussian_statistics(
            features,
            labels,
            low_rank=low_rank,
        )
    
    def _compute_linear_transform(
        self,
        f_before: torch.Tensor,
        f_after: torch.Tensor,
        gamma: float = 0.1,
        temp: float = 1.0
    ) -> LinearCompensator:
        """计算线性变换补偿器"""
        operation_name = "LinearCompensator"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()
        
        compensator = LinearCompensator(
            input_dim=f_before.size(1),
            gamma=gamma,
            temp=temp,
            device=self.device,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        
        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        
        # 记录时间损耗
        log_time_usage(operation_name, start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)
        
        return compensator

    def _compute_weaknonlinear_transform(
        self,
        f_before: torch.Tensor,
        f_after: torch.Tensor
    ) -> WeakNonlinearCompensator:
        """计算弱非线性变换补偿器"""
        operation_name = "WeakNonlinearCompensator"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()
        
        compensator = WeakNonlinearCompensator(
            input_dim=f_before.size(1),
            device=self.device,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        
        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        
        # 记录时间损耗
        log_time_usage(operation_name, start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)
        
        return compensator

    def _compute_attention_transform(
        self,
        f_before: torch.Tensor,
        f_after: torch.Tensor
    ) -> HopfieldDistributionCompensator:
        """计算HopDC补偿器"""
        operation_name = "HopDC"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()
        
        compensator = HopfieldDistributionCompensator(
            input_dim=f_before.size(1),
            device=self.device,
            base_temperature=self.hopfield_temp,
            top_k=self.hopfield_topk
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        
        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        
        # 记录时间损耗
        log_time_usage(operation_name, start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)
        
        return compensator

    def _compute_rff_transform(
        self,
        f_before: torch.Tensor,
        f_after: torch.Tensor,
        rff_dim: int = 2048,
        gamma: float = 1e-4,
        compensate_cov: bool = True
    ) -> RFFDriftCompensator:
        """计算 RFF 变换补偿器"""
        operation_name = "RFFDriftCompensator"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()
        
        compensator = RFFDriftCompensator(
            input_dim=f_before.size(1),
            rff_dim=rff_dim,
            gamma=gamma,
            compensate_cov=compensate_cov,
            device=self.device,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))
        
        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        
        # 记录时间损耗
        log_time_usage(operation_name, start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)
        
        return compensator

    def _compute_rff_hopdc_transform(
        self,
        f_before: torch.Tensor,
        f_after: torch.Tensor,
    ) -> RFFLinearAttentionHopDC:
        """Compute scalable RFF linear-attention HopDC."""
        operation_name = "RFF-HopDC"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()

        compensator = RFFLinearAttentionHopDC(
            input_dim=f_before.size(1),
            device=self.device,
            random_feature_dim=self.rff_hopdc_dim,
            gamma=self.rff_hopdc_gamma,
            feature_mode=self.rff_hopdc_feature_mode,
            compensate_cov=self.rff_hopdc_compensate_cov,
            den_eps=self.rff_hopdc_den_eps,
            drift_clip=self.rff_hopdc_drift_clip,
            cov_samples=self.rff_hopdc_cov_samples,
            seed=self.rff_hopdc_seed,
        )
        compensator.train(f_before.to(self.device), f_after.to(self.device))

        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        log_time_usage(operation_name, start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)

        return compensator

    def _update_variants_with_transforms(
        self,
        task_id: int,
        current_stats: Dict[int, GaussianStatistics],
        combined_before: torch.Tensor,
        combined_after: torch.Tensor
    ):
        """根据compensator_types使用指定的变换更新变体"""
        if task_id <= 1:
            # 对于第一个任务，直接使用当前统计量初始化所有变体
            for variant_key in self.variants:
                if variant_key != "SeqFT":  # SeqFT已经在主流程中更新
                    self.variants[variant_key].update(copy.deepcopy(current_stats))
            return
            
        # 只计算需要的变换
        transforms = {}
        if "SeqFT + linear" in self.compensator_types:
            transforms["linear"] = self._compute_linear_transform(combined_before, combined_after)

        if "SeqFT + weaknonlinear" in self.compensator_types:
            transforms["weaknonlinear"] = self._compute_weaknonlinear_transform(combined_before, combined_after)
            
        if "SeqFT + HopDC" in self.compensator_types:
            transforms["HopDC"] = self._compute_attention_transform(combined_before, combined_after)
            
        if "SeqFT + rff" in self.compensator_types:
            transforms["rff"] = self._compute_rff_transform(combined_before, combined_after)

        if "SeqFT + RFF-HopDC" in self.compensator_types:
            transforms["RFF-HopDC"] = self._compute_rff_hopdc_transform(
                combined_before, combined_after)
        
        # 应用变换到现有统计量并更新
        for transform_name, transform in transforms.items():
            variant_key = f"SeqFT + {transform_name}"
            
            # 对现有统计量进行补偿
            start_memory = self._get_gpu_memory_info()
            start_time = time.time()
            # logging.info(f"[GPU Memory] Starting compensation for {variant_key}...")
            
            compensated_existing_stats = transform.compensate(self.variants[variant_key])
            
            end_time = time.time()
            end_memory = self._get_gpu_memory_info()
            allocated_diff = end_memory["allocated"] - start_memory["allocated"]
            reserved_diff = end_memory["reserved"] - start_memory["reserved"]
            
            # 记录时间损耗
            log_time_usage(f"Compensation for {variant_key}", start_time, end_time)
            
            logging.info(
                f"[GPU Memory] Compensation for {variant_key} completed: "
                f"Allocated={end_memory['allocated']:.2f}GB (+{allocated_diff:.2f}GB), "
                f"Reserved={end_memory['reserved']:.2f}GB (+{reserved_diff:.2f}GB)"
            )
            
            # 添加当前任务的统计量
            compensated_existing_stats.update(copy.deepcopy(current_stats))
            
            self.variants[variant_key] = compensated_existing_stats

    # ============================================================
    #                  主入口方法
    # ============================================================
    
    def build_current_only(
        self,
        task_id: int,
        model_after: nn.Module,
        data_loader: DataLoader, 
        low_rank: bool = True):
        if task_id <= 0:
            raise ValueError("task_id must be positive after current_task_id increment")

        current_stats = self.statistics_collector.collect_current_stats(
            model_after,
            data_loader,
            low_rank=low_rank,
        )
        seqft_stats = self.variants.get("SeqFT", {})
        seqft_stats.update(copy.deepcopy(current_stats))
        self.variants = {"SeqFT": seqft_stats}
        logging.info(
            "[INFO] Distribution statistics collected without drift compensation for task %d.",
            task_id,
        )
        return self.variants

    def build_all_variants(
        self,
        task_id: int,
        model_before: nn.Module,
        model_after: nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, Dict[int, GaussianStatistics]]:
        # 参数验证
        if task_id <= 0:
            raise ValueError("task_id must be non-negative")
        
        operation_name = "build_all_variants"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()
        logging.info(f"[GPU Memory] Starting {operation_name} for task {task_id}...")
            
        # 一次性提取所有需要的特征，避免重复计算
        (current_before, current_after, current_labels, aux_before, aux_after,
         combined_before, combined_after) = self._extract_combined_features(
            model_before, model_after, data_loader)
        
        # 初始化特征维度缓存
        if self.feature_dim is None:
            self.feature_dim = current_after.size(1)
            self.cached_Z = torch.randn(50000, self.feature_dim)
        
        # 构建当前任务的统计量
        current_stats = self._build_gaussian_statistics(current_after, current_labels)
        
        # 更新基础变体
        self.variants["SeqFT"].update(copy.deepcopy(current_stats))
        
        
        if self.feature_combination_type == "combined":
            f_before = combined_before
            f_after = combined_after
        elif self.feature_combination_type == "aux_only":
            f_before = aux_before
            f_after = aux_after
        elif self.feature_combination_type == "current_only":
            f_before = current_before
            f_after = current_after
        else:
            raise ValueError(f"Invalid feature_combination_type: {self.feature_combination_type}")

        # 确保f_before和f_after不为None才调用_update_variants_with_transforms
        if f_before is not None and f_after is not None:
            self._update_variants_with_transforms(
                task_id, current_stats, f_before, f_after)
        else:
            logging.warning(f"Skipping transform computation for task {task_id} due to missing features")
        
        logging.info(f"[INFO] DistributionCompensator built {len(self.variants)} variants for task {task_id}.")
        
        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        
        # 记录时间损耗
        log_time_usage(f"{operation_name} for task {task_id}", start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)
            
        return self.variants

    def set_auxiliary_loader(self, aux_loader: DataLoader):
        """设置辅助数据加载器"""
        self.aux_loader = aux_loader
        self.statistics_collector.set_auxiliary_loader(aux_loader)

    def clear_cache(self):
        """清除缓存"""
        self.cached_Z = None

    def get_variant_names(self) -> list:
        """获取所有变体名称"""
        return list(self.variants.keys())
