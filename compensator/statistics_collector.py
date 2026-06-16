# -*- coding: utf-8 -*-
import logging
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compensator.gaussian_statistics import (
    GaussianStatistics,
    LowRankGaussianStatistics,
    fit_diag_gmm_statistics,
    fit_spherical_gmm_statistics,
    kmeans_centers,
)


def log_time_usage(operation_name: str, start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")


class DistributionStatisticsCollector:
    """Collect features and build compact per-class distribution statistics."""

    def __init__(
        self,
        device: str = "cuda",
        feature_combination_type: str = "combined",
        rgda_num_centers: int = 1,
        rgda_gmm_k: int = 4,
        rgda_gmm_backend: str = "sklearn_spherical",
    ):
        self.device = device
        self.feature_combination_type = feature_combination_type
        self.rgda_num_centers = max(1, int(rgda_num_centers))
        self.rgda_gmm_k = max(0, int(rgda_gmm_k))
        self.rgda_gmm_backend = str(rgda_gmm_backend).lower()
        if self.rgda_gmm_backend not in {"sklearn_spherical", "kmeans_diag"}:
            raise ValueError(f"Unsupported rgda_gmm_backend: {rgda_gmm_backend}")
        self.aux_loader: Optional[DataLoader] = None
        self.feature_dim: Optional[int] = None

    def set_auxiliary_loader(self, aux_loader: DataLoader):
        self.aux_loader = aux_loader

    def _get_gpu_memory_info(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def _log_memory_usage(
        self,
        operation_name: str,
        start_memory: Dict[str, float],
        end_memory: Dict[str, float],
    ):
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

    @torch.no_grad()
    def extract_features_before_after(
        self,
        model_before: Optional[nn.Module],
        model_after: nn.Module,
        data_loader: DataLoader,
        return_features_before: bool = True,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        operation_name = "extract_features_before_after"
        start_memory = self._get_gpu_memory_info()
        start_time = time.time()

        if model_before is not None:
            model_before.eval()
            model_before.to(self.device)

        model_after.eval()
        model_after.to(self.device)

        feats_before, feats_after, labels = [], [], []
        for batch in data_loader:
            inputs = batch[0].to(self.device)
            targets = batch[1]

            if return_features_before:
                if model_before is None:
                    raise ValueError("model_before is required when return_features_before=True")
                if hasattr(model_before, "forward_features"):
                    before = model_before.forward_features(inputs, random_projection=True)
                else:
                    before = model_before(inputs)
                feats_before.append(before.cpu())

            if hasattr(model_after, "forward_features"):
                after = model_after.forward_features(inputs, random_projection=True)
            else:
                after = model_after(inputs)
            feats_after.append(after.cpu())
            labels.append(targets)

        result = (
            torch.cat(feats_before) if return_features_before else None,
            torch.cat(feats_after),
            torch.cat(labels),
        )

        end_time = time.time()
        end_memory = self._get_gpu_memory_info()
        log_time_usage(operation_name, start_time, end_time)
        self._log_memory_usage(operation_name, start_memory, end_memory)
        return result

    @torch.no_grad()
    def extract_combined_features(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        current_loader: DataLoader,
    ):
        return_features_before = self.feature_combination_type in ["combined", "current_only"]
        current_before, current_after, current_labels = self.extract_features_before_after(
            model_before,
            model_after,
            current_loader,
            return_features_before=return_features_before,
        )

        if current_before is not None:
            current_cosine_sim = torch.nn.functional.cosine_similarity(
                current_before, current_after, dim=1).mean()
            current_norm_diff = (current_before - current_after).norm(p=2, dim=1).mean()
            logging.info(
                "[Feature Analysis] Current features - Cosine similarity: %.4f, Norm difference: %.4f",
                current_cosine_sim,
                current_norm_diff,
            )

        combined_before, combined_after = current_before, current_after
        aux_before, aux_after = None, None

        if self.aux_loader is not None and self.feature_combination_type in ["combined", "aux_only"]:
            aux_before, aux_after, _ = self.extract_features_before_after(
                model_before,
                model_after,
                self.aux_loader,
                return_features_before=True,
            )
            if current_before is not None:
                combined_before = torch.cat([current_before, aux_before])
                combined_after = torch.cat([current_after, aux_after])
            else:
                combined_before = aux_before
                combined_after = aux_after

            aux_cosine_sim = torch.nn.functional.cosine_similarity(
                aux_before, aux_after, dim=1).mean()
            aux_norm_diff = (aux_before - aux_after).norm(p=2, dim=1).mean()
            logging.info(
                "[Feature Analysis] Auxiliary features - Cosine similarity: %.4f, Norm difference: %.4f",
                aux_cosine_sim,
                aux_norm_diff,
            )

        return (
            current_before,
            current_after,
            current_labels,
            aux_before,
            aux_after,
            combined_before,
            combined_after,
        )

    def build_gaussian_statistics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        low_rank: bool = False,
    ):
        features = features.cpu()
        labels = labels.cpu()
        unique_labels = torch.unique(labels)

        stats = {}
        for lbl in unique_labels:
            mask = labels == lbl
            feats_class = features[mask]
            mu = feats_class.mean(0)
            if feats_class.size(0) >= 2:
                cov = torch.cov(feats_class.T)
            else:
                cov = torch.eye(feats_class.size(1)) * 1e-4

            if low_rank:
                stats[int(lbl.item())] = LowRankGaussianStatistics(mu, cov)
                continue

            centers = None
            if self.rgda_num_centers > 1:
                centers = kmeans_centers(
                    feats_class,
                    self.rgda_num_centers,
                    seed=42 + int(lbl.item()),
                )

            gmm_means, gmm_diag_vars, gmm_weights = None, None, None
            if self.rgda_gmm_k > 0:
                if self.rgda_gmm_backend == "sklearn_spherical":
                    gmm_means, gmm_diag_vars, gmm_weights = fit_spherical_gmm_statistics(
                        feats_class,
                        self.rgda_gmm_k,
                        seed=42,
                    )
                else:
                    gmm_means, gmm_diag_vars, gmm_weights = fit_diag_gmm_statistics(
                        feats_class,
                        self.rgda_gmm_k,
                        seed=1042 + int(lbl.item()),
                    )

            stats[int(lbl.item())] = GaussianStatistics(
                mu,
                cov,
                centers=centers,
                gmm_means=gmm_means,
                gmm_diag_vars=gmm_diag_vars,
                gmm_weights=gmm_weights,
            )

        return stats

    def collect_current_stats(
        self,
        model_after: nn.Module,
        data_loader: DataLoader,
        low_rank: bool = False,
    ):
        _, current_features, current_labels = self.extract_features_before_after(
            None,
            model_after,
            data_loader,
            return_features_before=False,
        )
        self.feature_dim = current_features.size(1)
        return self.build_gaussian_statistics(
            current_features,
            current_labels,
            low_rank=low_rank,
        )
