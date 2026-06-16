# compensator/sdc_compensator.py

import torch
import torch.nn.functional as F
import logging
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator

logger = logging.getLogger(__name__)


class HopfieldDistributionCompensator(BaseCompensator):
    """Hopfield-based Distribution Compensator (HopDC)."""

    def __init__(self, input_dim: int, device="cuda", compensate_cov: bool = True,
                 base_temperature=0.1, top_k=1000):
        super().__init__(input_dim, device)
        self.compensate_cov = compensate_cov
        self.base_temperature = base_temperature
        self.top_k = top_k
        self.drift_vectors = None
        self.features_before = None
        self.features_after = None

    def train(self, features_before, features_after):
        self.features_before = features_before.to(self.device)
        self.features_after = features_after.to(self.device)
        self.drift_vectors = self.features_after - self.features_before
        self.is_trained = True

    def _apply_drift_to_points(self, points, fb_norm, drift, base_temperature, top_k):
        points = points.to(self.device)
        points_norm = F.normalize(points, dim=1)
        att = F.softmax(torch.matmul(points_norm, fb_norm.t()) / base_temperature, dim=1)
        if top_k > 0 and top_k < fb_norm.size(0):
            k = min(top_k, fb_norm.size(0))
            top_vals, top_indices = torch.topk(att, k, dim=1, sorted=False)
            mask = torch.zeros_like(att)
            mask.scatter_(1, top_indices, top_vals)
            att = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return points + torch.einsum('bn,nd->bd', att, drift)

    @torch.no_grad()
    def compensate(
        self,
        stats_dict,
        base_temperature=None,
        top_k=None,
        n_samples=2000,
        chunk_size=512,
    ):
        # 使用实例变量作为默认值，如果参数未提供
        if base_temperature is None:
            base_temperature = self.base_temperature
        if top_k is None:
            top_k = self.top_k
        assert self.is_trained, "HopDC 尚未训练"
        out = {}
        fb = self.features_before  # assumed on device
        drift = self.drift_vectors
        fb_norm = F.normalize(fb, dim=1)
        N, d = fb.size()

        global_eps = torch.randn(n_samples, d, device=self.device)

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)

            # --- 均值补偿 ---
            mu_new = self._apply_drift_to_points(
                mu.unsqueeze(0),
                fb_norm,
                drift,
                base_temperature,
                top_k,
            ).squeeze(0)
            drift_c = mu_new - mu
            similarities = torch.matmul(fb_norm, F.normalize(mu.unsqueeze(0), dim=1).t()).squeeze(1)  # [N]

            # ====== Logging: detailed similarity stats ======
            if logger.isEnabledFor(logging.INFO):
                # Define requested k-values
                k_top10 = min(10, N)
                k_top100 = min(100, N)
                k_bottom1000 = min(1000, N)
                k_bottom500 = min(500, N)

                # Top-K
                top10_sim = torch.topk(similarities, k_top10, largest=True).values.mean().item()
                top100_sim = torch.topk(similarities, k_top100, largest=True).values.mean().item()

                # Bottom-K (smallest similarities)
                bottom1000_sim = torch.topk(similarities, k_bottom1000, largest=False).values.mean().item()
                bottom500_sim = torch.topk(similarities, k_bottom500, largest=False).values.mean().item()

                # logger.info(
                #     f"Class {cid} - "
                #     f"Top-10: {top10_sim:.4f}, "
                #     f"Top-100: {top100_sim:.4f}, "
                #     f"Bottom-1000: {bottom1000_sim:.4f}, "
                #     f"Bottom-500: {bottom500_sim:.4f}"
                # )
            # ==============================================

            if not self.compensate_cov:
                centers_new = None
                if getattr(stat, "centers", None) is not None:
                    centers_new = self._apply_drift_to_points(
                        stat.centers,
                        fb_norm,
                        drift,
                        base_temperature,
                        top_k,
                    ).cpu()
                out[cid] = GaussianStatistics(mu_new.cpu(), cov.cpu(), stat.reg, centers=centers_new)
                continue

            # --- 协方差补偿：复用 global_eps，分块处理 ---
            compensated_samples = []
            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                eps_chunk = global_eps[i:end]

                samples = stat.sample(cached_eps=eps_chunk).to(self.device)  # [chunk, d]
                samples_norm = F.normalize(samples, dim=1)
                sim = torch.matmul(samples_norm, fb_norm.t())  # [chunk, N]
                att = F.softmax(sim / base_temperature, dim=1)

                if top_k > 0 and top_k < N:
                    k = min(top_k, N)
                    top_vals, top_indices = torch.topk(att, k, dim=1, sorted=False)
                    mask = torch.zeros_like(att)
                    mask.scatter_(1, top_indices, top_vals)
                    att = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-12)

                drift_applied = torch.einsum('bn,nd->bd', att, drift)
                compensated_chunk = samples + drift_applied
                compensated_samples.append(compensated_chunk.cpu())

            compensated_samples = torch.cat(compensated_samples, dim=0)
            mu_new = 0.9 * compensated_samples.mean(dim=0) + 0.1 * mu.cpu()
            cov_new = 0.9 * torch.cov(compensated_samples.T) + 0.1 * cov.cpu()
            centers_new = None
            if getattr(stat, "centers", None) is not None:
                centers_new = self._apply_drift_to_points(
                    stat.centers,
                    fb_norm,
                    drift,
                    base_temperature,
                    top_k,
                ).cpu()
            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg, centers=centers_new)

        return out


class AttentionCompensator(HopfieldDistributionCompensator):
    """Backward-compatible alias for older SLDC/Hopfield scripts."""

    pass
