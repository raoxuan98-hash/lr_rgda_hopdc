import logging
import math
from typing import Dict

import torch
import torch.nn.functional as F

from compensator.base_compensator import BaseCompensator
from compensator.gaussian_statistics import GaussianStatistics, make_gaussian_statistics_like


logger = logging.getLogger(__name__)


class RFFLinearAttentionHopDC(BaseCompensator):
    """Linearized HopDC with random Fourier features.

    This approximates normalized RBF attention over drift pairs:

        drift(q) = phi(q)^T (Phi(K)^T V) / phi(q)^T (Phi(K)^T 1)

    where K are pre-drift features and V are drift vectors. Unlike the legacy
    RFF attention implementation, compensation does not build a query-by-memory
    attention matrix, so query-time cost depends on the RFF dimension rather
    than the number of memory features.
    """

    def __init__(
        self,
        input_dim: int,
        device: str = "cuda",
        random_feature_dim: int = 1024,
        gamma: float = 5.0,
        feature_mode: str = "cos_positive",
        compensate_cov: bool = False,
        den_eps: float = 1e-6,
        drift_clip: float = 0.0,
        cov_samples: int = 128,
        chunk_size: int = 512,
        seed: int = 42,
    ):
        super().__init__(input_dim, device)
        self.random_feature_dim = max(1, int(random_feature_dim))
        self.gamma = float(gamma)
        self.feature_mode = str(feature_mode).lower()
        self.compensate_cov = bool(compensate_cov)
        self.den_eps = float(den_eps)
        self.drift_clip = float(drift_clip)
        self.cov_samples = max(1, int(cov_samples))
        self.chunk_size = max(1, int(chunk_size))
        self.seed = int(seed)

        if self.feature_mode not in {"sincos", "cos", "cos_positive", "elu"}:
            raise ValueError(
                "feature_mode must be one of: sincos, cos, cos_positive, elu")

        self.omega = None
        self.phase = None
        self.kv = None
        self.k_sum = None
        self.phi_dim = None

    def _init_features(self, dtype: torch.dtype):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        # RBF exp(-gamma ||x-y||^2) uses omega ~ N(0, 2 gamma I).
        scale = math.sqrt(max(2.0 * self.gamma, 1e-12))
        self.omega = torch.randn(
            self.input_dim,
            self.random_feature_dim,
            device=self.device,
            dtype=dtype,
            generator=generator,
        ) * scale
        self.phase = torch.rand(
            self.random_feature_dim,
            device=self.device,
            dtype=dtype,
            generator=generator,
        ) * (2.0 * math.pi)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.omega is None or self.phase is None:
            self._init_features(x.dtype)

        x = F.normalize(x.to(self.device), dim=1)
        proj = x @ self.omega

        if self.feature_mode == "sincos":
            # Match the legacy RFF attention feature map:
            #   phi(x) = [sin(x @ omega), cos(x @ omega)].
            # The constant factor cancels in normalized linear attention.
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.feature_mode == "cos":
            return torch.cos(proj + self.phase) * math.sqrt(2.0 / self.random_feature_dim)
        if self.feature_mode == "cos_positive":
            return (torch.cos(proj + self.phase) + 1.0) / math.sqrt(self.random_feature_dim)
        # Positive linear-attention fallback. This is not an unbiased RBF RFF
        # map, but gives stable non-negative weights for ablations.
        return F.elu(proj) + 1.0

    def train(self, features_before: torch.Tensor, features_after: torch.Tensor):
        keys = features_before.to(self.device)
        values = features_after.to(self.device) - keys

        phi_k = self._phi(keys)
        self.phi_dim = phi_k.size(1)
        self.kv = phi_k.transpose(0, 1) @ values
        self.k_sum = phi_k.sum(dim=0)
        self.is_trained = True
        logger.info(
            "RFF-HopDC trained: N=%d, D=%d, projection_dim=%d, phi_dim=%d, gamma=%.4g, mode=%s, compensate_cov=%s",
            keys.size(0),
            keys.size(1),
            self.random_feature_dim,
            self.phi_dim,
            self.gamma,
            self.feature_mode,
            self.compensate_cov,
        )
        return self

    def _clip_drift(self, drift: torch.Tensor) -> torch.Tensor:
        if self.drift_clip <= 0:
            return drift
        norm = drift.norm(dim=1, keepdim=True).clamp(min=1e-12)
        scale = torch.clamp(self.drift_clip / norm, max=1.0)
        return drift * scale

    @torch.no_grad()
    def predict_drift(self, points: torch.Tensor) -> torch.Tensor:
        assert self.is_trained, "RFFLinearAttentionHopDC is not trained"
        points = points.to(self.device)
        phi_q = self._phi(points)
        numerator = phi_q @ self.kv
        denominator = phi_q @ self.k_sum
        if self.feature_mode in {"cos_positive", "elu"}:
            denominator = denominator.clamp(min=self.den_eps)
        else:
            # Signed RFF features follow the legacy sin/cos map but can produce
            # negative denominators. The efficient associativity trick cannot
            # reproduce the legacy pairwise clamp without materializing [B, N],
            # so we stabilize only the normalization magnitude here.
            denominator = denominator.sign() * denominator.abs().clamp(min=self.den_eps)
        denominator = denominator.unsqueeze(1)
        return self._clip_drift(numerator / denominator)

    @torch.no_grad()
    def _compensate_points(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(self.device)
        return points + self.predict_drift(points)

    @torch.no_grad()
    def _compensate_covariance(
        self,
        stat: GaussianStatistics,
        mu_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cov = stat.cov.to(self.device)
        compensated_samples = []
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        eps = torch.randn(
            self.cov_samples,
            self.input_dim,
            device=self.device,
            generator=generator,
        )
        for start in range(0, self.cov_samples, self.chunk_size):
            end = min(start + self.chunk_size, self.cov_samples)
            samples = stat.sample(cached_eps=eps[start:end]).to(self.device)
            compensated_samples.append(self._compensate_points(samples).cpu())

        samples = torch.cat(compensated_samples, dim=0)
        mu_est = samples.mean(dim=0)
        cov_est = torch.cov(samples.T)
        mu_final = 0.9 * mu_est + 0.1 * mu_new.cpu()
        cov_final = 0.9 * cov_est + 0.1 * cov.cpu()
        return mu_final, cov_final

    @torch.no_grad()
    def compensate(
        self,
        stats_dict: Dict[int, GaussianStatistics],
    ) -> Dict[int, GaussianStatistics]:
        assert self.is_trained, "RFFLinearAttentionHopDC is not trained"
        out = {}

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)

            mu_new = self._compensate_points(mu.unsqueeze(0)).squeeze(0)
            cov_new = cov

            if self.compensate_cov:
                mu_out, cov_out = self._compensate_covariance(stat, mu_new)
            else:
                mu_out, cov_out = mu_new.cpu(), cov_new.cpu()

            centers_new = None
            if getattr(stat, "centers", None) is not None:
                centers_new = self._compensate_points(stat.centers).cpu()

            gmm_means_new = None
            if getattr(stat, "gmm_means", None) is not None:
                gmm_means_new = self._compensate_points(stat.gmm_means).cpu()

            out[cid] = make_gaussian_statistics_like(
                stat,
                mu_out,
                cov_out,
                centers=centers_new,
                gmm_means=gmm_means_new,
            )

        return out
