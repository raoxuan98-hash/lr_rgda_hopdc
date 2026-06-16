# compensator/base_compensator.py
from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn.functional as F
from compensator.gaussian_statistics import GaussianStatistics


class BaseCompensator(ABC):
    """所有补偿器的抽象基类：包含 train() + compensate() 两阶段"""

    def __init__(self, input_dim: int, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.input_dim = input_dim
        self.device = device
        self.is_trained = False

    @abstractmethod
    def train(self, features_before: torch.Tensor, features_after: torch.Tensor):
        """拟合补偿模型"""
        pass

    @abstractmethod
    def compensate(self, stats_dict: Dict[int, GaussianStatistics]) -> Dict[int, GaussianStatistics]:
        """对高斯分布统计进行补偿"""
        pass


class RFFDriftCompensator(BaseCompensator):
    """
    基于 RFF 的残差预测补偿器：
      - 学习映射: phi(x_before) → drift = x_after - x_before
      - 补偿时: x_comp = x + W^T phi(x)
    """

    def __init__(
        self,
        input_dim: int,
        rff_dim: int = 2048,
        gamma: float = 1e-4,
        compensate_cov: bool = True,
        device="cuda"
    ):
        super().__init__(input_dim, device)
        self.rff_dim = rff_dim
        self.gamma = gamma
        self.compensate_cov = compensate_cov

        # RFF parameters (fixed)
        self.omega = torch.randn(rff_dim, input_dim, device=device)
        self.bias = torch.rand(rff_dim, device=device) * 2 * torch.pi

        self.W = None  # shape: (rff_dim, input_dim), so that drift ≈ phi(x) @ W

    def _rff_map(self, x):
        """x: (..., d) → phi(x): (..., D)"""
        proj = x @ self.omega.t() + self.bias  # (..., D)
        phi = torch.cos(proj) * (2.0 / self.rff_dim) ** 0.5
        return phi

    def train(self, features_before, features_after):
        X = features_before.to(self.device)
        Y = features_after.to(self.device)
        drift = Y - X  # (N, d)

        # Normalize features (optional but recommended for RFF)
        X = F.normalize(X, dim=1)

        # Map to RFF space
        Phi_X = self._rff_map(X)  # (N, D)

        # Solve: Phi_X @ W ≈ drift  →  W = argmin ||Phi_X W - drift||^2 + γ||W||^2
        # (D, d) solution
        try:
            ATA = Phi_X.t() @ Phi_X + self.gamma * torch.eye(self.rff_dim, device=self.device)
            ATb = Phi_X.t() @ drift
            W = torch.linalg.solve(ATA, ATb)  # (D, d)
        except RuntimeError:
            W = torch.linalg.pinv(Phi_X) @ drift

        self.W = W
        self.is_trained = True
        return self.W

    @torch.no_grad()
    def compensate(
        self,
        stats_dict,
        n_samples=2000,
        chunk_size=512,
    ):
        assert self.is_trained, "RFFDriftCompensator 尚未训练"
        W = self.W  # (D, d) on device
        out = {}

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)  # (d,)
            cov = stat.cov.to(self.device)  # (d, d)

            # --- 均值补偿 ---
            mu_norm = F.normalize(mu.unsqueeze(0), dim=1).squeeze(0)  # (d,)
            phi_mu = self._rff_map(mu_norm.unsqueeze(0))  # (1, D)
            drift_mu = (phi_mu @ W).squeeze(0)  # (d,)
            mu_new = mu + drift_mu  # (d,)

            if not self.compensate_cov:
                out[cid] = GaussianStatistics(mu_new.cpu(), cov.cpu(), stat.reg)
                continue

            # --- 协方差补偿：采样 + 残差预测 ---
            compensated_samples = []

            # 生成随机噪声（可复用，但每次类独立也可以）
            global_eps = torch.randn(n_samples, self.input_dim, device=self.device)

            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                eps_chunk = global_eps[i:end]  # (chunk, d)

                # 从原始高斯分布采样
                samples = stat.sample(cached_eps=eps_chunk).to(self.device)  # (chunk, d)
                samples_norm = F.normalize(samples, dim=1)  # (chunk, d)

                # RFF 映射
                phi_samples = self._rff_map(samples_norm)  # (chunk, D)

                # 预测残差
                drift_pred = phi_samples @ W  # (chunk, d)

                # 补偿样本
                compensated_chunk = samples + drift_pred  # (chunk, d)
                compensated_samples.append(compensated_chunk.cpu())

            compensated_samples = torch.cat(compensated_samples, dim=0)  # (n_samples, d)
            mu_est = compensated_samples.mean(dim=0)
            cov_est = torch.cov(compensated_samples.T)
            mu_final = 0.9 * mu_est + 0.1 * mu_new.cpu()
            cov_final = 0.9 * cov_est + 0.1 * cov.cpu()
            out[cid] = GaussianStatistics(mu_final, cov_final, stat.reg)

        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out
