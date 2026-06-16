# compensator/linear_compensator.py
import torch
import torch.nn.functional as F
import math
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator


class LinearCompensator(BaseCompensator):
    """线性补偿器: W = (XᵀX + λI)^(-1) XᵀY"""
    def __init__(self, input_dim: int, gamma: float = 1e-4, temp: float = 1.0, device="cuda"):
        super().__init__(input_dim, device)
        self.gamma = gamma
        self.temp = temp
        self.W = None
    def train(self, features_before, features_after, normalize=True):
        X, Y = features_before.to(self.device), features_after.to(self.device)
        if normalize:
            X, Y = F.normalize(X, dim=1), F.normalize(Y, dim=1)
        n, d = X.size()
        XTX = X.T @ X + self.gamma * torch.eye(d, device=self.device)
        XTY = X.T @ Y
        W = torch.linalg.solve(XTX, XTY)
        w = math.exp(-n / (self.temp * d))
        self.W = (1 - w) * W + w * torch.eye(d, device=self.device)
        self.is_trained = True
        return self.W

    @torch.no_grad()
    def compensate(self, stats_dict):
        assert self.is_trained, "LinearCompensator 尚未训练"
        W, WT = self.W.cpu(), self.W.t().cpu()
        out = {}
        for cid, s in stats_dict.items():
            mu_new = s.mean @ W
            cov_new = WT @ s.cov @ W + 1e-3 * torch.eye(s.cov.size(0))
            out[cid] = GaussianStatistics(mu_new, cov_new, s.reg)
        return out