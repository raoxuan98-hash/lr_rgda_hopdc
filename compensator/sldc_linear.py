# compensator/linear_compensator.py
import torch
import torch.nn.functional as F
import math
from compensator.gaussian_statistics import make_gaussian_statistics_like
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
            centers_new = s.centers @ W if getattr(s, "centers", None) is not None else None
            gmm_means_new = (
                s.gmm_means @ W if getattr(s, "gmm_means", None) is not None else None
            )
            gmm_diag_vars_new = None
            if getattr(s, "gmm_diag_vars", None) is not None:
                gmm_diag_vars_new = s.gmm_diag_vars @ (W ** 2)
            out[cid] = make_gaussian_statistics_like(
                s,
                mu_new,
                cov_new,
                centers=centers_new,
                gmm_means=gmm_means_new,
                gmm_diag_vars=gmm_diag_vars_new,
            )
        return out
