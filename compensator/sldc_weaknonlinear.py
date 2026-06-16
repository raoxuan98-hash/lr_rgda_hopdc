# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator

class ResidMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc1.weight.data = torch.eye(dim)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False))
        
        self.alphas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x):
        scale = torch.exp(self.log_scale)
        weights = F.softmax(self.alphas / scale, dim=0)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return weights[0] * y1 + weights[1] * y2

    def reg_loss(self):
        weights = F.softmax(self.alphas, dim=0)
        return (weights[0] - 1.0) ** 2

class WeakNonlinearCompensator(BaseCompensator):
    """弱非线性补偿器 (Residual MLP)"""
    def __init__(self, input_dim: int, device="cuda"):
        super().__init__(input_dim, device)
        self.net = ResidMLP(input_dim).to(self.device)

    def train(self, features_before: torch.Tensor, features_after: torch.Tensor, epochs: int = 4000, lr: float = 0.001):
        device = features_before.device
        self.net = self.net.to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 3)
        crit = nn.MSELoss()
        X = features_before
        Y = features_after
        for ep in range(epochs):
            idx = torch.randint(0, X.size(0), (64,), device=device)
            pred = self.net(X[idx])
            loss = crit(pred, Y[idx]) + 0.5 * self.net.reg_loss()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if (ep + 1) % 2000 == 0:
                logging.info(f"[SLDC WeakNonlinearTransform] step {ep+1}/{epochs}, loss={loss.item():.6f}")
        self.is_trained = True

    @torch.no_grad()
    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_trained:
            raise ValueError("非线性变换器尚未训练。")
        return self.net(features)

    @torch.no_grad()
    def compensate(self, stats_dict, n_samples=5000):
        assert self.is_trained, "WeakNonlinearCompensator 尚未训练"
        device = self.device
        out = {}
        for cid, s in stats_dict.items():
            samples = s.sample(n_samples).to(device)
            transformed = self.net(samples)
            mu_new = transformed.mean(0).cpu()
            cov_new = torch.cov(transformed.T).cpu()
            out[cid] = GaussianStatistics(mu_new, cov_new, s.reg)
            
            # 清理临时变量
            del transformed, mu_new, cov_new
            
        return out