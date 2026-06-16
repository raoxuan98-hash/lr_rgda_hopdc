import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from classifier.base_classifier_builder import BaseClassifierBuilder

def log_time_usage(operation_name: str, start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")

class TSVDClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, threshold=1e-3, device="cuda"):
        self.device = device
        self.threshold = threshold

    def build(self, stats_dict):
        start_time = time.time()
        
        d = list(stats_dict.values())[0].mean.size(0)
        C = len(stats_dict)
        num_samples_per_class = 1024
        
        Xs, Ys = [], []
        class_ids = sorted(stats_dict.keys())
        for cid in class_ids:
            gs = stats_dict[cid]
            mu, L = gs.mean.to(self.device), gs.L.to(self.device)
            # 独立采样噪声
            Z = torch.randn(num_samples_per_class, d, device=self.device)
            X = mu + Z @ L.t()
            y = torch.full((num_samples_per_class,), int(cid), device=self.device)
            Xs.append(X); Ys.append(y)
            
        X, Y = torch.cat(Xs), torch.cat(Ys)
        Xn = F.normalize(X, dim=1)
        Y_oh = F.one_hot(Y, num_classes=C).float()
        
        # TSVD computation
        U, S, Vh = torch.linalg.svd(Xn, full_matrices=False)
        
        # Truncate small singular values
        S_inv = torch.zeros_like(S)
        mask = S > self.threshold
        S_inv[mask] = 1.0 / S[mask]
        
        # X^+ = V * S^{-1} * U^T
        X_pinv = Vh.T @ torch.diag(S_inv) @ U.T
        W = X_pinv @ Y_oh
        
        model = nn.Sequential(nn.Linear(d, C, bias=False)).to(self.device)
        model[0].weight.data = W.T.clone()
        
        end_time = time.time()
        log_time_usage("TSVD Classifier build", start_time, end_time)
        
        return model.cpu()
