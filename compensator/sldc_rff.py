import torch
import torch.nn.functional as F
import logging
from compensator.gaussian_statistics import GaussianStatistics
from compensator.base_compensator import BaseCompensator
import math

logger = logging.getLogger(__name__)


class RFFDriftCompensator(BaseCompensator):
    """基于核化注意力（FAVOR+）的语义漂移补偿器"""

    def __init__(
        self,
        input_dim: int,
        device="cuda",
        compensate_cov: bool = True,
        random_feature_dim=512,  # m in FAVOR+
        use_orthogonal=False,     # 是否使用正交随机特征（更稳定）
    ):
        super().__init__(input_dim, device)
        self.compensate_cov = compensate_cov
        self.random_feature_dim = random_feature_dim
        self.use_orthogonal = use_orthogonal
        self.drift_vectors = None
        self.features_before = None
        self.features_after = None
        self._omega = None  # 随机投影矩阵 (d, m)

    def _create_omega(self, d, m, device):
        """生成随机或正交的投影矩阵 omega ~ N(0, I)"""
        if self.use_orthogonal and d >= m:
            # 生成正交矩阵（QR 分解）
            random_mat = torch.randn(d, d, device=device)
            q, _ = torch.linalg.qr(random_mat)
            omega = q[:, :m]
        else:
            omega = torch.randn(d, m, device=device)
        # 缩放：FAVOR+ 要求 omega ~ N(0, I / tau^2)，但我们通过 temperature 外部控制
        return omega

    def _compute_kernelized_attention(self, queries, keys, drift, temperature=0.1, top_k=1000):
        """
        计算核化注意力并应用漂移补偿
        Args:
            queries: 查询向量 [B, d]
            keys: 键向量 [N, d]
            drift: 漂移向量 [N, d]
            temperature: 温度参数
            top_k: top-k 稀疏化
            
        Returns:
            补偿后的查询向量 [B, d]
        """
        B = queries.size(0)
        N = keys.size(0)
        
        # 归一化
        queries_norm = F.normalize(queries, dim=1)
        keys_norm = F.normalize(keys, dim=1)
        
        # 缩放
        temperature = math.sqrt(temperature)
        scale = 1.0 / temperature
        q_scaled = queries_norm * scale  # [B, d]
        k_scaled = keys_norm * scale     # [N, d]
        
        # 投影到随机特征空间
        q_proj = q_scaled @ self._omega  # [B, m]
        k_proj = k_scaled @ self._omega  # [N, m]
        
        # 随机傅里叶特征
        phi_q = torch.cat([torch.sin(q_proj), torch.cos(q_proj)], dim=-1)  # [B, 2m]
        phi_k = torch.cat([torch.sin(k_proj), torch.cos(k_proj)], dim=-1)  # [N, 2m]
        
        # 计算注意力
        k_sum = phi_k.sum(dim=0, keepdim=True)  # [1, 2m]
        denominator = phi_q @ k_sum.t()         # [B, 1]
        numerator = phi_q @ phi_k.t()           # [B, N]
        
        att = numerator / (denominator + 1e-8)  # [B, N]
        att = torch.clamp(att, min=0.0)
        att = att / (att.sum(dim=1, keepdim=True) + 1e-12)
        
        # top-k 稀疏化（可选）
        if top_k > 0 and top_k < N:
            k = min(top_k, N)
            top_vals, top_indices = torch.topk(att, k, dim=1, sorted=False)
            mask = torch.zeros_like(att)
            mask.scatter_(1, top_indices, top_vals)
            att = mask / mask.sum(dim=1, keepdim=True).clamp(min=1e-12)
        
        # 应用漂移补偿
        drift_applied = torch.einsum('bn,nd->bd', att, drift)
        compensated_queries = queries + drift_applied
        
        return compensated_queries

    def train(self, features_before, features_after):
        self.features_before = features_before.to(self.device)
        self.features_after = features_after.to(self.device)
        self.drift_vectors = self.features_after - self.features_before
        d = features_before.shape[1]
        self._omega = self._create_omega(d, self.random_feature_dim, device=self.device)
        self.is_trained = True

    @torch.no_grad()
    def compensate(
        self,
        stats_dict,
        base_temperature=0.1,
        top_k=1000,
        n_samples=2000,
    ):
        assert self.is_trained, "KernelizedAttentionCompensator 尚未训练"

        out = {}
        fb = self.features_before  # [N, d]
        drift = self.drift_vectors  # [N, d]
        N, d = fb.size()

        global_eps = torch.randn(n_samples, d, device=self.device)

        for cid, stat in stats_dict.items():
            mu = stat.mean.to(self.device)
            cov = stat.cov.to(self.device)
            
            samples = stat.sample(cached_eps=global_eps).to(self.device)  # [n_samples, d]
            compensated_samples = self._compute_kernelized_attention(
                queries=samples,
                keys=fb,
                drift=drift,
                temperature=base_temperature,
                top_k=top_k
            )
            
            # 计算新的统计量
            compensated_samples = compensated_samples.cpu()
            mu_new = 0.9 * compensated_samples.mean(dim=0) + 0.1 * mu.cpu()
            cov_new = 0.9 * torch.cov(compensated_samples.T) + 0.1 * cov.cpu()
            out[cid] = GaussianStatistics(mu_new, cov_new, stat.reg)

        return out
