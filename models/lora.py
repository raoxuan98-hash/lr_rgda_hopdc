# In[]
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Union
from typing import Any

# ----------------------------------------------
#  lora.py
# ----------------------------------------------

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set

class NullSpaceViT(nn.Module):
    """
    Wrapper for a frozen Vision‑Transformer that enables **null‑space adaptation**
    on the linear weights of the attention qkv projection and the two FFN
    linear layers (fc1, fc2) of the selected blocks, together with the final
    LayerNorm weight.
    Only these weights are trainable; every bias and every other parameter
    (including all other LayerNorms) stays frozen.
    """
    def __init__(
        self,
        vit_model: nn.Module,
        nullspace_layer: Optional[List[int]] = None,
        use_projection: bool = True,
    ):
        super().__init__()
        # ---------- 1️⃣ 记录要适配的模块 ----------
        self.lora_modules = nn.ModuleDict()
        # 若未显式给出层号，则默认对所有 block 进行适配
        self.nullspace_layer: Set[int] = (
            set(nullspace_layer) if nullspace_layer is not None else
            set(range(len(vit_model.blocks)))
        )
        # 收集注意力 qkv 与 FFN 两个 Linear（只保留 weight）
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.nullspace_layer:
                continue
            # 注意力 qkv ： nn.Linear
            self.lora_modules[f"block_{idx}_attn_qkv"] = blk.attn.qkv
            # FFN
            self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2
        # ---------- 2️⃣ 保存原始 ViT ----------
        self.lora_vit = vit_model
        # ---------- 3️⃣ 冻结全部参数 ----------
        for n, p in self.lora_vit.named_parameters():
            p.requires_grad = False
        # ---------- 4️⃣ 只解冻目标权重 ----------
        # 4.1 注意力 & FFN 的 weight → trainable；bias → frozen
        for name, module in self.lora_modules.items():
            # weight
            if hasattr(module, "weight"):
                module.weight.requires_grad = True
            # bias (if exists) must stay frozen
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = False
        # 4.2 ViT 最后一个 LayerNorm 的 weight（bias 仍冻）
        # 多数 ViT 实现把最终 norm 存在 `vit_model.norm`
        for n, p in self.lora_vit.norm.named_parameters():
            if "weight" in n:
                p.requires_grad = False
            else:  # bias
                p.requires_grad = False
        # ---------- 5️⃣ 为每个可训练 weight 注册梯度投影 hook ----------
        self.use_projection = use_projection
        self._param_to_name: Dict[torch.nn.Parameter, str] = {}
        for name, module in self.lora_modules.items():
            if hasattr(module, "weight"):
                w = module.weight
                self._param_to_name[w] = name
                # 这里的 hook 会在 backward 时把梯度投影到 Null‑Space
                w.register_hook(self._make_grad_projection_hook(w))
        # 将最后的 norm.weight 也加入映射表，保持统一
        final_norm_weight = self.lora_vit.norm.weight
        self._param_to_name[final_norm_weight] = "final_norm_weight"
        # ----------
        self.projection_matrices: Dict[str, torch.Tensor] = {}

        print(self.summary())
    # --------------------------------------------------------------------- #
    # 梯度投影工具
    # --------------------------------------------------------------------- #
    def _make_grad_projection_hook(self, param: torch.nn.Parameter, weight: float = 1.0):
        """
        返回一个 `hook`，在反向传播得到 `grad` 后把它映射为
        ``weight * (grad @ P) + (1-weight) * grad``
        其中 ``P`` 是该参数对应的投影矩阵（若不存在则直接返回原梯度）。
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self.use_projection:
                return grad
            name = self._param_to_name.get(param, None)
            if name is None:
                return grad
            proj = self.projection_matrices.get(name, None)
            if proj is None:
                return grad
            # 保证设备/dtype 一致
            if proj.device != grad.device or proj.dtype != grad.dtype:
                proj = proj.to(device=grad.device, dtype=grad.dtype)
            # 这里采用加权混合的方式，保持数值稳定
            with torch.no_grad():
                new_grad = weight * torch.matmul(grad, proj) + (1.0 - weight) * grad
            return new_grad
        return hook
    # --------------------------------------------------------------------- #
    # 前向传播（直接走冻结的 ViT）
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接使用冻结的 ViT 进行前向计算。"""
        return self.lora_vit(x)
    # --------------------------------------------------------------------- #
    # 投影矩阵更新（依据协方差计算 null‑space）
    # --------------------------------------------------------------------- #
    def update_projection_matrices(
        self,
        covariances: Dict[str, torch.Tensor],
        eps: float = 0.05,
        soft: bool = True,
        temp: float = 5.0,
    ) -> None:
        """依据每个层的协方差矩阵重新构造投影矩阵。
        参数
        ----
        covariances: 形如 ``{layer_name: Σ}`` 的字典，Σ 为对应权重的协方差
        eps:          硬截断模式下保留的累计特征值比例阈值
        soft:         若为 ``True`` 使用软投影（指数衰减），否则使用硬截断
        temp:         软投影的温度系数（越大越“硬”）
        """
        if not self.use_projection:
            return
        self.projection_matrices = {}
        for name, module in self.lora_modules.items():
            if name not in covariances:
                continue
            cov = covariances[name]
            # 为数值稳定添加小的对角正则
            cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov)          # eigvals 已经从小到大
            eigvals = torch.abs(eigvals)                       # 防止负数（数值误差）
            if soft:
                # ------------------------------------------------- #
                # 软投影：   P = V @ diag(exp(-temp * λ̂)) @ V^T
                # ------------------------------------------------- #
                max_eig = eigvals.max()
                scaled = eigvals / (max_eig + 1e-12)           # 归一化到 (0,1]
                weights = torch.exp(-temp * scaled)           # 小特征值 → 大权重
                diag_w = torch.diag(weights)
                proj = eigvecs @ diag_w @ eigvecs.t()
                # 归一化使投影矩阵的范数约为 1，防止梯度尺度爆炸
                proj = proj / torch.norm(proj)
            else:
                # ------------------------------------------------- #
                # 硬截断：保留累计特征值比例 >= eps 的前 m 个特征向量
                # ------------------------------------------------- #
                total = eigvals.sum()
                cumsum = torch.cumsum(eigvals, dim=0)
                ratio = cumsum / total
                idx_candidates = (ratio >= eps).nonzero()
                if idx_candidates.numel() > 0:
                    m = idx_candidates[0].item()
                else:
                    m = eigvals.numel()
                # 取前 m 个特征向量构成 null‑space 基（列向量）
                null_basis = eigvecs[:, :m]                     # (d, m)
                proj = null_basis @ null_basis.t()              # (d, d)
            # 把投影矩阵移动到对应权重所在的设备 / dtype
            self.projection_matrices[name] = proj.to(module.weight.device)
    # --------------------------------------------------------------------- #
    # 开关 & 辅助函数
    # --------------------------------------------------------------------- #
    def enable_projection(self) -> None:
        """打开梯度投影开关（默认开启）。"""
        self.use_projection = True
    def disable_projection(self) -> None:
        """关闭梯度投影开关——此时梯度不会被投影。"""
        self.use_projection = False
    def get_module_names(self) -> List[str]:
        """返回所有被标记为 Null‑Space 适配的模块名称（不包括 final_norm）。"""
        return list(self.lora_modules.keys())
    # --------------------------------------------------------------------- #
    # 实用：查看可训练参数
    # --------------------------------------------------------------------- #
    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """返回当前模型中所有 `requires_grad=True` 的 Parameter。"""
        return [p for p in self.parameters() if p.requires_grad]
    def summary(self) -> str:
        """人类可读的简要信息，展示哪些层是可训练的。"""
        lines = [
            "=== NullSpaceViT Summary ===",
            f"Total trainable params : {sum(p.numel() for p in self.trainable_parameters()):_}",
            "Trainable modules:",
        ]
        for name, module in self.lora_modules.items():
            lines.append(f"  • {name}.weight   (shape={tuple(module.weight.shape)})")
        lines.append(f"  • final_norm_weight (shape={tuple(self.lora_vit.norm.weight.shape)})")
        lines.append(f"Projection enabled : {self.use_projection}")
        return "\n".join(lines)

class _LoRA_linear(nn.Module):
    """LoRA包装器用于单个线性层"""
    def __init__(self, linear, r):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scale = nn.Parameter(torch.tensor(0.8), requires_grad=True)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, std=0.02)

    def forward(self, x):
        orig_out = self.linear(x)
        lora_update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return orig_out + lora_update
    
class _LoRA_qkv_timm(nn.Module):
    """LoRA包装器用于注意力层的QKV矩阵"""
    def __init__(self, qkv, r):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.scale = nn.Parameter(torch.tensor(0.8), requires_grad=False)
        
        # 添加LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(r, self.dim), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(3 * self.dim, r))
    def forward(self, x):
        orig_qkv = self.qkv(x)
        lora_update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return orig_qkv + lora_update

class LoRAViT(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, lora_layer=None, use_projection: bool = False):
        super().__init__()
        assert r > 0
        self.lora_layer = lora_layer if lora_layer else list(range(len(vit_model.blocks)))
        
        for param in vit_model.parameters():
            param.requires_grad = False
        
        self.lora_modules = nn.ModuleDict()
        
        for idx, blk in enumerate(vit_model.blocks):
            if idx in self.lora_layer:
                layer_name = f"block_{idx}_attn_qkv"
                original_qkv = blk.attn.qkv
                new_qkv = _LoRA_qkv_timm(original_qkv, r)
                blk.attn.qkv = new_qkv
                self.lora_modules[layer_name] = new_qkv
                
                layer_name_fc1 = f"block_{idx}_mlp_fc1"
                layer_name_fc2 = f"block_{idx}_mlp_fc2"
                
                original_fc1 = blk.mlp.fc1
                new_fc1 = _LoRA_linear(original_fc1, r)
                blk.mlp.fc1 = new_fc1
                self.lora_modules[layer_name_fc1] = new_fc1
                
                original_fc2 = blk.mlp.fc2
                new_fc2 = _LoRA_linear(original_fc2, r)
                blk.mlp.fc2 = new_fc2
                self.lora_modules[layer_name_fc2] = new_fc2
        
        self.lora_vit = vit_model
        self.reset_parameters_svd()

        self.use_projection = use_projection
        self.projection_matrices = {}  # 存储每个LoRA模块A的投影矩阵，key为模块名
        
        # 为LoRA参数A创建梯度投影hook
        self._param_to_name = {}
        for name, module in self.lora_modules.items():
            # 只投影 A 的梯度（右乘）——已在原实现中
            self._param_to_name[module.lora_A] = (name, "A")
            module.lora_A.register_hook(self._make_grad_projection_hook(module.lora_A, name, side="right"))
    
    def reset_parameters_svd(self):
        """使用SVD分解初始化所有LoRA参数"""
        for name, module in self.lora_modules.items():
            if 'attn_qkv' in name:
                W = module.qkv.weight.data
                U, S, Vh = torch.linalg.svd(W)
                module.lora_A.data = Vh[:module.r, :].clone()
                module.lora_B.data.zero_()
            
            elif 'mlp_fc' in name:
                W = module.linear.weight.data
                U, S, Vh = torch.linalg.svd(W)
                module.lora_A.data = Vh[:module.r, :].clone()
                module.lora_B.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def _make_grad_projection_hook(self,
                                   param: torch.nn.Parameter,
                                   name: str,
                                   side: str = "right"):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self.use_projection:
                return grad
            # 取对应的投影矩阵
            proj = self.projection_matrices.get(name, None)
            if proj is None:
                return grad
            # 保证设备/dtype 一致
            if proj.device != grad.device or proj.dtype != grad.dtype:
                proj = proj.to(device=grad.device, dtype=grad.dtype)
            # 真正的投影
            with torch.no_grad():
                if side == "right":          # A： grad (r, in)  →  right‑multiply by P (in, in)
                    new_grad = torch.matmul(grad, proj)
                else:                        # B： grad (out, r) → left‑multiply by P (out, out)
                    new_grad = torch.matmul(proj, grad)
            return new_grad
        return hook
        
    def update_projection_matrices(
        self,
        covariances: Dict[str, torch.Tensor],
        eps: float = 0.05,
        soft: bool = True,
        temp: float = 0.5,
    ) -> None:
        if not self.use_projection:
            return
            
        self.projection_matrices = {}
        
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue

            # 为数值稳定添加小的对角正则
            cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvals 已经从小到大排好
            eigvals = torch.abs(eigvals)              # 防止数值误差导致的负特征值

            if soft:
                # ------------------------------------------------- #
                # 软投影：   P = V @ diag(exp(-temp * λ̂)) @ V^T
                # ------------------------------------------------- #
                # max_eig = eigvals.max()
                eig_sum = eigvals.sum()
                # 归一化特征值到 (0, 1]
                scaled_eigvals = eigvals / (eig_sum + 1e-12)
                # 小特征值获得大权重 (接近1)，大特征值获得小权重 (接近0)
                weights = torch.exp(-temp * scaled_eigvals)
                # 构建对角权重矩阵
                diag_w = torch.diag(weights)
                # 计算投影矩阵 P = V * D_w * V^T
                proj = eigvecs @ diag_w @ eigvecs.t()
                # 归一化投影矩阵以稳定梯度尺度
                proj = proj / torch.norm(proj)
            else:
                # ------------------------------------------------- #
                # 硬截断：保留累计特征值比例 >= eps 的前 m 个特征向量
                # ------------------------------------------------- #
                total = eigvals.sum()
                cumsum = torch.cumsum(eigvals, dim=0)
                ratio = cumsum / (total + 1e-12)
                
                # 找到第一个累计比例超过阈值 eps 的索引
                idx_candidates = (ratio >= eps).nonzero()
                if idx_candidates.numel() > 0:
                    m = idx_candidates[0].item()
                else:
                    # 如果所有特征值加起来都不够，则保留所有
                    m = eigvals.numel()
                
                null_space_basis = eigvecs[:, :m]  # (d, m)
                proj = null_space_basis @ null_space_basis.t() # (d, d)
            
            self.projection_matrices[name] = proj

    def enable_projection(self):
        self.use_projection = True
        
    def disable_projection(self):
        self.use_projection = False
                    

class FeatureCovarianceCalculator:
    def __init__(self, model, module_names, device='cuda'):
        self.model = model
        self.module_names = module_names
        self.device = device
        self.covariances = {name: None for name in module_names}
        self.counts = {name: 0 for name in module_names}
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """为指定模块注册前向钩子"""
        for name in self.module_names:
            try:
                module = self.model.lora_modules[name]
            except:
                module = self.model.lora_modules[name]
            if module is None:
                raise ValueError(f"模块 {name} 不存在于模型中")
            
            def hook_fn(module, input, output, name=name):
                self._update_covariance(name, input[0])
            
            hook = module.register_forward_hook(hook_fn)
            self.hooks.append(hook)
    
    def _update_covariance(self, name, features):
        """在线更新协方差矩阵"""
        # 特征形状: (batch_size, in_features)
        features = features.detach().to(self.device)
        B, N, D = features.size()
        features = features.view(B*N, D)
        
        # 非中心协方差: X^T X / n
        cov_batch = features.t() @ features  # (in_features, in_features)
        if self.covariances[name] is None:
            self.covariances[name] = cov_batch
        else:
            self.covariances[name] += cov_batch
        
        self.counts[name] += B*N
    
    def compute_final_covariances(self):
        """计算最终的协方差矩阵"""
        final_covs = {}
        for name in self.module_names:
            if self.counts[name] > 0:
                final_covs[name] = self.covariances[name] / self.counts[name]
            else:
                final_covs[name] = None
        return final_covs
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()

def build_projection(
    cov: torch.Tensor,
    eps: float = 0.10,
    soft: bool = True,
    temp: float = 20,
) -> torch.Tensor:
    """Identical projection builder function."""
    cov = cov + 1e-6 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.abs(eigvals)

    if soft:
        # 计算权重
        weights = torch.exp(-temp * eigvals)
        weights = weights / weights.sum()
        diag_w = torch.diag(weights)
        diag = diag_w.diag()
        diag_clamped = torch.clamp(diag, max=2.0)
        diag_w = torch.diag(diag_clamped)
        P = eigvecs @ diag_w @ eigvecs.t()
    else:
        total = eigvals.sum()
        cumsum = torch.cumsum(eigvals, dim=0)
        ratio = cumsum / (total + 1e-12)
        idx = (ratio >= eps).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        P = V_keep @ V_keep.t()
        I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
        P = eps * I + (1 - eps) * P
    return P
