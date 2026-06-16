
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
from models.sgp_lora import build_projection

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
        use_projection: bool = True,
    ):
        super().__init__()
        # ---------- 1️⃣ 记录要适配的模块 ----------
        self.lora_modules = nn.ModuleDict()
        # 若未显式给出层号，则默认对所有 block 进行适配
        self.nullspace_layer: Set[int] = set(range(len(vit_model.blocks)))
        
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
    # 清除投影矩阵功能
    # --------------------------------------------------------------------- #
    def finalize_without_lora(self) -> None:
        """
        清除投影矩阵以释放内存（NullSpaceViT 不使用 LoRA）
        """
        self.eval()
        
        # 直接删除投影矩阵以释放内存
        if hasattr(self, 'projection_matrices'):
            # 清空投影矩阵字典
            self.projection_matrices.clear()
            # 可选：设置标志表示投影矩阵已被清除
            self.projection_cleared = True
            
        # 取消注册所有梯度投影钩子以防止访问已清除的投影矩阵
        if hasattr(self, '_param_to_name'):
            for param in self._param_to_name.keys():
                if hasattr(param, 'grad') and param.grad is not None:
                    # 需要记录原始钩子，但我们没有直接的方法取消它们
                    # 因此设置一个标志表明投影矩阵已被清除
                    pass
    
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
            # 检查投影矩阵是否已被清除
            if hasattr(self, 'projection_cleared') and self.projection_cleared:
                return grad
                
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
                new_grad = torch.matmul(grad, proj) + (1.0 - weight) * grad
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
        soft_projection: bool = True,
        weight_temp: float = 5.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.0,
    ) -> None:
        if not self.use_projection:
            return
        
        self.projection_matrices = {}
        for name, module in self.lora_modules.items():
            if name not in covariances:
                continue
            cov = covariances[name]
            
            # 使用与LoRA_SGP相同的build_projection函数
            proj = build_projection(
                cov,
                soft_projection=soft_projection,
                weight_temp=weight_temp,
                weight_kind=weight_kind,
                weight_p=weight_p,
                nsp_eps=nsp_eps,
                nsp_weight=nsp_weight
            )
            
            # 把投影矩阵移动到对应权重所在的设备 / dtype
            try:
                # 获取模块的设备
                device = None
                if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                    device = module.weight.device
                else:
                    # 如果模块没有weight属性，使用第一个参数的设备
                    for param in module.parameters():
                        device = param.device
                        break
                
                if device is not None:
                    # 确保设备类型正确
                    import torch
                    if isinstance(device, torch.device):
                        self.projection_matrices[name] = proj.to(device=device, dtype=proj.dtype)
                    else:
                        # 如果不是torch.device类型，尝试转换
                        self.projection_matrices[name] = proj.to(str(device), dtype=proj.dtype)
                else:
                    # 使用默认设备
                    self.projection_matrices[name] = proj
            except Exception as e:
                print(f"Warning: Could not determine device for module {name}, using default device. Error: {e}")
                self.projection_matrices[name] = proj
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
            if hasattr(module, 'weight') and module.weight is not None:
                lines.append(f"  • {name}.weight   (shape={tuple(module.weight.shape)})")
            else:
                lines.append(f"  • {name}.weight   (shape=unknown)")
        
        # 安全地获取norm权重形状
        if hasattr(self.lora_vit, 'norm') and self.lora_vit.norm is not None and hasattr(self.lora_vit.norm, 'weight') and self.lora_vit.norm.weight is not None:
            lines.append(f"  • final_norm_weight (shape={tuple(self.lora_vit.norm.weight.shape)})")
        else:
            lines.append(f"  • final_norm_weight (shape=unknown)")
            
        lines.append(f"Projection enabled : {self.use_projection}")
        return "\n".join(lines)
    
    def get_param_groups(self):
        """返回可训练参数列表，与PlainLoRAViT保持兼容"""
        params = []
        for name, module in self.lora_modules.items():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                params.append(module.weight)
        
        # 添加最终norm的权重（如果可训练）
        if (hasattr(self.lora_vit, 'norm') and
            hasattr(self.lora_vit.norm, 'weight') and
            self.lora_vit.norm.weight is not None and
            self.lora_vit.norm.weight.requires_grad):
            params.append(self.lora_vit.norm.weight)
            
        return params
    
    def merge_lora_weights(self):
        pass


class FullFineTuneViT(nn.Module):
    """
    Wrapper for a Vision‑Transformer that enables **full fine‑tuning**
    on the linear weights of the attention qkv projection and the two FFN
    linear layers (fc1, fc2) of the selected blocks, together with the final
    LayerNorm weight.
    Only these weights are trainable; every bias and every other parameter
    (including all other LayerNorms) stays frozen.
    This is similar to NullSpaceViT but without null‑space projection.
    """
    def __init__(
        self,
        vit_model: nn.Module,
        finetune_layer: Optional[List[int]] = None,
    ):
        super().__init__()
        # ---------- 1️⃣ 记录要适配的模块 ----------
        self.lora_modules = nn.ModuleDict()
        # 若未显式给出层号，则默认对所有 block 进行适配
        self.finetune_layer: Set[int] = (
            set(finetune_layer) if finetune_layer is not None else
            set(range(len(vit_model.blocks))))
        
        # 收集注意力 qkv 与 FFN 两个 Linear（只保留 weight）
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.finetune_layer:
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
                p.requires_grad = True  # 与NullSpaceViT不同，这里设为可训练
            else:  # bias
                p.requires_grad = False

        print(self.summary())
        
    # --------------------------------------------------------------------- #
    # 前向传播（直接走 ViT）
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接使用 ViT 进行前向计算。"""
        return self.lora_vit(x)
    
    # --------------------------------------------------------------------- #
    # 实用：查看可训练参数
    # --------------------------------------------------------------------- #
    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """返回当前模型中所有 `requires_grad=True` 的 Parameter。"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def summary(self) -> str:
        """人类可读的简要信息，展示哪些层是可训练的。"""
        lines = [
            "=== FullFineTuneViT Summary ===",
            f"Total trainable params : {sum(p.numel() for p in self.trainable_parameters()):_}",
            "Trainable modules:",
        ]
        for name, module in self.lora_modules.items():
            if hasattr(module, 'weight') and module.weight is not None:
                lines.append(f"  • {name}.weight   (shape={tuple(module.weight.shape)})")
            else:
                lines.append(f"  • {name}.weight   (shape=unknown)")
        
        # 安全地获取norm权重形状
        if hasattr(self.lora_vit, 'norm') and self.lora_vit.norm is not None and hasattr(self.lora_vit.norm, 'weight') and self.lora_vit.norm.weight is not None:
            lines.append(f"  • final_norm_weight (shape={tuple(self.lora_vit.norm.weight.shape)})")
        else:
            lines.append(f"  • final_norm_weight (shape=unknown)")
            
        return "\n".join(lines)
    
    def get_module_names(self) -> List[str]:
        """返回所有被标记为微调的模块名称（不包括 final_norm）。"""
        return list(self.lora_modules.keys())
    
    def get_param_groups(self):
        """返回可训练参数列表，与其他模型保持兼容"""
        params = []
        for name, module in self.lora_modules.items():
            if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                params.append(module.weight)
        
        # 添加最终norm的权重（如果可训练）
        if (hasattr(self.lora_vit, 'norm') and
            hasattr(self.lora_vit.norm, 'weight') and
            self.lora_vit.norm.weight is not None and
            self.lora_vit.norm.weight.requires_grad):
            params.append(self.lora_vit.norm.weight)
            
        return params
    
    def finalize_without_lora(self) -> None:
        """兼容性方法，不做任何操作"""
        self.eval()
    
    def merge_lora_weights(self):
        """兼容性方法，不做任何操作"""
        pass


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

def compute_covariances(model, data_loader, device='cuda'):
    module_names = model.get_module_names()
    
    cov_calculator = FeatureCovarianceCalculator(model, module_names, device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device)
            model(images)
    covariances = cov_calculator.compute_final_covariances()
    cov_calculator.remove_hooks()
    return covariances
