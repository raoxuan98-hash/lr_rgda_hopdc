import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from models.sgp_lora import build_projection
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
        use_projection: bool = True,
    ):
        super().__init__()
        # ---------- 1️⃣ 记录要适配的模块 ----------
        self.lora_modules = nn.ModuleDict()
        # 若未显式给出层号，则默认对所有 block 进行适配
        self.nullspace_layers: Set[int] = set(range(len(vit_model.blocks)))
        
        # 收集注意力 qkv 与 FFN 两个 Linear（只保留 weight）
        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.nullspace_layers:
                continue

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
            module.weight.requires_grad = True


        # ---------- 5️⃣ 为每个可训练 weight 注册梯度投影 hook ----------
        self.use_projection = use_projection
        self._param_to_name: Dict[torch.nn.Parameter, str] = {}
        for name, module in self.lora_modules.items():
            w = module.weight
            self._param_to_name[w] = name
            # 这里的 hook 会在 backward 时把梯度投影到 Null‑Space
            w.register_hook(self._make_grad_projection_hook(w))

        self.projection_matrices: Dict[str, torch.Tensor] = {}

        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768

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
            self.projection_matrices.clear()
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
    def _make_grad_projection_hook(self, param: torch.nn.Parameter):
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
            
            if proj.device != grad.device or proj.dtype != grad.dtype:
                proj = proj.to(device=grad.device, dtype=grad.dtype)

            with torch.no_grad():
                new_grad = torch.matmul(grad, proj)
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
        soft_projection: bool = False,
        weight_temp: float = 5.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.02,
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
        return params
    def merge_lora_weights(self):
        pass

class FullFinetuneViT(nn.Module):
    """
    全参数微调ViT模型
    只微调MLP(FFN)模块，与现有LoRA变体保持接口一致性
    """
    def __init__(
        self,
        vit_model: timm_ViT,
        include_norm: bool = False,
        freeze_patch_embed: bool = True,
        finetune_layers: Optional[Iterable[int]] = None):
        
        super().__init__()
        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768
        
        self.lora_vit = vit_model
        # 默认对所有block进行微调
        self.finetune_layers = set(finetune_layers) if finetune_layers is not None else set(range(len(vit_model.blocks)))
        
        # 首先冻结所有参数
        for p in self.lora_vit.parameters():
            p.requires_grad = False
        
        # 可选：冻结patch embedding层（通常不需要微调）
        if freeze_patch_embed:
            for p in self.lora_vit.patch_embed.parameters():
                p.requires_grad = False
        
        # 解冻指定层的FFN模块
        for idx, blk in enumerate(self.lora_vit.blocks):
            if idx not in self.finetune_layers:
                continue
                
            # 解冻FFN模块
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'parameters'):
                for n, p in blk.mlp.named_parameters():
                    if "bias" not in n:
                        p.requires_grad = True
        
        # 可选：解冻norm层
        if include_norm:
            for name, p in self.lora_vit.named_parameters():
                if "norm" in name:
                    p.requires_grad = True
        
        # 存储可训练模块信息，用于接口一致性
        self.lora_modules = nn.ModuleDict()
        for idx, blk in enumerate(self.lora_vit.blocks):
            if idx not in self.finetune_layers:
                continue
                
            # FFN模块
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'fc1'):
                self.lora_modules[f"block_{idx}_mlp_fc1"] = blk.mlp.fc1
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'fc2'):
                self.lora_modules[f"block_{idx}_mlp_fc2"] = blk.mlp.fc2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.lora_vit(x)
    
    def get_param_groups(self):
        """
        返回所有可训练参数，与现有LoRA接口保持一致
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_module_names(self):
        """
        返回模块名称列表，与现有LoRA接口保持一致
        """
        return list(self.lora_modules.keys())
    
    def finalize_without_lora(self) -> None:
        """
        与现有LoRA接口保持一致，但在全参数微调中不执行任何操作
        """
        pass
    
    def merge_lora_weights(self):
        """
        与现有LoRA接口保持一致，但在全参数微调中不执行任何操作
        """
        pass
    
    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        """
        与现有LoRA接口保持一致，但在全参数微调中不执行任何操作
        """
        pass
    
    @property
    def use_projection(self):
        """
        与现有LoRA接口保持一致，全参数微调不使用投影
        """
        return False
    
    def count_trainable_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """统计总参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def print_parameter_statistics(self) -> None:
        """打印参数统计信息"""
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()
        
        print(f"=== MLP微调模型统计 ===")
        print(f"总模型参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 计算参数效率
        efficiency = (trainable_params / total_params) * 100
        print(f"参数效率: {efficiency:.2f}%")