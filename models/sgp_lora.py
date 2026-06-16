import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple
from timm.models.vision_transformer import VisionTransformer as timm_ViT

class FixedProjection(nn.Module):
    def __init__(self, P: torch.Tensor):
        super().__init__()
        self.register_buffer("P", P)

    def forward(self) -> torch.Tensor:
        return self.P

# ==================== LoRA 基类 ====================
class SGPBaseLoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        proj: nn.Module):

        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj

        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)

        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_active:
            P_scaled = self.P()
            A_eff = self.A @ P_scaled
            lora_delta = self.B @ A_eff
            return F.linear(self.linear.weight + lora_delta, self.linear.bias)
        else:
           return self.linear(x)

    def merge_lora_weights(self, lora_active: bool=True) -> None:
        with torch.no_grad():
            P_scaled = self.P()
            delta = self.B @ self.A @ P_scaled
            self.linear.weight.data.add_(delta.to(self.linear.weight.device))
            self.B.data.zero_()
            self.lora_active = torch.tensor(lora_active)


class SGPBaseDoRA(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        proj: nn.Module):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.P = proj

        # LoRA 参数
        self.A = nn.Parameter(torch.zeros(r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # DoRA 参数：方向 + 幅度
        with torch.no_grad():
            weight = linear.weight.data
            weight_norm = weight.norm(p=2, dim=1, keepdim=True) + 1e-8
            self.weight_directions = nn.Parameter(
                weight / weight_norm, requires_grad=False)
            self.magnitude = nn.Parameter(weight_norm.clone(), requires_grad=True)

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_buffer("bias", None)
        self.register_buffer("lora_active", torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P_scaled = self.P()
        if self.lora_active:
            A_eff = self.A @ P_scaled
            lora_delta = self.B @ A_eff  # (out, in)
            adapted_weight = (self.weight_directions + lora_delta) * self.magnitude
        else:
            adapted_weight = self.weight_directions * self.magnitude
        return F.linear(x, adapted_weight, self.bias)

    def merge_lora_weights(self, lora_active: bool=True) -> None:
        with torch.no_grad():
            P_scaled = self.P()
            lora_delta = self.B @ self.A @ P_scaled
            self.weight_directions.data.add_(lora_delta)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B.data.zero_()
            self.lora_active = torch.tensor(lora_active)


class BaseLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        lora_class: type = SGPBaseDoRA,
        placeholder_proj_factory: Optional[callable] = None,
        include_norm: bool = False):
        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        try:
            self.feature_dim = vit_model.embed_dim
        except:
            self.feature_dim = 768

        self.use_projection = True
        self.lora_layer = (list(lora_layer) if lora_layer is not None else list(range(len(vit_model.blocks))))

        for n, p in vit_model.named_parameters():
            if include_norm and "norm" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad = False

        self.lora_modules = nn.ModuleDict()

        dev = vit_model.patch_embed.proj.weight.device
        def make_placeholder(d, dtype):
            if placeholder_proj_factory is not None:
                return placeholder_proj_factory(d, dev, dtype)
            else:
                return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

        for idx, blk in enumerate(vit_model.blocks):
            if idx not in self.lora_layer:
                continue

            dev = vit_model.patch_embed.proj.weight.device  # 或者保持在循环外（推荐）

            # ---- QKV ----
            qkv_in = blk.attn.qkv.in_features
            qkv_dtype = blk.attn.qkv.weight.dtype
            qkv_proj = make_placeholder(qkv_in, qkv_dtype)
            new_qkv = lora_class(blk.attn.qkv, r, qkv_proj)
            blk.attn.qkv = new_qkv
            self.lora_modules[f"block_{idx}_attn_qkv"] = new_qkv

            # ---- Attention Proj ----
            proj_in = blk.attn.proj.in_features
            proj_dtype = blk.attn.proj.weight.dtype
            proj_proj = make_placeholder(proj_in, proj_dtype)
            new_proj = lora_class(blk.attn.proj, r, proj_proj)
            blk.attn.proj = new_proj
            self.lora_modules[f"block_{idx}_attn_proj"] = new_proj

            # ---- MLP fc1 ----
            fc1_in = blk.mlp.fc1.in_features
            fc1_dtype = blk.mlp.fc1.weight.dtype
            fc1_proj = make_placeholder(fc1_in, fc1_dtype)
            new_fc1 = lora_class(blk.mlp.fc1, r, fc1_proj)
            blk.mlp.fc1 = new_fc1
            self.lora_modules[f"block_{idx}_mlp_fc1"] = new_fc1

            # ---- MLP fc2 ----
            fc2_in = blk.mlp.fc2.in_features
            fc2_dtype = blk.mlp.fc2.weight.dtype
            fc2_proj = make_placeholder(fc2_in, fc2_dtype)
            new_fc2 = lora_class(blk.mlp.fc2, r, fc2_proj)
            blk.mlp.fc2 = new_fc2
            self.lora_modules[f"block_{idx}_mlp_fc2"] = new_fc2

        self.lora_vit = vit_model
        # self.reset_parameters_svd()
        self.reset_parameter_standard()

    def reset_parameter_standard(self) -> None:
        for _, module in self.lora_modules.items():
            nn.init.kaiming_uniform_(module.A, a=math.sqrt(5))
            nn.init.zeros_(module.B)

    def reset_parameters_svd(self) -> None:
        for _, module in self.lora_modules.items():
            if isinstance(module, SGPBaseLoRA):
                W = module.linear.weight
            elif isinstance(module, SGPBaseDoRA):
                W = module.weight_directions

            _, _, Vh = torch.linalg.svd(W, full_matrices=False)
            module.A.data = Vh[: self.r, :].clone()
            module.B.data.zero_()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_vit(x)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def finalize_without_lora(self) -> None:
        self.eval()
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights(lora_active=False)
        
        # if hasattr(self, 'lora_modules'):
        #     for name, module in self.lora_modules.items():
        #         if hasattr(module, 'P'):
        #             del module.P
        #     self.projection_cleared = True

    def merge_lora_weights(self):
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()        

class SGPLoRAViT(BaseLoRAViT):

    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        use_soft_projection: bool = True,
        weight_temp: float = 1.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.0):

        super().__init__(
            vit_model, r, lora_layer,
            lora_class=SGPBaseDoRA,
            placeholder_proj_factory=lambda d, dev, dtype: FixedProjection(torch.eye(d, device=dev, dtype=dtype)))
        
        self.use_soft_projection = use_soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p

        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            P = build_projection(
                cov,
                soft_projection=self.use_soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight)

            self.lora_modules[name].P = FixedProjection(P)

    def get_param_groups(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params


    # 继承 Base 的 kl_regularization=0


# ------------------------------------------------------------------
#  Original SGP projection builder (kept unchanged)
# ------------------------------------------------------------------
    # ============ 新增：可切换权重函数 ============
def compute_weights(x: torch.Tensor, weight_kind="log1p", beta=1.0, weight_p=1.0, weight_alpha=0.5, weight_kappa=2.0) -> torch.Tensor:
    if weight_kind == "exp":
        # 原版：exp(-β x) → 指数尾部（最快）
        return torch.exp(-beta * x)

    elif weight_kind == "rational1":
        # 1 / (1 + β x) → 一次幂律尾部（~ 1/x）
        return 1.0 / (1.0 + beta * x)

    elif weight_kind == "rational2":
        # 1 / (1 + β x^2) → 二次幂律尾部（~ 1/x^2）
        return 1.0 / (1.0 + beta * (x ** 2))

    elif weight_kind == "sqrt_rational2":
        # 1 / sqrt(1 + β x^2) ：小 x 二次起步，尾部 ~ 1/x
        return 1.0 / torch.sqrt(1.0 + beta * (x ** 2))

    elif weight_kind == "log1p":
        return 1.0 / (1.0 + beta * torch.log1p(x**weight_p))

    elif weight_kind == "power_family":
        return (1.0 + beta * (x ** weight_p)) ** (-weight_alpha)

    elif weight_kind == "stretched_exp":
        return torch.exp(- (beta * x) ** weight_kappa)

    else:
        raise ValueError(
            f"Unknown weight_kind='{weight_kind}'. "
            f"Choose from ['exp','rational1','rational2','sqrt_rational2','log1p','power_family','stretched_exp']")


class SGPLoRACLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        clip_vision_model: nn.Module,  # 应为 CLIPVisionTransformer
        r: int,
        lora_layer: Optional[Iterable[int]] = None,
        use_soft_projection: bool = True,
        weight_temp: float = 1.0,
        weight_kind: str = "log1p",
        weight_p: float = 1.0,
        nsp_eps: float = 0.05,
        nsp_weight: float = 0.0,
        lora_class: type = SGPBaseDoRA,  # 默认 DoRA，可换 LoRA
        include_norm: bool = False):

        super().__init__()
        assert r > 0, "LoRA rank r must be positive"
        self.r = r
        self.feature_dim = clip_vision_model.embeddings.patch_embedding.out_channels  # 768

        self.use_soft_projection = use_soft_projection
        self.weight_temp = weight_temp
        self.weight_kind = weight_kind
        self.weight_p = weight_p
        self.nsp_eps = nsp_eps
        self.nsp_weight = nsp_weight

        # 冻结原始参数
        for n, p in clip_vision_model.named_parameters():
            if include_norm and ("norm" in n or "layernorm" in n.lower()):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        # 层索引
        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(clip_vision_model.encoder.layers)))

        # 存储 LoRA 模块
        self.lora_modules = nn.ModuleDict()

        # 设备和 dtype 推断
        dev = clip_vision_model.embeddings.patch_embedding.weight.device
        dtype = clip_vision_model.embeddings.patch_embedding.weight.dtype

        def make_placeholder(d):
            return FixedProjection(torch.eye(d, device=dev, dtype=dtype))

        # 遍历每一层 Transformer
        for idx, layer in enumerate(clip_vision_model.encoder.layers):
            if idx not in self.lora_layer:
                continue

            # === Self-Attention Projections ===
            for proj_name in ["k_proj", "v_proj", "q_proj", "out_proj"]:
                linear = getattr(layer.self_attn, proj_name)
                proj = make_placeholder(linear.in_features)
                lora_mod = lora_class(linear, r, proj)
                setattr(layer.self_attn, proj_name, lora_mod)
                self.lora_modules[f"layer_{idx}_attn_{proj_name}"] = lora_mod

            # === MLP ===
            for mlp_name in ["fc1", "fc2"]:
                linear = getattr(layer.mlp, mlp_name)
                proj = make_placeholder(linear.in_features)
                lora_mod = lora_class(linear, r, proj)
                setattr(layer.mlp, mlp_name, lora_mod)
                self.lora_modules[f"layer_{idx}_mlp_{mlp_name}"] = lora_mod

        self.clip_vision_model = clip_vision_model

    @torch.no_grad()
    def _ensure_merged_before_rebuild(self):
        self.merge_lora_weights()

    def update_projection_matrices(self, covariances: Dict[str, torch.Tensor]) -> None:
        self._ensure_merged_before_rebuild()
        for name, cov in covariances.items():
            if name not in self.lora_modules:
                continue
            P = build_projection(
                cov,
                soft_projection=self.use_soft_projection,
                weight_temp=self.weight_temp,
                weight_kind=self.weight_kind,
                weight_p=self.weight_p,
                nsp_eps=self.nsp_eps,
                nsp_weight=self.nsp_weight)
            self.lora_modules[name].P = FixedProjection(P)

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.clip_vision_model(pixel_values, kwargs)

    def get_module_names(self):
        return list(self.lora_modules.keys())

    def finalize_without_lora(self) -> None:
        self.eval()
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights(lora_active=False)
        
        # if hasattr(self, 'lora_modules'):
        #     for name, module in self.lora_modules.items():
        #         if hasattr(module, 'P'):
        #             del module.P
        #     self.projection_cleared = True


    def merge_lora_weights(self):
        for _, mod in self.lora_modules.items():
            mod.merge_lora_weights()

    def get_param_groups(self):
        scale_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "scale" in name.lower():
                scale_params.append(param)
            else:
                other_params.append(param)
        
        return {"scales": scale_params, "others": other_params}


def build_projection(
    cov: torch.Tensor,
    soft_projection: bool = True,
    weight_temp: float = 5.0,
    nsp_eps = 0.05, 
    nsp_weight = 0.0,
    *,
    # 新增：可切换的权重函数及其超参（全部可选）
    weight_kind: str = "log1p",
    weight_alpha: float = 0.5,
    weight_p: float = 2.0,
    weight_kappa: float = 2 ) -> torch.Tensor:
    """
    Construct the *soft* or *hard* projection matrix from a covariance.
    Kept backward-compatible by default (weight_kind='exp').
    """
    eps = 1e-6
    cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)          # ascending order
    eigvals = torch.abs(eigvals)
    d = cov.size(0)
    sum_vals = eigvals.sum()
    scale_ = d / (sum_vals + eps)
    eigvals = eigvals * scale_


    if soft_projection:
        weights = compute_weights(eigvals, weight_kind, weight_temp, weight_p, weight_alpha, weight_kappa)
        max_weight = weights.max()
        weights = weights / max_weight
        diag_w = torch.diag(weights)
        P = eigvecs @ diag_w @ eigvecs.t()
    else:
        eps_hard = nsp_eps
        total = eigvals.sum()
        cumsum = torch.cumsum(eigvals, dim=0)
        ratio = cumsum / (total + 1e-12)
        idx = (ratio >= eps_hard).nonzero(as_tuple=False)
        m = idx[0].item() if idx.numel() > 0 else eigvals.numel()
        V_keep = eigvecs[:, :m]
        P = V_keep @ V_keep.t()
        I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
        P = (1 - nsp_weight) * P + nsp_weight * I
    return P