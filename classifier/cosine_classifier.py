# classifier/cosine_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from compensator.gaussian_statistics import GaussianStatistics

class CosineClassifier(nn.Module):
    """
    余弦分类器 (Cosine Classifier)
    基于输入特征与类别权重之间的余弦相似度进行分类。
    
    判别函数：
      g_c(x) = tau * cosine_similarity(x, w_c)
    其中 cosine_similarity = (x·w_c) / (||x|| * ||w_c||)
    tau 是缩放因子（温度参数）。
    """
    
    def __init__(
        self,
        weights: torch.Tensor,
        tau: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化余弦分类器
        
        参数:
            weights: 类别权重矩阵，形状 [C, D]
            tau: 缩放因子
            device: 设备类型
        """
        super().__init__()
        init_device = torch.device(device)
        self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
        
        self.num_classes, self.feature_dim = weights.shape
        self.tau = tau
        
        # 归一化权重
        normalized_weights = F.normalize(weights, p=2, dim=1)
        self.register_buffer("weight", normalized_weights, persistent=False)  # [C, D]
    
    @property
    def device(self):
        """获取设备"""
        return self._device_indicator.device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算每个类别的判别函数值
        
        参数:
            x: 输入特征，形状 [B, D]
            
        返回:
            logits: 判别函数值，形状 [B, C]
        """
        device_str = str(self._device_indicator.device)
        x = x.to(device_str)
        
        # 归一化输入特征
        x_normalized = F.normalize(x, p=2, dim=1)
        
        # 计算余弦相似度并缩放
        # [B, D] @ [D, C] -> [B, C]
        logits = self.tau * torch.mm(x_normalized, self.weight.T)
        
        return logits
