#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCM (Nearest Class Mean) Classifier
最近类均值分类器 - 基于余弦相似度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from compensator.gaussian_statistics import GaussianStatistics


class NCMClassifier(nn.Module):
    """
    最近类均值分类器 (Nearest Class Mean Classifier)
    基于余弦相似度而非欧氏距离
    
    原理：
    1. 计算每个类别的均值向量
    2. 对于测试样本，计算其与各个类别均值的余弦相似度
    3. 将样本分类到余弦相似度最高的类别
    
    判别函数：
      g_c(x) = cosine_similarity(x, μ_c) + log π_c
    其中 cosine_similarity = (x·μ_c) / (||x|| * ||μ_c||)
    """
    
    def __init__(
        self,
        stats_dict: Dict[int, GaussianStatistics],
        class_priors: Optional[Dict[int, float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化NCM分类器
        
        参数:
            stats_dict: 类别统计信息字典，{class_id: GaussianStatistics}
            class_priors: 类别先验概率，如果为None则使用均匀分布
            device: 设备类型
        """
        super().__init__()
        init_device = torch.device(device)
        self.register_buffer("_device_indicator", torch.empty(0, device=init_device), persistent=False)
        
        self.class_ids = sorted(stats_dict.keys())
        self.num_classes = len(self.class_ids)
        
        # 类先验
        if class_priors is None:
            priors_list = [1.0 / self.num_classes for _ in self.class_ids]
        else:
            priors_list = [class_priors[cid] for cid in self.class_ids]
        
        device_str = str(self._device_indicator.device)
        self.log_priors = nn.Parameter(
            torch.log(torch.tensor(priors_list, device=device_str)),
            requires_grad=False
        )  # [C]
        
        # 收集均值向量
        means = []
        for cid in self.class_ids:
            s = stats_dict[cid]
            means.append(s.mean.float().to(device_str))
        
        self.register_buffer("means", torch.stack(means), persistent=False)  # [C, D]
    
    @property
    def device(self):
        """获取设备"""
        return self._device_indicator.device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算每个类别的判别函数值（基于余弦相似度）
        
        参数:
            x: 输入特征，形状 [B, D]
            
        返回:
            logits: 判别函数值，形状 [B, C]
        """
        device_str = str(self._device_indicator.device)
        x = x.to(device_str)
        
        # 计算余弦相似度: similarity = (x · μ) / (||x|| * ||μ||)
        
        # 计算点积: [B, C]
        dot_product = torch.mm(x, self.means.T)
        
        # 计算x的模: [B, 1]
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        
        # 计算均值的模: [C]
        mu_norm = torch.norm(self.means, p=2, dim=1)
        
        # 计算余弦相似度: [B, C]
        # 添加小常数避免除零
        cosine_similarity = dot_product / (x_norm * mu_norm.unsqueeze(0) + 1e-8)
        
        # NCM的判别函数: g_c(x) = cosine_similarity + log π_c
        logits = cosine_similarity + self.log_priors.unsqueeze(0)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别标签
        
        参数:
            x: 输入特征，形状 [B, D]
            
        返回:
            predictions: 预测类别，形状 [B]
        """
        return torch.argmax(self.forward(x), dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率
        
        参数:
            x: 输入特征，形状 [B, D]
            
        返回:
            probabilities: 类别概率，形状 [B, C]
        """
        return F.softmax(self.forward(x), dim=1)