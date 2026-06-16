# classifier/cosine_classifier_builder.py
import torch
import torch.nn as nn
import time
import logging
from typing import Dict
from classifier.base_classifier_builder import BaseClassifierBuilder
from classifier.cosine_classifier import CosineClassifier
from compensator.gaussian_statistics import GaussianStatistics

def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")

class CosineClassifierBuilder(BaseClassifierBuilder):
    """余弦分类器构建器"""
    def __init__(self, tau: float = 1.0, device: str = "cuda"):
        self.tau = tau
        self.device = device

    def build(self, stats_dict: Dict[int, GaussianStatistics]) -> nn.Module:
        """
        根据每个类别的高斯统计量构建余弦分类器。
        使用类别均值作为权重向量。
        """
        start_time = time.time()
        
        class_ids = sorted(stats_dict.keys())
        num_classes = len(class_ids)
        
        # 获取特征维度
        first_cid = class_ids[0]
        feature_dim = stats_dict[first_cid].mean.size(0)
        
        # 收集均值向量作为类别权重
        means = []
        for cid in class_ids:
            mu = stats_dict[cid].mean.float().to(self.device)
            means.append(mu)
        
        weights = torch.stack(means)  # [C, D]
        
        # 构建模型
        model = CosineClassifier(weights=weights, tau=self.tau, device=self.device)
        
        end_time = time.time()
        log_time_usage("Cosine Classifier build", start_time, end_time)
        
        return model.cpu()
