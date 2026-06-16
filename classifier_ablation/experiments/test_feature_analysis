#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征奇异值分析代码的基本导入和函数
"""

import sys
import os

# 添加路径以便导入模块
sys.path.append('/home/raoxuan/projects/low_rank_rda')

def test_import():
    """测试导入功能"""
    try:
        print("测试导入...")
        
        # 测试基本库导入
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        import sklearn.decomposition
        print("✓ 基本库导入成功")
        
        # 测试特征分析代码导入
        from classifier_ablation.experiments.feature_singular_value_analysis import (
            compute_class_wise_singular_values,
            plot_singular_value_curves,
            analyze_singular_value_statistics,
            save_singular_values_data
        )
        print("✓ 特征分析模块导入成功")
        
        # 测试依赖模块导入
        from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics
        print("✓ 依赖模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def test_color_mapping():
    """测试颜色映射功能"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # 测试颜色映射
        n_classes = 10
        colors = cm.get_cmap('viridis')(np.linspace(0.2, 0.9, n_classes))
        print(f"✓ 颜色映射测试成功，生成了 {n_classes} 种颜色")
        print(f"  颜色形状: {colors.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 颜色映射测试失败: {e}")
        return False

def test_singular_value_computation():
    """测试奇异值计算功能"""
    try:
        import numpy as np
        import torch
        
        # 创建模拟数据
        n_samples = 50
        n_features = 20
        n_classes = 3
        
        # 生成模拟特征数据
        features = torch.randn(n_samples, n_features)
        labels = torch.randint(0, n_classes, (n_samples,))
        dataset_ids = torch.zeros(n_samples, dtype=torch.long)
        
        # 模拟一个类别的奇异值计算
        class_mask = (labels == 0)
        class_features = features[class_mask]
        
        if len(class_features) >= 2:
            # 转换为numpy并进行奇异值分解
            class_features_np = class_features.numpy()
            class_features_centered = class_features_np - np.mean(class_features_np, axis=0, keepdims=True)
            cov_matrix = np.cov(class_features_centered.T)
            U, s, Vt = np.linalg.svd(cov_matrix, full_matrices=False)
            
            print(f"✓ 奇异值计算测试成功")
            print(f"  协方差矩阵形状: {cov_matrix.shape}")
            print(f"  奇异值数量: {len(s)}")
            print(f"  最大奇异值: {np.max(s):.4f}")
            
            return True
        else:
            print("❌ 模拟数据不足")
            return False
            
    except Exception as e:
        print(f"❌ 奇异值计算测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("特征奇异值分析代码测试")
    print("="*60)
    
    tests = [
        ("导入测试", test_import),
        ("颜色映射测试", test_color_mapping),
        ("奇异值计算测试", test_singular_value_computation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n运行 {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"{test_name} 失败!")
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 代码应该可以正常工作。")
        print("\n使用示例:")
        print("python classifier_ablation/experiments/feature_singular_value_analysis.py \\")
        print("    --model vit-b-p16 \\")
        print("    --gpu 0 \\")
        print("    --iterations 0 \\")
        print("    --num_shots 128")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")
    
    print("="*60)

if __name__ == '__main__':
    main()