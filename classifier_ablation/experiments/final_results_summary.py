import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path

def display_final_summary():
    """
    显示最终结果总结
    """
    print("="*100)
    print("多架构分类器性能对比实验 - 最终结果总结")
    print("="*100)
    
    # 读取全面性能汇总表
    summary_path = "实验结果保存/分类器消融实验/complete_multi_architecture_comparison/comprehensive_performance_summary.csv"
    df = pd.read_csv(summary_path)
    
    # 按模型分组显示最佳性能
    print("\n1. 各模型最佳性能对比:")
    print("-" * 60)
    
    model_names = df['Model'].unique()
    for model in model_names:
        model_data = df[df['Model'] == model]
        best_row = model_data.loc[model_data['Best Accuracy (%)'].astype(float).idxmax()]
        print(f"{model:20} | {best_row['Classifier']:15} | {best_row['Best Accuracy (%)']:8}%")
    
    # 按分类器分组显示最佳性能
    print("\n2. 各分类器在不同模型上的性能对比:")
    print("-" * 60)
    
    classifier_names = df['Classifier'].unique()
    for classifier in classifier_names:
        classifier_data = df[df['Classifier'] == classifier]
        best_row = classifier_data.loc[classifier_data['Best Accuracy (%)'].astype(float).idxmax()]
        print(f"{classifier:15} | {best_row['Model']:20} | {best_row['Best Accuracy (%)']:8}%")
    
    # 分析最佳α1值
    print("\n3. 最佳α1值分析:")
    print("-" * 60)
    
    for classifier in ['QDA', 'SGD-linear', 'SGD-nonlinear']:
        classifier_data = df[(df['Classifier'] == classifier) & (df['Best α1'] != 'N/A')]
        if not classifier_data.empty:
            avg_alpha1 = classifier_data['Best α1'].astype(float).mean()
            print(f"{classifier:15} | 平均最佳α1: {avg_alpha1:.3f}")
    
    # 性能差异分析
    print("\n4. 模型间性能差异分析:")
    print("-" * 60)
    
    qda_performance = df[df['Classifier'] == 'QDA']['Best Accuracy (%)'].astype(float)
    sgd_performance = df[df['Classifier'] == 'SGD-linear']['Best Accuracy (%)'].astype(float)
    
    print(f"QDA最佳性能范围: {qda_performance.min():.2f}% - {qda_performance.max():.2f}%")
    print(f"SGD-linear最佳性能范围: {sgd_performance.min():.2f}% - {sgd_performance.max():.2f}%")
    
    # 模型排名
    print("\n5. 模型综合性能排名 (基于QDA最佳性能):")
    print("-" * 60)
    
    qda_df = df[df['Classifier'] == 'QDA'].copy()
    qda_df['Best Accuracy (%)'] = qda_df['Best Accuracy (%)'].astype(float)
    qda_df = qda_df.sort_values('Best Accuracy (%)', ascending=False)
    
    for idx, (_, row) in enumerate(qda_df.iterrows(), 1):
        print(f"{idx}. {row['Model']:20} | {row['Best Accuracy (%)']:8}%")
    
    # 详细性能分析
    print("\n6. 详细性能分析:")
    print("-" * 60)
    
    for model in model_names:
        model_data = df[df['Model'] == model]
        print(f"\n{model}:")
        for _, row in model_data.iterrows():
            classifier = row['Classifier']
            best_acc = row['Best Accuracy (%)']
            mean_acc = row['Mean Accuracy (%)']
            std_acc = row['Std Accuracy (%)']
            best_alpha1 = row['Best α1']
            
            print(f"  {classifier:15} | 最佳: {best_acc:6}% | 平均: {mean_acc:6}% | 标准差: {std_acc:6}% | 最佳α1: {best_alpha1}")
    
    print("\n" + "="*100)
    print("生成的文件:")
    print("-" * 60)
    print("1. 四架构网格对比图:")
    print("   实验结果保存/分类器消融实验/complete_multi_architecture_comparison/four_architecture_grid_comparison.png")
    print("\n2. 四架构综合对比图:")
    print("   实验结果保存/分类器消融实验/complete_multi_architecture_comparison/four_architecture_combined_comparison.png")
    print("\n3. 全面性能汇总表:")
    print("   实验结果保存/分类器消融实验/complete_multi_architecture_comparison/comprehensive_performance_summary.csv")
    print("\n4. 三架构对比图:")
    print("   实验结果保存/分类器消融实验/multi_architecture_comparison/multi_architecture_subplot_comparison.png")
    print("\n5. 三架构综合对比图:")
    print("   实验结果保存/分类器消融实验/multi_architecture_comparison/multi_architecture_combined_comparison.png")
    print("\n6. 三架构性能汇总表:")
    print("   实验结果保存/分类器消融实验/multi_architecture_comparison/performance_summary.csv")
    print("="*100)
    
    return df

def display_generated_images():
    """
    显示生成的图像
    """
    # 四架构网格对比图
    grid_img_path = "实验结果保存/分类器消融实验/complete_multi_architecture_comparison/four_architecture_grid_comparison.png"
    # 四架构综合对比图
    combined_img_path = "实验结果保存/分类器消融实验/complete_multi_architecture_comparison/four_architecture_combined_comparison.png"
    
    try:
        # 加载图像
        if os.path.exists(grid_img_path):
            grid_img = mpimg.imread(grid_img_path)
        else:
            print(f"找不到文件: {grid_img_path}")
            return
            
        if os.path.exists(combined_img_path):
            combined_img = mpimg.imread(combined_img_path)
        else:
            print(f"找不到文件: {combined_img_path}")
            return
        
        # 显示图像
        fig, axes = plt.subplots(2, 1, figsize=(15, 20))
        
        axes[0].imshow(grid_img)
        axes[0].set_title("四架构网格对比图", fontsize=16)
        axes[0].axis('off')
        
        axes[1].imshow(combined_img)
        axes[1].set_title("四架构综合对比图", fontsize=16)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"无法显示图像: {e}")
        print("请手动查看生成的图像文件:")
        print("1. 实验结果保存/分类器消融实验/complete_multi_architecture_comparison/four_architecture_grid_comparison.png")
        print("2. 实验结果保存/分类器消融实验/complete_multi_architecture_comparison/four_architecture_combined_comparison.png")

if __name__ == '__main__':
    # 显示结果总结
    df = display_final_summary()
    
    # 显示图像
    try:
        display_generated_images()
    except Exception as e:
        print(f"无法显示图像: {e}")