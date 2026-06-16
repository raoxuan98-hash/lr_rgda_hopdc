import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, HTML

def show_results_summary():
    """
    展示多架构对比结果的总结
    """
    result_dir = "实验结果保存/分类器消融实验/multi_architecture_comparison"
    
    # 读取性能汇总表
    import pandas as pd
    df = pd.read_csv(os.path.join(result_dir, "performance_summary.csv"))
    
    print("="*80)
    print("多架构分类器性能对比分析报告")
    print("="*80)
    
    # 按模型分组显示最佳性能
    print("\n1. 各模型最佳性能对比:")
    print("-" * 50)
    
    model_names = df['Model'].unique()
    for model in model_names:
        model_data = df[df['Model'] == model]
        best_row = model_data.loc[model_data['Best Accuracy (%)'].astype(float).idxmax()]
        print(f"{model:20} | {best_row['Classifier']:15} | {best_row['Best Accuracy (%)']:8}%")
    
    # 按分类器分组显示最佳性能
    print("\n2. 各分类器在不同模型上的性能对比:")
    print("-" * 50)
    
    classifier_names = df['Classifier'].unique()
    for classifier in classifier_names:
        classifier_data = df[df['Classifier'] == classifier]
        best_row = classifier_data.loc[classifier_data['Best Accuracy (%)'].astype(float).idxmax()]
        print(f"{classifier:15} | {best_row['Model']:20} | {best_row['Best Accuracy (%)']:8}%")
    
    # 分析最佳α1值
    print("\n3. 最佳α1值分析:")
    print("-" * 50)
    
    for classifier in ['QDA', 'SGD-linear', 'SGD-nonlinear']:
        classifier_data = df[(df['Classifier'] == classifier) & (df['Best α1'] != 'N/A')]
        if not classifier_data.empty:
            avg_alpha1 = classifier_data['Best α1'].astype(float).mean()
            print(f"{classifier:15} | 平均最佳α1: {avg_alpha1:.3f}")
    
    # 性能差异分析
    print("\n4. 模型间性能差异分析:")
    print("-" * 50)
    
    qda_performance = df[df['Classifier'] == 'QDA']['Best Accuracy (%)'].astype(float)
    sgd_performance = df[df['Classifier'] == 'SGD-linear']['Best Accuracy (%)'].astype(float)
    
    print(f"QDA最佳性能范围: {qda_performance.min():.2f}% - {qda_performance.max():.2f}%")
    print(f"SGD-linear最佳性能范围: {sgd_performance.min():.2f}% - {sgd_performance.max():.2f}%")
    
    # 模型排名
    print("\n5. 模型综合性能排名 (基于QDA最佳性能):")
    print("-" * 50)
    
    qda_df = df[df['Classifier'] == 'QDA'].copy()
    qda_df['Best Accuracy (%)'] = qda_df['Best Accuracy (%)'].astype(float)
    qda_df = qda_df.sort_values('Best Accuracy (%)', ascending=False)
    
    for idx, (_, row) in enumerate(qda_df.iterrows(), 1):
        print(f"{idx}. {row['Model']:20} | {row['Best Accuracy (%)']:8}%")
    
    print("\n" + "="*80)
    print("生成的文件:")
    print("-" * 50)
    print(f"1. 子图对比: {os.path.join(result_dir, 'multi_architecture_subplot_comparison.png')}")
    print(f"2. 综合对比图: {os.path.join(result_dir, 'multi_architecture_combined_comparison.png')}")
    print(f"3. 性能汇总表: {os.path.join(result_dir, 'performance_summary.csv')}")
    print("="*80)
    
    return df

def display_images():
    """
    显示生成的图像
    """
    result_dir = "实验结果保存/分类器消融实验/multi_architecture_comparison"
    
    # 加载图像
    subplot_img = mpimg.imread(os.path.join(result_dir, "multi_architecture_subplot_comparison.png"))
    combined_img = mpimg.imread(os.path.join(result_dir, "multi_architecture_combined_comparison.png"))
    
    # 显示图像
    fig, axes = plt.subplots(2, 1, figsize=(15, 20))
    
    axes[0].imshow(subplot_img)
    axes[0].set_title("多架构子图对比", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(combined_img)
    axes[1].set_title("多架构综合对比图", fontsize=16)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 显示结果总结
    df = show_results_summary()
    
    # 显示图像
    try:
        display_images()
    except Exception as e:
        print(f"无法显示图像: {e}")
        print("请手动查看生成的图像文件:")
        print("1. 实验结果保存/分类器消融实验/multi_architecture_comparison/multi_architecture_subplot_comparison.png")
        print("2. 实验结果保存/分类器消融实验/multi_architecture_comparison/multi_architecture_combined_comparison.png")