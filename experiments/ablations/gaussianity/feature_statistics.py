import os
import torch
import argparse
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_features(features_dir, output_name="feature_statistics"):
    """
    Load cached features, calculate skewness and kurtosis, and plot distributions.
    """
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('_features.pt')]
    if not feature_files:
        print(f"No feature files found in {features_dir}")
        return
    
    feature_files.sort(key=lambda x: int(x.split('_')[1]))
    
    results = []
    detailed_stats = []
    
    for f in feature_files:
        task_id = int(f.split('_')[1])
        data = torch.load(os.path.join(features_dir, f), map_location='cpu')
        
        train_features = data['train_features'].numpy()
        train_labels = data['train_labels'].numpy()
        
        unique_classes = np.unique(train_labels)
        task_skewness = []
        task_kurtosis = []
        
        for cls in unique_classes:
            cls_features = train_features[train_labels == cls]
            if len(cls_features) < 3: # Need at least a few samples for skew/kurt
                continue
                
            # Calculate skewness and kurtosis along feature dimension
            s = skew(cls_features, axis=0)
            k = kurtosis(cls_features, axis=0)
            
            # Filter out NaNs if any
            s = s[~np.isnan(s)]
            k = k[~np.isnan(k)]
            
            if len(s) == 0: continue
            
            mean_s = np.mean(s)
            mean_k = np.mean(k)
            
            task_skewness.append(mean_s)
            task_kurtosis.append(mean_k)
            
            detailed_stats.append({
                'task_id': task_id,
                'class_id': cls,
                'skewness': mean_s,
                'kurtosis': mean_k
            })
            
        if not task_skewness: continue
        
        avg_skew = np.mean(task_skewness)
        avg_kurt = np.mean(task_kurtosis)
        
        results.append({
            'task_id': task_id,
            'avg_skewness': avg_skew,
            'avg_kurtosis': avg_kurt
        })
        
        print(f"Task {task_id}: Avg Skew={avg_skew:.4f}, Avg Kurt={avg_kurt:.4f}")

    if not results:
        print("No results to plot.")
        return

    # Create DataFrame for better plotting
    df_detailed = pd.DataFrame(detailed_stats)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # 1. Skewness Violin Plot
    plt.subplot(1, 2, 1)
    sns.violinplot(data=df_detailed, x='task_id', y='skewness', inner="quart")
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Feature Skewness Distribution per Task')
    plt.xlabel('Task ID')
    plt.ylabel('Skewness (Ideal=0)')

    # 2. Kurtosis Violin Plot
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df_detailed, x='task_id', y='kurtosis', inner="quart")
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5) # Fisher definition (normal=0)
    plt.title('Feature Kurtosis Distribution per Task')
    plt.xlabel('Task ID')
    plt.ylabel('Kurtosis (Ideal=0)')

    plt.tight_layout()
    plot_path = os.path.join(features_dir, f'{output_name}.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    # Save CSV for table results
    csv_path = os.path.join(features_dir, f'{output_name}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved CSV stats to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, required=True, help='Directory containing cached features')
    parser.add_argument('--output_name', type=str, default="feature_statistics", help='Name of output files')
    args = parser.parse_args()
    analyze_features(args.features_dir, args.output_name)
