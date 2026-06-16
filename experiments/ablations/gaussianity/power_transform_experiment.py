import torch
import numpy as np
import argparse
from sklearn.preprocessing import PowerTransformer
from classifier.gaussian_classifier import LowRankGaussianDA

def compute_stats(features, labels):
    stats_dict = {}
    unique_classes = torch.unique(labels)
    for cls in unique_classes:
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        cls_features = features[cls_idx]
        mean = cls_features.mean(dim=0)
        
        # Center features
        centered = cls_features - mean
        n_samples = cls_features.shape[0]
        
        # Compute covariance
        cov = (centered.T @ centered) / (n_samples - 1) if n_samples > 1 else torch.eye(features.shape[1])
        
        stats_dict[cls.item()] = {
            'mean': mean,
            'cov': cov,
            'n_samples': n_samples
        }
    return stats_dict

def simulate_long_tail(features, labels, imb_factor=0.01):
    """
    Simulate long-tail distribution by randomly discarding samples from classes 
    to create an imbalanced dataset based on an exponential decay.
    """
    unique_classes = torch.unique(labels)
    num_classes = len(unique_classes)
    
    # Calculate number of samples for each class
    class_counts = [ (labels == cls).sum().item() for cls in unique_classes ]
    max_count = max(class_counts)
    
    # Create exponentially decaying sample counts
    new_counts = []
    for i in range(num_classes):
        new_count = int(max_count * (imb_factor ** (i / (num_classes - 1.0))))
        new_counts.append(max(1, new_count)) # at least 1 sample
        
    # Sample data
    new_features_list = []
    new_labels_list = []
    
    for i, cls in enumerate(unique_classes):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        # Randomly permute and select subset
        perm = torch.randperm(len(cls_idx))
        selected_idx = cls_idx[perm[:new_counts[i]]]
        
        new_features_list.append(features[selected_idx])
        new_labels_list.append(labels[selected_idx])
        
    return torch.cat(new_features_list, dim=0), torch.cat(new_labels_list, dim=0)

def simulate_nonlinear_projection(features, output_dim=None):
    """
    Simulate a high-dimensional non-linear projection using Random Projection + ReLU
    This explicitly breaks the Gaussian assumption.
    """
    if output_dim is None:
        # 降维处理，因为Yeo-Johnson对高维数据非常慢，默认取特征维度的一半，但至少为64
        output_dim = max(64, features.shape[1] // 2)
        
    # Generate random projection matrix
    torch.manual_seed(42) # For reproducibility
    projection_matrix = torch.randn(features.shape[1], output_dim) / np.sqrt(output_dim)
    
    # Apply projection and non-linearity (ReLU)
    projected = torch.matmul(features, projection_matrix)
    non_linear_features = torch.relu(projected)
    
    return non_linear_features

def test_power_transform(features_path, mode="original", output_csv=None):
    print(f"\n[{mode.upper()}] Loading features from {features_path}...")
    data = torch.load(features_path)
    
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    test_labels = data['test_labels']
    
    # 构造符合 LowRankGaussianDA 要求的统计量字典
    # LowRankGaussianDA 期望 stats_dict 的值具有 .mean 和 .cov 属性
    class StatsWrapper:
        def __init__(self, mean, cov):
            self.mean = mean
            self.cov = cov
            
    # Apply perturbations based on mode
    if mode == "long_tail":
        print("Simulating Long-Tail distribution on training data...")
        train_features, train_labels = simulate_long_tail(train_features, train_labels)
        # Test data remains balanced
        # Re-compute stats because training data changed
        wrapped_stats = {cid: StatsWrapper(s['mean'], s['cov']) for cid, s in compute_stats(train_features, train_labels).items()}
        priors = {cid: 1.0 / len(wrapped_stats) for cid in wrapped_stats}
    elif mode == "nonlinear":
        print("Simulating High-dimensional Non-linear Projection...")
        # Apply same projection to train and test
        output_dim = max(64, train_features.shape[1] // 2) # Reduce dimension for speed
        torch.manual_seed(42)
        proj_matrix = torch.randn(train_features.shape[1], output_dim) / np.sqrt(output_dim)
        
        train_features = torch.relu(torch.matmul(train_features, proj_matrix))
        test_features = torch.relu(torch.matmul(test_features, proj_matrix))
        # Re-compute stats because feature space changed
        wrapped_stats = {cid: StatsWrapper(s['mean'], s['cov']) for cid, s in compute_stats(train_features, train_labels).items()}
        priors = {cid: 1.0 / len(wrapped_stats) for cid in wrapped_stats}
        
    print("--- Testing Without Transform (Baseline LR-RGDA) ---")
    
    # 构造符合 LowRankGaussianDA 要求的统计量字典
    if mode == "original":
        wrapped_stats = {cid: StatsWrapper(s['mean'], s['cov']) for cid, s in compute_stats(train_features, train_labels).items()}
        priors = {cid: 1.0 / len(wrapped_stats) for cid in wrapped_stats}
        
    # 强制将所有数据转移到 CPU 进行后续处理
    train_features = train_features.cpu()
    train_labels = train_labels.cpu()
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    baseline_model = LowRankGaussianDA(
        stats_dict=wrapped_stats,
        class_priors=priors,
        rank=64,
        device="cpu" # 强制使用 CPU 避免 CUDA 初始化错误
    )
    
    with torch.no_grad():
        baseline_preds = baseline_model(test_features)
        baseline_acc = (baseline_preds.argmax(dim=1) == test_labels).float().mean().item()
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    print("\n--- Testing With Yeo-Johnson Transform ---")
    
    # Subsample for fitting if training data is too large to speed up Yeo-Johnson
    MAX_FIT_SAMPLES = 10000
    train_features_np = train_features.numpy()
    
    # Apply Yeo-Johnson transform
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    
    if train_features_np.shape[0] > MAX_FIT_SAMPLES:
        print(f"Subsampling {MAX_FIT_SAMPLES} out of {train_features_np.shape[0]} samples for fitting...")
        idx = np.random.choice(train_features_np.shape[0], MAX_FIT_SAMPLES, replace=False)
        fit_subset = train_features_np[idx]
        pt.fit(fit_subset)
        train_features_transformed = pt.transform(train_features_np)
    else:
        train_features_transformed = pt.fit_transform(train_features_np)
        
    train_features_transformed = torch.tensor(train_features_transformed, dtype=torch.float32)
    
    # Transform test features
    test_features_np = test_features.numpy()
    test_features_transformed = pt.transform(test_features_np)
    test_features_transformed = torch.tensor(test_features_transformed, dtype=torch.float32)
    
    transformed_stats = compute_stats(train_features_transformed, train_labels)
    wrapped_transformed_stats = {cid: StatsWrapper(s['mean'], s['cov']) for cid, s in transformed_stats.items()}
    
    transformed_model = LowRankGaussianDA(
        stats_dict=wrapped_transformed_stats,
        class_priors=priors,
        rank=64,
        device="cpu"
    )
    
    with torch.no_grad():
        transformed_preds = transformed_model(test_features_transformed)
        transformed_acc = (transformed_preds.argmax(dim=1) == test_labels).float().mean().item()
    print(f"Transformed Accuracy: {transformed_acc:.4f}")
    
    improvement = (transformed_acc - baseline_acc) * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    if output_csv:
        import csv
        import os
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Features Path', 'Mode', 'Baseline Acc', 'Transformed Acc', 'Improvement (%)'])
            writer.writerow([features_path, mode, f"{baseline_acc:.4f}", f"{transformed_acc:.4f}", f"{improvement:.2f}"])
            
    return baseline_acc, transformed_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True, help='Path to the .pt file containing features')
    parser.add_argument('--mode', type=str, default='all', choices=['original', 'long_tail', 'nonlinear', 'all'], help='Type of distribution to test')
    parser.add_argument('--output_csv', type=str, default='power_transform_results.csv', help='CSV file to append results to')
    args = parser.parse_args()
    
    if args.mode == 'all':
        for m in ['original', 'long_tail', 'nonlinear']:
            test_power_transform(args.features_path, mode=m, output_csv=args.output_csv)
    else:
        test_power_transform(args.features_path, mode=args.mode, output_csv=args.output_csv)
