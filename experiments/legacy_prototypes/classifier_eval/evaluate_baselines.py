import torch
import argparse
import os
from classifier.classifier_builder import ClassifierReconstructor
from classifier_ablation.experiments.exp1_performance_surface import build_gaussian_statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True, help='Path to cached_features/task_X_features.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Loading features from {args.features_path}...")
    data = torch.load(args.features_path)
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    test_labels = data['test_labels']
    
    print("Building Gaussian statistics...")
    train_stats = build_gaussian_statistics(train_features, train_labels)
    variants = {"baseline": train_stats}
    
    print("Building classifiers...")
    # Add our new classifiers: "ls" (Ridge Regression), "tsvd" (TSVD), "cosine" (Cosine)
    # Also evaluate LR-RGDA ("qda") and standard baselines for comparison
    classifier_types = ["qda", "ls", "tsvd", "cosine", "ncm", "sgd", "lda"]
    
    builder = ClassifierReconstructor(device=args.device)
    classifiers = builder.build_classifiers(variants, classifier_type=classifier_types)
    
    print("\n=== Evaluation Results ===")
    print(f"{'Classifier':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for cls_name, cls_model in classifiers.items():
        cls_model.eval()
        cls_model.to(args.device)
        test_f = test_features.to(args.device)
        test_l = test_labels.to(args.device)
        
        with torch.no_grad():
            logits = cls_model(test_f)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == test_l).float().mean().item()
            
        cls_type = cls_name.split(" + ")[1]
        print(f"{cls_type:<15} | {acc*100:6.2f}%")

if __name__ == '__main__':
    main()
