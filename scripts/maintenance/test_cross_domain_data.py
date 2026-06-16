import os
import sys
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append(os.getcwd())

from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore

def test_cross_domain_data_loading():
    # Setup logging to see CDM/BCDM output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Configuration mimicking the failing experiment
    args = {
        'cross_domain': True,
        'cross_domain_datasets': ['cifar100_224', 'imagenet-r', 'cars196_224', 'cub200_224', 'caltech-101', 'oxford-flower-102', 'food-101'],
        'enable_incremental_split': True,
        'num_incremental_splits': 2,
        'incremental_split_seed': 42,
        'num_shots': 64,
        'seed': 1993,
    }
    
    print("\n" + "="*80)
    print("Initializing BalancedCrossDomainDataManagerCore...")
    print(f"Datasets: {args['cross_domain_datasets']}")
    print(f"Incremental splits: {args['num_incremental_splits']}")
    print(f"Few-shot: {args['num_shots']} shots")
    print("="*80 + "\n")
    
    try:
        data_manager = BalancedCrossDomainDataManagerCore(
            dataset_names=args['cross_domain_datasets'],
            balanced_datasets_root="balanced_datasets",
            seed=args['seed'],
            num_shots=args['num_shots'],
            use_balanced_datasets=True,
            enable_incremental_split=args['enable_incremental_split'],
            num_incremental_splits=args['num_incremental_splits'],
            incremental_split_seed=args['incremental_split_seed']
        )
    except Exception as e:
        print(f"❌ Failed to initialize data manager: {e}")
        import traceback
        traceback.print_exc()
        return

    nb_tasks = data_manager.nb_tasks
    print(f"\n✅ Data manager initialized with {nb_tasks} tasks.")
    
    # Iterate through all tasks and verify subsets
    for task_id in range(nb_tasks):
        print(f"\n--- Checking Task {task_id} ---")
        
        # 1. Check Training Set (Non-cumulative)
        try:
            train_set = data_manager.get_subset(task_id, source="train", cumulative=False, mode="test")
            train_size = len(train_set)
            
            # Extract labels to check range
            train_labels = []
            loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=4)
            for batch in loader:
                train_labels.extend(batch[1].tolist())
            
            train_labels = np.array(train_labels)
            if len(train_labels) > 0:
                l_min, l_max = np.min(train_labels), np.max(train_labels)
                offset = data_manager.global_label_offset[task_id]
                num_cls = data_manager.get_task_size(task_id)
                expected_min = offset
                expected_max = offset + num_cls - 1
                
                status = "✅" if (l_min == expected_min and l_max == expected_max) else "❌"
                print(f"{status} Train Set: {train_size} samples, Label Range: [{l_min}, {l_max}] (Expected: [{expected_min}, {expected_max}])")
            else:
                print(f"❌ Train Set: EMPTY!")
                
        except Exception as e:
            print(f"❌ Error checking train set for task {task_id}: {e}")

        # 2. Check Test Set (Cumulative)
        try:
            test_set = data_manager.get_subset(task_id, source="test", cumulative=True, mode="test")
            test_size = len(test_set)
            
            test_labels = []
            loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
            for batch in loader:
                test_labels.extend(batch[1].tolist())
            
            test_labels = np.array(test_labels)
            if len(test_labels) > 0:
                l_min, l_max = np.min(test_labels), np.max(test_labels)
                expected_min = 0
                expected_max = data_manager.global_label_offset[task_id] + data_manager.get_task_size(task_id) - 1
                
                status = "✅" if (l_min == expected_min and l_max == expected_max) else "❌"
                print(f"{status} Cumulative Test Set: {test_size} samples, Label Range: [{l_min}, {l_max}] (Expected: [{expected_min}, {expected_max}])")
            else:
                print(f"❌ Cumulative Test Set: EMPTY!")
                
        except Exception as e:
            print(f"❌ Error checking test set for task {task_id}: {e}")

    print("\n" + "="*80)
    print("Test finished.")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_cross_domain_data_loading()
