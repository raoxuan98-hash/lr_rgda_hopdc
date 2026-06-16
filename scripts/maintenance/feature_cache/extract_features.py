# In[]

import os
import torch
import argparse
import logging
from utils.data_manager import WithinDomainDataManager, CrossDomainDataManagerCore
from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore
from models.subspace_lora import SubspaceLoRA
from main import build_parser, set_smart_defaults
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract_features(args):
    """
    Load checkpoints for each task and extract features for train and test sets.
    Save them to a .pt file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args['cross_domain']:
        data_manager = BalancedCrossDomainDataManagerCore(
            dataset_names=args['cross_domain_datasets'],
            balanced_datasets_root="balanced_datasets",
            seed=args['seed'],
            num_shots=args.get('num_shots', 0),
            use_balanced_datasets=True,
            enable_incremental_split=args.get('enable_incremental_split', False),
            num_incremental_splits=args.get('num_incremental_splits', 5),
            incremental_split_seed=args.get('incremental_split_seed', 42))
    else:
        data_manager = WithinDomainDataManager(
            dataset_name=args['dataset'],
            seed=args['seed'],
            init_cls=args['init_cls'],
            increment=args['increment'],
            args=args)

    model = SubspaceLoRA(args)
    model.network.eval()
    
    log_path = args.get('log_path', '.')
    save_dir = os.path.join(log_path, 'cached_features')
    os.makedirs(save_dir, exist_ok=True)
    
    nb_tasks = data_manager.nb_tasks
    
    for task_id in range(nb_tasks):
        checkpoint_path = os.path.join(log_path, f"after_task_{task_id + 1}.pth")
        if not os.path.exists(checkpoint_path):
            logging.warning(f"Checkpoint {checkpoint_path} not found. Skipping task {task_id}.")
            continue
            
        logging.info(f"Loading checkpoint {checkpoint_path}...")
        param_dict = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
        model.network.load_state_dict(param_dict, strict=False)
        model.network.to(device)
        model.network.eval()
        
        # We need to extract features for the cumulative training and testing data up to this task
        train_set = data_manager.get_incremental_subset(task=task_id, source="train", cumulative=True, mode="test")
        test_set = data_manager.get_incremental_subset(task=task_id, source="test", cumulative=True, mode="test")
        
        train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)
        
        def extract(loader):
            features = []
            labels = []
            with torch.no_grad():
                for inputs, targets, _ in tqdm(loader, desc="Extracting"):
                    inputs = inputs.to(device)
                    feats = model.network.extract_vector(inputs)
                    features.append(feats.cpu())
                    labels.append(targets.cpu())
            return torch.cat(features, dim=0), torch.cat(labels, dim=0)
            
        logging.info(f"Extracting train features for task {task_id}...")
        train_features, train_labels = extract(train_loader)
        
        logging.info(f"Extracting test features for task {task_id}...")
        test_features, test_labels = extract(test_loader)
        
        save_path = os.path.join(save_dir, f"task_{task_id}_features.pt")
        torch.save({
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels
        }, save_path)
        logging.info(f"Saved features to {save_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    parser.add_argument('--log_path', type=str, required=True, help='Path to the directory containing checkpoints')
    parser.add_argument('--seed', type=int, help='Seed used for data shuffling. If not provided, will try to use the first from seed_list.')
    args = parser.parse_args()
    args = set_smart_defaults(args)
    
    # Ensure seed is set for extract_features
    if args.seed is None:
        if args.seed_list:
            args.seed = args.seed_list[0]
            logging.info(f"Seed not provided, using first from seed_list: {args.seed}")
        else:
            args.seed = 1993 # Default fallback
            logging.info(f"Seed not provided and seed_list empty, using default: {args.seed}")
            
    extract_features(vars(args))
