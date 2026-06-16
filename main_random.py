# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


import argparse
from trainer_random import train_random

def set_smart_defaults(ns):
    if not ns.smart_defaults:
        return ns
    if ns.dataset == 'cars196_224':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 2000
    elif ns.dataset == 'imagenet-r':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 2000
    elif ns.dataset == 'cifar100_224':
        ns.init_cls, ns.increment, ns.iterations = 10, 10, 2000
    elif ns.dataset == 'cub200_224':
        ns.init_cls, ns.increment, ns.iterations = 20, 20, 1000
    

    if ns.lora_type == 'full':
        ns.lrate = 1e-3
        ns.optimizer = 'sgd'
        ns.head_scale = 1.0

    if ns.test:
        ns.seed_list = [1993]
        ns.iterations = 100

    return ns


def main(args):
    results = train_random(args)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--dataset', type=str, default='cifar100_224', choices=['imagenet-r', 'cifar100_224', 'cub200_224', 'cars196_224', 'caltech101_224', 'oxfordpet37_224', 'food101_224', 'resisc45_224', 'cross_domain_elevater'], help='Dataset to use')
    basic.add_argument('--smart_defaults', action='store_true', default=False, help='If set, overwrite a few hyper‑parameters according to the dataset.')
    basic.add_argument('--user', type=str, default='2025-12-06-random_proj', choices=['authors'], help='User identifier (currently unused).')
    basic.add_argument('--test', action='store_true', default=False, help='If set, run a quick test with reduced settings.')
    basic.add_argument('--cross_domain', type=bool, default=False, help='Enable cross-domain class-incremental learning')
    '''order 1 simple datasets'''
    basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['cifar100_224', 'imagenet-r', 'cars196_224', 'cub200_224', 'caltech-101', 'oxford-flower-102', 'food-101'], help='List of datasets for cross-domain experiments')

    '''Test datasets'''
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['mnist', 'cifar100_224'], help='List of datasets for cross-domain experiments')
    basic.add_argument('--num_shots', type=int, default=64, help='Number of samples per class for few-shot learning. If > 0, randomly sample num_shots samples per class.')

    # 增量拆分参数
    inc = parser.add_argument_group('incremental', 'Incremental split settings for cross-domain datasets')
    inc.add_argument('--enable_incremental_split', type=bool, default=False,
                     help='Enable incremental split for cross-domain datasets. If set, each dataset will be split into multiple incremental subsets.')
    inc.add_argument('--num_incremental_splits', type=int, default=2,
                     help='Number of incremental splits per dataset when enable_incremental_split is True.')
    inc.add_argument('--incremental_split_seed', type=int, default=42,
                     help='Random seed for incremental split to ensure reproducibility.')

    cls = parser.add_argument_group('class', 'Class increment settings')
    cls.add_argument('--init_cls', type=int, default=10, help='Number of classes in the first task.')
    cls.add_argument('--increment', type=int, default=90, help='Number of new classes added per task.')

    model = parser.add_argument_group('model', 'Backbone & LoRA settings')
    model.add_argument('--model_name', type=str, default='random_projector', help='Model identifier.')
    model.add_argument('--vit_type', type=str, default='vit-b-p16', choices=['vit-b-p16', 'vit-b-p16-dino', 'vit-b-p16-mae', 'vit-b-p16-clip', 'vit-b-p16-mocov3'], help='ViT backbone variant.')
    model.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')

    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')  
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=1500, help='Training iterations per task.')
    train_grp.add_argument('--warmup_ratio', type=int, default=0.1, help='Warm‑up ratio for learning rate schedule.')
    train_grp.add_argument('--ca_epochs', type=int, default=5, help='Classifier alignment epochs.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=1e-4, help='Learning rate.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--evaluate_final_only', type=bool, default=False)
    train_grp.add_argument('--gamma_kd', type=float, default=0.0, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--update_teacher_each_task', type=bool, default=True, help='If set, update the teacher network after each task.')
    train_grp.add_argument('--use_aux_for_kd', action='store_true', default=False, help='If set, use auxiliary data for KD.')
    train_grp.add_argument('--kd_type', type=str, default='cos', help='KD type (feat / logit).')
    train_grp.add_argument('--distillation_transform', type=str, default='linear', help='Distillation head transform (identity / linear / weaknonlinear).')
    train_grp.add_argument('--eval_only', action='store_true', default=False)

    model.add_argument('--lora_rank', type=int, default=4, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="basic_lora", choices=['basic_lora'], help='Type of LoRA adaptor.')

    model.add_argument('--random_projection_dim', type=int, default=6000, help='Dimension of random projection when random_projection > 0.')
    model.add_argument('--first_section_adaptation', type=bool, default=True, help='Enable adaptation in the first section.')

    gda = parser.add_argument_group('gda', 'Gaussian discriminate analysis settings')
    gda.add_argument('--lda_reg_alpha',  type=float, default=0.10, help='LDA regularisation alpha.')
    gda.add_argument('--qda_reg_alpha1', type=float, default=0.20, help='QDA regularisation alpha 1.')
    gda.add_argument('--qda_reg_alpha2', type=float, default=2.00, help='QDA regularisation alpha 2.')
    gda.add_argument('--qda_reg_alpha3', type=float, default=0.50, help='QDA regularisation alpha 3.')
    return parser

# In[]
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args = set_smart_defaults(args)
    args = vars(args)
    main(args)