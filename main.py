# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


import argparse
from trainer import train


DATASET_NUM_CLASSES = {
    'cifar100_224': 100,
    'cars196_224': 196,
    'cub200_224': 200,
    'imagenet-r': 200,
    'imagenet-a': 200,
    'vtab': 50,
    'caltech101_224': 102,
    'oxfordpet37_224': 37,
    'food101_224': 101,
    'resisc45_224': 45,
}


def apply_non_incremental_classifier_eval(ns):
    if not getattr(ns, 'non_incremental_classifier_eval', False):
        return ns

    ns.classifier_only_eval = True
    ns.evaluate_final_only = True
    ns.iterations = 0
    ns.gamma_kd = 0.0

    if ns.cross_domain:
        ns.enable_incremental_split = False
        return ns

    if ns.dataset not in DATASET_NUM_CLASSES:
        raise ValueError(
            f"Unknown class count for non-incremental evaluation dataset: {ns.dataset}")
    ns.init_cls = DATASET_NUM_CLASSES[ns.dataset]
    ns.increment = 0
    return ns


def set_smart_defaults(ns):
    # 如果是跨域实验，只处理 joint_full 方法的默认值
    if ns.cross_domain:
        # 为 cross-domain 实验设置 joint_full 方法的默认值
        if ns.lora_type == 'joint_full':
            ns.lrate = 1e-5
            ns.iterations = 6000
        return apply_non_incremental_classifier_eval(ns)
    
    # 只有在非跨域实验且启用了smart_defaults时，才设置默认值
    if not ns.smart_defaults:
        return apply_non_incremental_classifier_eval(ns)
    
    # 根据数据集设置基础参数
    if ns.dataset == 'cars196_224':
        ns.init_cls, ns.increment = 20, 20
        # 根据是否使用知识蒸馏设置不同的迭代次数
        if ns.gamma_kd > 0:
            ns.iterations = 1500
        else:
            ns.iterations = 1000
            
    elif ns.dataset in ['imagenet-r', 'imagenet-a']:
        ns.init_cls, ns.increment = 20, 20
        if ns.gamma_kd > 0:
            ns.iterations = 1500
        else:
            ns.iterations = 1000

    elif ns.dataset == 'vtab':
        ns.init_cls, ns.increment = 10, 10
        if ns.gamma_kd > 0:
            ns.iterations = 1500
        else:
            ns.iterations = 1000
            
    elif ns.dataset == 'cifar100_224':
        ns.init_cls, ns.increment = 10, 10
        if ns.gamma_kd > 0:
            ns.iterations = 1500 # 使用知识蒸馏时的迭代次数
        else:
            ns.iterations = 1000  # 无知识蒸馏时的迭代次数
            
    elif ns.dataset == 'cub200_224':
        ns.init_cls, ns.increment = 20, 20
        if ns.gamma_kd > 0:
            ns.iterations = 1000
        else:
            ns.iterations = 500
    
    if ns.lora_type == 'full':
        ns.lrate = 5e-6

    if ns.lora_type == 'full_nsp':
        ns.lrate = 5e-6
        if ns.gamma_kd > 0:
            ns.gamma_kd = 0.5

    if ns.test:
        ns.seed_list = [1993]
        ns.iterations = 100

    if ns.lora_type in ['joint_lora', 'joint_full']:
        if ns.lora_type == 'joint_full':
            ns.lrate = 1e-5
        
        if ns.dataset == 'cars196_224':
            ns.init_cls, ns.increment = 196, 0

        elif ns.dataset in ['imagenet-r', 'imagenet-a']:
            ns.init_cls, ns.increment = 200, 0

        elif ns.dataset == 'vtab':
            ns.init_cls, ns.increment = 50, 0

        elif ns.dataset == 'cifar100_224':
            ns.init_cls, ns.increment = 100, 0
            
        elif ns.dataset == 'cub200_224':
            ns.init_cls, ns.increment = 200, 0
        
        ns.iterations = 6000

    return apply_non_incremental_classifier_eval(ns)


def main(args):
    results = train(args)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'General / high‑level options')
    basic.add_argument('--dataset', type=str, default='cifar100_224', choices=['imagenet-r', 'imagenet-a', 'vtab', 'cifar100_224', 'cub200_224', 'cars196_224', 'caltech101_224', 'oxfordpet37_224', 'food101_224', 'resisc45_224', 'cross_domain_elevater'], help='Dataset to use')
    basic.add_argument('--smart_defaults', dest='smart_defaults', action='store_true', default=True, help='Use dataset-specific default hyper-parameters.')
    basic.add_argument('--no_smart_defaults', dest='smart_defaults', action='store_false', help='Keep explicitly provided hyper-parameters instead of applying dataset-specific defaults.')
    basic.add_argument('--user', type=str, default='2025-12-19-within', choices=['authors'], help='User identifier (currently unused).')
    basic.add_argument('--test', action='store_true', default=False, help='If set, run a quick test with reduced settings.')
    basic.add_argument('--cross_domain', action="store_true", default=False)
    '''order 1 datasets'''
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['cifar100_224', 'cub200_224', 'resisc45', 'caltech-101', 'dtd', 'imagenet-r', 'fgvc-aircraft-2013b-variants102', 'food-101', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'cars196_224'], help='List of datasets for cross-domain experiments')
    
    '''order 1 simple datasets'''
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['cifar100_224', 'imagenet-r'], help='List of datasets for cross-domain experiments')
    basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['cifar100_224', 'imagenet-r', 'cars196_224', 'cub200_224', 'caltech-101', 'oxford-flower-102', 'food-101'], help='List of datasets for cross-domain experiments')
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['imagenet-r', 'caltech-101', 'oxford-flower-102', 'cars196_224', 'cub200_224', 'food-101', 'cifar100_224'], help='List of datasets for cross-domain experiments')
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['cub200_224', 'cifar100_224', 'caltech-101', 'food-101', 'oxford-flower-102', 'cars196_224', 'imagenet-r'], help='List of datasets for cross-domain experiments')

    '''Test datasets'''
    # basic.add_argument('--cross_domain_datasets', type=str, nargs='+', default=['mnist', 'cifar100_224'], help='List of datasets for cross-domain experiments')
    basic.add_argument('--num_shots', type=int, default=64, help='Number of samples per class for few-shot learning. If > 0, randomly sample num_shots samples per class.')
    
    # 增量拆分参数
    inc = parser.add_argument_group('incremental', 'Incremental split settings for cross-domain datasets')
    inc.add_argument('--enable_incremental_split', action="store_true", default=True, help='Enable incremental split for cross-domain datasets. If set, each dataset will be split into multiple incremental subsets.')
    inc.add_argument('--num_incremental_splits', type=int, default=2, help='Number of incremental splits per dataset when enable_incremental_split is True.')
    inc.add_argument('--incremental_split_seed', type=int, default=42, help='Random seed for incremental split to ensure reproducibility.')

    cls = parser.add_argument_group('class', 'Class increment settings')
    cls.add_argument('--init_cls', type=int, default=20, help='Number of classes in the first task.')
    cls.add_argument('--increment', type=int, default=20, help='Number of new classes added per task.')

    model = parser.add_argument_group('model', 'Backbone & LoRA settings')
    model.add_argument('--model_name', type=str, default='RGDA', help='Model identifier.')
    model.add_argument('--vit_type', type=str, default='vit-b-p16', choices=['vit-b-p16', 'vit-b-p16-dino', 'vit-b-p16-mae', 'vit-b-p16-clip', 'vit-b-p16-mocov3'], help='ViT backbone variant.')
    model.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay.')

    train_grp = parser.add_argument_group('training', 'Optimisation & schedule')  
    train_grp.add_argument('--seed_list', nargs='+', type=int, default=[1993, 1996, 1997], help='Random seeds for multiple runs.')
    train_grp.add_argument('--iterations', type=int, default=10, help='Training iterations per task.')
    train_grp.add_argument('--warmup_ratio', type=float, default=0.10, help='Warm-up ratio for warmup_cosine learning rate schedule.')
    train_grp.add_argument('--lr_scheduler', type=str, default='warmup_cosine', choices=['warmup_cosine', 'onecycle'], help='Learning-rate scheduler. onecycle follows the official LADA-style per-batch OneCycleLR recipe.')
    train_grp.add_argument('--per_dataset_iterations', type=str, default='none', choices=['none', 'lada_16shot'], help='Override per-task iteration count. lada_16shot maps official LADA 16-shot dataset epochs to optimizer steps.')
    train_grp.add_argument('--ca_epochs', type=int, default=5, help='Classifier alignment epochs.')
    train_grp.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (adamw / sgd).')
    train_grp.add_argument('--lrate', type=float, default=1e-4, help='Learning rate.')
    train_grp.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    train_grp.add_argument('--evaluate_final_only', action="store_true", default=False)
    train_grp.add_argument('--gamma_kd', type=float, default=0.0, help='Knowledge‑distillation weight.')
    train_grp.add_argument('--update_teacher_each_task', action="store_true", default=True, help='If set, update the teacher network after each task.')
    train_grp.add_argument('--use_aux_for_kd', action='store_true', default=False, help='If set, use auxiliary data for KD.')
    train_grp.add_argument('--kd_type', type=str, default='feat', help='KD type (feat / cos).')
    train_grp.add_argument('--distillation_transform', type=str, default='identity', help='Distillation head transform (identity / linear / weaknonlinear).')
    train_grp.add_argument('--eval_only', action='store_true', default=False)
    train_grp.add_argument('--classifier_only_eval', action='store_true', default=False, help='Skip backbone training and drift compensation; collect fixed-backbone statistics and evaluate classifiers only.')
    train_grp.add_argument('--non_incremental_classifier_eval', action='store_true', default=False, help='Run one non-incremental classifier-only evaluation task over all classes; no backbone training and no drift compensation.')

    model.add_argument('--lora_rank', type=int, default=4, help='LoRA rank.')
    model.add_argument('--lora_type', type=str, default="full", choices=['basic_lora', 'sgp_lora', 'nsp_lora', 'full', 'full_nsp', 'joint_lora', 'joint_full'], help='Type of LoRA adaptor.')
    model.add_argument('--weight_temp', type=float, default=2.0, help='Projection temperature.')
    model.add_argument('--weight_kind', type=str, default='log1p', choices=["exp", "log1p", "rational1", "rational2", "sqrt_rational2", "power_family", "stretchqed_exp"])
    model.add_argument('--weight_p', type=float, default=1.0, help='Weight p.')
    model.add_argument('--nsp_eps', type=float, default=0.05, choices=[0.05, 0.10])
    model.add_argument('--nsp_weight', type=float, default=0.0, choices=[0.0, 0.02, 0.05])

    gda = parser.add_argument_group('gda', 'Gaussian discriminate analysis settings')
    gda.add_argument('--lda_reg_alpha', type=float, default=0.10, help='LDA regularisation alpha.')
    gda.add_argument('--rgda_alpha1', '--qda_reg_alpha1', dest='qda_reg_alpha1', type=float, default=0.20, help='RGDA class-specific covariance weight alpha1.')
    gda.add_argument('--rgda_alpha2', '--qda_reg_alpha2', dest='qda_reg_alpha2', type=float, default=2.00, help='RGDA shared covariance weight alpha2.')
    gda.add_argument('--rgda_alpha3', '--qda_reg_alpha3', dest='qda_reg_alpha3', type=float, default=0.50, help='RGDA identity regularization weight alpha3.')
    gda.add_argument('--rgda_rank', type=int, default=64, help='Low-rank dimension for LR-RGDA and LDA-TopK-LR-RGDA classifiers.')
    gda.add_argument('--rgda_num_centers', '--rgda_mc_num_centers', dest='rgda_mc_num_centers', type=int, default=4, help='Per-class centers for lr_rgda_mc. The old --rgda_num_centers spelling is kept as an alias.')
    gda.add_argument('--rgda_train_iter', '--rgda_mc_train_iter', dest='rgda_mc_train_iter', type=int, default=200, help='Affine-only fine-tuning iterations for lr_rgda_mc.')
    gda.add_argument('--rgda_fit_lr', '--rgda_mc_fit_lr', dest='rgda_mc_fit_lr', type=float, default=0.01, help='Learning rate for lr_rgda_mc affine-only fine-tuning.')
    gda.add_argument('--rgda_fit_samples_per_class', '--rgda_mc_fit_samples_per_class', dest='rgda_mc_fit_samples_per_class', type=int, default=16, help='GMM replay pseudo-samples per class for lr_rgda_mc.')
    gda.add_argument('--rgda_gmm_k', type=int, default=4, help='GMM components saved per class for lr_rgda_mc replay.')
    gda.add_argument('--rgda_gmm_backend', type=str, default='sklearn_spherical', choices=['sklearn_spherical', 'kmeans_diag'], help='GMM replay backend: sklearn_spherical follows project_clip main_joint.py; kmeans_diag is the faster deterministic fallback.')
    gda.add_argument('--rgda_gmm_sample_mode', type=str, default='mean', choices=['mean', 'sample'], help='GMM replay mode for lr_rgda_mc: component means or Gaussian samples.')
    gda.add_argument('--rgda_gmm_seed', type=int, default=42, help='Random seed for lr_rgda_mc GMM replay sampling.')
    gda.add_argument('--rgda_rerank_topk', type=int, default=50, help='LDA coarse top-k size for LDA-TopK-LR-RGDA reranking classifiers.')
    
    aux = parser.add_argument_group('auxiliary', 'External / auxiliary dataset')
    aux.add_argument('--auxiliary_data_path', type=str, default='/data1/open_datasets', help='Root path of the auxiliary dataset.')
    aux.add_argument('--aux_dataset', type=str, default='imagenet', help='Dataset type for auxiliary data (e.g. imagenet, cifar).', choices=['imagenet', 'flickr8k'])
    aux.add_argument('--auxiliary_data_size', type=int, default=2048, help='Number of samples drawn from the auxiliary dataset each epoch.')
    aux.add_argument('--feature_combination_type', type=str, default="combined", choices=['combined', 'aux_only', 'current_only'], help='Type of feature combination.')

    comp = parser.add_argument_group('compensator', 'Distribution compensator settings')
    comp.add_argument('--compensator_types', type=str, nargs='+', default=['SeqFT', 'SeqFT + HopDC'],
                    choices=['SeqFT', 'SeqFT + linear', 'SeqFT + weaknonlinear', 'SeqFT + HopDC', 'SeqFT + Hopfield', 'SeqFT + rff', 'SeqFT + RFF-HopDC', 'SeqFT + RFFHopDC', 'SeqFT + LinearHopDC'],
                    help='Distribution-statistics variants to evaluate. Use "SeqFT + HopDC" for the paper method.')
                    
    comp.add_argument('--hopfield_temp', type=float, default=0.05, help='Temperature parameter for Hopfield attention compensator.')
    comp.add_argument('--hopfield_topk', type=int, default=400, help='Top-k parameter for Hopfield attention compensator.')
    comp.add_argument('--rff_hopdc_dim', type=int, default=1024, help='Random feature dimension for RFF-HopDC linear attention.')
    comp.add_argument('--rff_hopdc_gamma', type=float, default=5.0, help='RBF gamma for RFF-HopDC. For normalized features, gamma ~= 1 / (2 * HopDC temperature).')
    comp.add_argument('--rff_hopdc_feature_mode', type=str, default='cos_positive', choices=['cos', 'cos_positive', 'elu'], help='Random feature map for RFF-HopDC.')
    comp.add_argument('--rff_hopdc_compensate_cov', action='store_true', default=False, help='If set, RFF-HopDC also estimates compensated covariance from samples.')
    comp.add_argument('--rff_hopdc_den_eps', type=float, default=1e-6, help='Denominator clamp for normalized RFF-HopDC attention.')
    comp.add_argument('--rff_hopdc_drift_clip', type=float, default=0.0, help='Optional max norm for predicted RFF-HopDC drift. 0 disables clipping.')
    comp.add_argument('--rff_hopdc_cov_samples', type=int, default=128, help='Samples per class when RFF-HopDC covariance compensation is enabled.')
    comp.add_argument('--rff_hopdc_seed', type=int, default=42, help='Random seed for RFF-HopDC random features.')
    
    # 权重插值参数
    interp = parser.add_argument_group('interpolation', 'Weight interpolation settings')
    interp.add_argument('--interpolation_alpha', type=float, default=0.8, help='Weight interpolation alpha value. Set to 1.0 to disable interpolation. Range [0.0, 1.0]. 0.0=use only prev_net, 1.0=use only current net, 0.5=equal weighting.')
    interp.add_argument('--enable_weight_interpolation', action="store_true", default=False, help='Enable weight interpolation with previous network.')
    
    # 分类器参数
    cls.add_argument('--classifier_types', type=str, nargs='+', default=['lr_rgda'], choices=['lr_rgda', 'lrrgda', 'low_rank_rgda', 'lr_rgda_mc', 'lrrgda_mc', 'multi_center_rgda', 'multicenter_rgda', 'lr_rgda_rerank', 'lrrgda_rerank', 'lda_lr_rgda_rerank', 'lr_rgda_mc_rerank', 'lrrgda_mc_rerank', 'lda_lr_rgda_mc_rerank', 'rgda', 'rgda_full', 'full_rgda', 'qda', 'lda', 'sgd', 'ls', 'tsvd', 'ncm', 'cosine'], help='Classifier reconstruction methods. lr_rgda is the single-center analytic baseline; lr_rgda_mc adds multi-centers and GMM replay fitting; *_rerank uses LDA top-k coarse ranking followed by LR-RGDA reranking.')
    
    return parser

# In[]
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args = set_smart_defaults(args)
    args = vars(args)
    main(args)
