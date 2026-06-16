import os
import sys
import logging
import torch
import random
import numpy as np
from collections.abc import Mapping, Sequence
# from models.subspace_lora import SubspaceLoRA
from models.random_projector import RandomProjector
from utils.data_manager import WithinDomainDataManager, CrossDomainDataManagerCore
from utils.balanced_cross_domain_data_manager import BalancedCrossDomainDataManagerCore
from utils.toolkit import count_parameters
import re

def train_random(args):
    all_results = {}
    
    for run_id, seed in enumerate(args['seed_list']):
        args['seed'], args['run_id'] = seed, run_id
        logfile_head, logfile_name = build_log_dirs(args)
        args['log_path'] = logfile_name
        
        # Configure logging with unbuffered file handler for real-time updates
        log_file_path = os.path.join(logfile_name, 'record.log')
        
        # 清除现有的日志处理器，避免冲突
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(filename=log_file_path, mode='a', encoding='utf-8')
        file_handler.stream.reconfigure(line_buffering=True)  # Enable line buffering
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s [%(filename)s] => %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 配置根日志记录器
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 打印日志文件路径，方便用户查找
        print(f"📁 日志文件路径: {log_file_path}")
        print(f"💡 提示: 使用 'tail -f {log_file_path}' 实时查看日志")
        print("-" * 80)
        
        args['log_path'] = logfile_name
        results = train_single_run_random(args)
        all_results[f"seed_{seed}"] = results
    
    # 在所有种子运行完成后，进行统计分析
    if len(all_results) > 1:  # 只有多于一个种子时才进行统计分析
        dataset_names = args.get('cross_domain_datasets', None)
        analyze_all_results(all_results, dataset_names, save_json=True)
    

def train_single_run_random(args, return_model: bool = False):
    # Setting random seed and device for reproducibility
    set_random(args['seed'])
    print_args(args)
    
    # Initialize data manager and model

    if args['cross_domain']:
        # 使用平衡后的cross-domain数据集
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
    
    model = RandomProjector(args)
    logging.info(f'All params: {count_parameters(model.network)}')
    logging.info(f'Trainable params: {count_parameters(model.network, True)}')
    final_results = model.loop(data_manager)
    
    # 添加log_path到结果中，以便aggregate_seed_results可以找到它
    if 'log_path' in args:
        final_results['log_path'] = args['log_path']
    
    if return_model:
        return final_results, model
    return final_results

def set_device(device_type):
    """Properly set the device (either CPU or GPU) based on input"""
    if isinstance(device_type, (list, tuple)):
        return [torch.device(f'cuda:{d}' if d != -1 else 'cpu') for d in device_type]
    return torch.device('cuda' if device_type != -1 else 'cpu')

def set_random(seed):
    """Set random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    """Log the arguments for this run"""
    for key, value in args.items():
        logging.info(f'{key}: {value}')


import os
from pathlib import Path
import json

def _filter_args_by_lora_type(args: dict) -> dict:
    """
    过滤参数字典，只保留与当前LoRA类型相关的参数
    这样可以避免在params.json中保存不相关的参数，导致日志命名混乱
    """
    lora_type = args.get('lora_type', 'basic_lora')
    filtered_args = args.copy()
    
    # 定义每种LoRA类型相关的参数
    sgp_lora_params = {'weight_temp', 'weight_kind', 'weight_p'}
    nsp_lora_params = {'nsp_eps', 'nsp_weight'}
    # Hopfield参数适用于所有LoRA类型
    hopfield_params = {'hopfield_temp', 'hopfield_topk'}
    
    # 移除与当前LoRA类型不相关的参数
    if lora_type == 'sgp_lora':
        # 保留SGP参数和Hopfield参数，移除NSP参数
        for param in nsp_lora_params:
            filtered_args.pop(param, None)
    elif lora_type == 'nsp_lora':
        # 保留NSP参数和Hopfield参数，移除SGP参数
        for param in sgp_lora_params:
            filtered_args.pop(param, None)
    elif lora_type == 'basic_lora':
        # 只保留Hopfield参数，移除其他LoRA特定参数
        for param in sgp_lora_params.union(nsp_lora_params):
            filtered_args.pop(param, None)
    elif lora_type == 'full':
        # 只保留Hopfield参数，移除其他LoRA特定参数
        for param in sgp_lora_params.union(nsp_lora_params):
            filtered_args.pop(param, None)
    
    return filtered_args

def build_log_dirs(args: dict, root_dir="."):
    """
    根据 args 构建多级日志目录，确保不同 LoRA 类型的参数正确分离
    
    目录结构改进：
    - 顶层目录现在包含实验类型标识（cross_domain 或 within_domain）
    - cross_domain实验：random_projector_logs_{user}_cross_domain/{datasets}_{vit_type}/...
    - within_domain实验：random_projector_logs_{user}_within_domain/{dataset}_{vit_type}/...
    
    这样可以明确区分cross-domain和within-domain的实验结果，避免混淆
    """

    def sanitize_filename(s: str) -> str:
        """移除或替换文件名中的非法字符"""
        # Windows 非法字符: \ / : * ? " < > |
        s = re.sub(r'[\\/:*?"<>|]', '_', str(s))
        # 可选：压缩连续下划线
        s = re.sub(r'_+', '_', s)
        return s.strip('_')

    def short(s: str, maxlen=40):
        """截断过长字符串，不加 hash，仅保留可读性"""
        s = sanitize_filename(str(s))
        if len(s) <= maxlen:
            return s
        return s[:maxlen].rstrip('_')  # 避免截断在下划线处

    def _get_lora_specific_params(lora_type: str, args: dict) -> list:
        """获取特定 LoRA 类型的参数，避免交叉污染"""
        params = []
        
        if lora_type == 'sgp_lora':
            # SGP LoRA 特有参数
            if 'weight_temp' in args:
                params.append(f"t-{short(args['weight_temp'])}")
            if 'weight_kind' in args:
                params.append(f"k-{short(args['weight_kind'])}")
            # 始终包含 weight_p 参数，即使是默认值，以确保不同参数组合的实验结果被正确区分
            if 'weight_p' in args:
                params.append(f"p-{short(args['weight_p'])}")
                
        elif lora_type == 'nsp_lora':
            # NSP LoRA 特有参数
            if 'nsp_eps' in args:
                params.append(f"eps-{short(args['nsp_eps'])}")
            if 'nsp_weight' in args:
                params.append(f"w-{short(args['nsp_weight'])}")
                
        elif lora_type == 'basic_lora':
            # Basic LoRA 通常没有额外参数，但可以在这里添加
            pass
            
        elif lora_type == 'full':
            # Full fine-tuning 可能有的参数
            pass
            
        # 添加Hopfield参数，适用于所有LoRA类型
        if 'hopfield_temp' in args:
            params.append(f"htemp-{short(args['hopfield_temp'])}")
        if 'hopfield_topk' in args:
            params.append(f"hk-{short(args['hopfield_topk'])}")
            
        return params

    def _get_kd_params(args: dict) -> list:
        """获取知识蒸馏相关参数，统一命名规则"""
        kd_params = []
        
        if args.get('gamma_kd', 0.0) > 0.0:
            kd_params.append(f"kd-{short(args['gamma_kd'])}")
            if 'kd_type' in args:
                kd_params.append(f"type-{short(args['kd_type'])}")
            if 'distillation_transform' in args:
                kd_params.append(f"dt-{short(args['distillation_transform'])}")
            if args.get('use_aux_for_kd', False):
                kd_params.append("aux")
            # 添加update_teacher_each_task参数，简写为utt
            if 'update_teacher_each_task' in args:
                kd_params.append(f"utt-{short(args['update_teacher_each_task'])}")
                
        return kd_params

    def _get_auxiliary_params(args: dict) -> list:
        """获取辅助数据相关参数，包括feature_combination_type和auxiliary_data_size"""
        aux_params = []
        
        # 添加feature_combination_type参数到日志命名（更短的形式）
        if 'feature_combination_type' in args:
            fc_type = args['feature_combination_type']
            if fc_type == 'combined':
                aux_params.append("fc-combined")
            elif fc_type == 'aux_only':
                aux_params.append("fc-aux")
            elif fc_type == 'current_only':
                aux_params.append("fc-cur")
        
        # 添加auxiliary_data_size参数到日志命名（仅当不等于默认值2048时）
        aux_size = args.get('auxiliary_data_size', 2048)
        if aux_size != 2048:  # 只在非默认情况下显示
            aux_params.append(f"aux-{short(aux_size)}")
            
        return aux_params

    def _get_incremental_split_params(args: dict) -> list:
        """获取增量拆分相关参数，包括enable_incremental_split和num_incremental_splits"""
        inc_params = []
        
        # 添加enable_incremental_split参数（使用简写形式）
        if args.get('enable_incremental_split', False):
            inc_params.append("inc-enabled")
            # 始终显示num_incremental_splits参数，方便区分不同的分割配置
            num_splits = args.get('num_incremental_splits', 2)
            inc_params.append(f"s-{short(num_splits)}")
        else:
            inc_params.append("inc-disabled")
            
        return inc_params

    def _get_random_projector_params(args: dict) -> list:
        """获取RandomProjector特定参数"""
        rp_params = []
        
        # 添加random_projection参数
        if 'random_projection' in args and args['random_projection'] > 0:
            rp_params.append(f"rp-{short(args['random_projection'])}")
            if 'random_projection_dim' in args:
                rp_params.append(f"rpd-{short(args['random_projection_dim'])}")
        
        # 添加use_projection参数
        if 'use_projection' in args and args['use_projection']:
            rp_params.append("proj")
        
        # 添加first_section_adaptation参数
        if 'first_section_adaptation' in args and args['first_section_adaptation']:
            rp_params.append("fsa")
            
        return rp_params

    def _validate_parameters(args: dict) -> None:
        """验证参数组合的合理性"""
        lora_type = args.get('lora_type', 'basic_lora')
        
        # 检查 LoRA 特定参数是否被误用
        if lora_type != 'sgp_lora':
            sgp_params = ['weight_temp', 'weight_kind', 'weight_p']
            for param in sgp_params:
                if param in args and args[param] is not None:
                    logging.warning(f"⚠️ Parameter '{param}' is being used with lora_type='{lora_type}', but it's specific to sgp_lora")
        
        if lora_type != 'nsp_lora':
            nsp_params = ['nsp_eps', 'nsp_weight']
            for param in nsp_params:
                if param in args and args[param] is not None:
                    logging.warning(f"⚠️ Parameter '{param}' is being used with lora_type='{lora_type}', but it's specific to nsp_lora")

    # 参数验证
    _validate_parameters(args)

    # 确定实验类型：cross-domain 或 within-domain
    is_cross_domain = args.get('cross_domain', False)
    
    # 顶层：模型、用户信息和实验类型（明确区分cross-domain和within-domain）
    experiment_type = "cross_domain" if is_cross_domain else "within_domain"
    base_dir = os.path.join(
        root_dir,
        f"{short(args['model_name'])}_logs_{short(args['user'])}_{experiment_type}"
    )

    # 根据实验类型构建不同的二级目录结构（进一步优化路径长度）
    if is_cross_domain:
        # 跨域实验：使用order标识而不是具体的数据集列表
        if 'cross_domain_datasets' in args:
            # 使用简化的标识符
            task_dir = os.path.join(
                base_dir,
                f"order1_{short(args['vit_type'])}",
                f"shots-{short(args.get('num_shots', 0))}",
                f"lrank-{short(args.get('lora_rank', 4))}_ltype-{short(args.get('lora_type', 'basic_lora'))}"
            )
        else:
            # 如果没有指定跨域数据集，使用默认标识
            task_dir = os.path.join(
                base_dir,
                f"unknown_{short(args['vit_type'])}",
                f"shots-{short(args.get('num_shots', 0))}",
                f"lrank-{short(args.get('lora_rank', 4))}_ltype-{short(args.get('lora_type', 'basic_lora'))}"
            )
    else:
        # 域内实验：使用传统的init_cls和increment参数
        task_dir = os.path.join(
            base_dir,
            f"{short(args['dataset'])}_{short(args['vit_type'])}",
            f"init-{short(args['init_cls'])}_inc-{short(args['increment'])}",
            f"lrank-{short(args.get('lora_rank', 4))}_ltype-{short(args.get('lora_type', 'basic_lora'))}"
        )

    # 三级：LoRA 特定参数（只包含相关的）
    lora_params = _get_lora_specific_params(args.get('lora_type', 'basic_lora'), args)
    
    # 四级：知识蒸馏参数（如果有）
    kd_params = _get_kd_params(args)
    
    # 五级：辅助数据参数（如果有）
    aux_params = _get_auxiliary_params(args)
    
    # 六级：增量拆分参数（如果有）
    inc_split_params = _get_incremental_split_params(args)
    
    # 七级：RandomProjector特定参数
    rp_params = _get_random_projector_params(args)
    
    # 合并所有参数（简化参数名称）
    method_params = lora_params + kd_params + aux_params + inc_split_params + rp_params
    
    # 构建方法参数目录（减少最大长度以避免过长路径）
    if method_params:
        method_subdir = "_".join(method_params)
        method_dir = os.path.join(task_dir, short(method_subdir, maxlen=60))
    else:
        method_dir = task_dir

    # 八级：优化器和训练参数（不包含种子，种子将在子目录中处理）
    opt_params = [
        f"opt-{args['optimizer']}",
        f"lr-{short(args['lrate'])}",
        f"b-{args['batch_size']}",
        f"i-{args['iterations']}"
    ]
    opt_str = "_".join(opt_params)
    opt_dir = os.path.join(method_dir, short(opt_str, maxlen=60))

    # === 逐级创建目录 ===
    abs_log_dir = os.path.abspath(opt_dir)
    current = Path(abs_log_dir).root
    for part in Path(abs_log_dir).parts[1:]:
        current = Path(current) / part
        current.mkdir(exist_ok=True)

    # 保存过滤后的参数到 JSON，避免参数交叉污染
    filtered_args = _filter_args_by_lora_type(args)
    params_json = Path(abs_log_dir) / "params.json"
    if not params_json.exists():
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(filtered_args, f, ensure_ascii=False, indent=2)

    # 为每个种子创建子目录
    seed_dir = os.path.join(abs_log_dir, f"seed_{args['seed']}")
    os.makedirs(seed_dir, exist_ok=True)

    # 注意：这里不能使用 logging.info，因为日志还没有配置
    # 目录信息会在日志配置完成后通过 print_args 函数记录

    return os.path.dirname(abs_log_dir), str(seed_dir)

def analyze_all_results(all_results: dict, dataset_names: list = [], save_json: bool = True, output_path: str = "") -> dict:
    """
    分析all_results中多个随机种子的结果，计算平均值和标准差并记录到日志
    适配新的final_results格式
    
    Args:
        all_results: 包含多个随机种子结果的字典
        dataset_names: 数据集名称列表，用于日志输出
        save_json: 是否将统计结果保存为JSON文件
        output_path: JSON文件保存路径，如果为None则自动生成
    
    Returns:
        dict: 包含统计结果的字典
    """
    import numpy as np
    import json
    from pathlib import Path
    
    if not all_results:
        logging.warning("📊 all_results为空，无法进行统计分析")
        return {}
    
    # 获取所有种子和变体名称
    seed_keys = list(all_results.keys())
    if len(seed_keys) == 0:
        logging.warning("📊 没有找到任何种子结果")
        return {}
    
    # 从第一个种子结果中获取变体名称
    first_seed_result = all_results[seed_keys[0]]
    variant_names = set()
    
    # 从last_task_accuracies获取变体名称
    if 'last_task_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['last_task_accuracies'].keys())
    
    # 从average_accuracies获取变体名称
    if 'average_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['average_accuracies'].keys())
    
    # 从cumulative_task_wise_accuracies获取变体名称
    if 'cumulative_task_wise_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['cumulative_task_wise_accuracies'].keys())
    
    # 从cumulative_class_wise_accuracies获取变体名称
    if 'cumulative_class_wise_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['cumulative_class_wise_accuracies'].keys())
    
    # 从task_wise_accuracies获取变体名称
    if 'task_wise_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['task_wise_accuracies'].keys())
    
    # 从class_wise_accuracies获取变体名称
    if 'class_wise_accuracies' in first_seed_result:
        variant_names.update(first_seed_result['class_wise_accuracies'].keys())
    
    variant_names = sorted(list(variant_names))
    
    if not variant_names:
        logging.warning("📊 没有找到任何变体名称")
        return {}
    
    # 获取任务ID列表（从last_task_id推断）
    task_ids = []
    if 'last_task_id' in first_seed_result:
        last_task_id = first_seed_result['last_task_id']
        task_ids = list(range(1, last_task_id + 1))  # 假设任务ID从1开始
    
    num_seeds = len(seed_keys)
    logging.info(f"📊 开始分析 {num_seeds} 个随机种子的实验结果")
    logging.info(f"📊 发现 {len(variant_names)} 个变体: {', '.join(variant_names)}")
    if task_ids:
        logging.info(f"📊 发现 {len(task_ids)} 个任务: {', '.join(map(str, task_ids))}")
    
    # 初始化统计结果字典
    statistics_results = {
        "summary": {
            "num_seeds": num_seeds,
            "num_variants": len(variant_names),
            "num_tasks": len(task_ids),
            "variant_names": variant_names,
            "task_ids": task_ids,
            "dataset_names": dataset_names
        },
        "variants": {}
    }
    
    # 记录统计结果
    logging.info("=" * 80)
    logging.info("📈 多种子统计分析结果")
    logging.info("=" * 80)
    
    for variant in variant_names:
        logging.info(f"\n🔍 变体: {variant}")
        logging.info("-" * 60)
        
        # 初始化变体统计结果
        variant_stats = {
            "data_wise_accuracy": {},
            "cumulative_average_accuracy": {},
            "cumulative_task_wise_accuracy": {},
            "cumulative_class_wise_accuracy": {},
            "task_wise_accuracy": {},
            "class_wise_accuracy": {}
        }
        
        # 1. 收集数据集级别准确率数据（data-wise）
        data_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'last_task_accuracies' in seed_result and variant in seed_result['last_task_accuracies']:
                data_wise_accs.append(seed_result['last_task_accuracies'][variant])
        
        if data_wise_accs:
            mean_data_wise = np.mean(data_wise_accs)
            std_data_wise = np.std(data_wise_accs)
            variant_stats["data_wise_accuracy"] = {
                "mean": float(round(mean_data_wise, 2)),
                "std": float(round(std_data_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in data_wise_accs]
            }
            logging.info(f"  数据集级别准确率: {mean_data_wise:.2f}% ± {std_data_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in data_wise_accs])}")
        else:
            variant_stats["data_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  数据集级别准确率: 无数据")
        
        # 2. 收集累积平均准确率数据
        cumulative_avg_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'average_accuracies' in seed_result and variant in seed_result['average_accuracies']:
                cumulative_avg_accs.append(seed_result['average_accuracies'][variant])
        
        if cumulative_avg_accs:
            mean_cumulative_avg = np.mean(cumulative_avg_accs)
            std_cumulative_avg = np.std(cumulative_avg_accs)
            variant_stats["cumulative_average_accuracy"] = {
                "mean": float(round(mean_cumulative_avg, 2)),
                "std": float(round(std_cumulative_avg, 2)),
                "raw_values": [float(round(acc, 2)) for acc in cumulative_avg_accs]
            }
            logging.info(f"  累积平均准确率: {mean_cumulative_avg:.2f}% ± {std_cumulative_avg:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in cumulative_avg_accs])}")
        else:
            variant_stats["cumulative_average_accuracy"] = {"error": "无数据"}
            logging.info(f"  累积平均准确率: 无数据")
        
        # 2.5. 收集任务级别累积平均准确率数据
        cumulative_task_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'cumulative_task_wise_accuracies' in seed_result and variant in seed_result['cumulative_task_wise_accuracies']:
                cumulative_task_wise_accs.append(seed_result['cumulative_task_wise_accuracies'][variant])
        
        if cumulative_task_wise_accs:
            mean_cumulative_task_wise = np.mean(cumulative_task_wise_accs)
            std_cumulative_task_wise = np.std(cumulative_task_wise_accs)
            variant_stats["cumulative_task_wise_accuracy"] = {
                "mean": float(round(mean_cumulative_task_wise, 2)),
                "std": float(round(std_cumulative_task_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in cumulative_task_wise_accs]
            }
            logging.info(f"  任务级别累积平均准确率: {mean_cumulative_task_wise:.2f}% ± {std_cumulative_task_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in cumulative_task_wise_accs])}")
        else:
            variant_stats["cumulative_task_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  任务级别累积平均准确率: 无数据")
        
        # 2.6. 收集类别级别累积平均准确率数据
        cumulative_class_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'cumulative_class_wise_accuracies' in seed_result and variant in seed_result['cumulative_class_wise_accuracies']:
                cumulative_class_wise_accs.append(seed_result['cumulative_class_wise_accuracies'][variant])
        
        if cumulative_class_wise_accs:
            mean_cumulative_class_wise = np.mean(cumulative_class_wise_accs)
            std_cumulative_class_wise = np.std(cumulative_class_wise_accs)
            variant_stats["cumulative_class_wise_accuracy"] = {
                "mean": float(round(mean_cumulative_class_wise, 2)),
                "std": float(round(std_cumulative_class_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in cumulative_class_wise_accs]
            }
            logging.info(f"  类别级别累积平均准确率: {mean_cumulative_class_wise:.2f}% ± {std_cumulative_class_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in cumulative_class_wise_accs])}")
        else:
            variant_stats["cumulative_class_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  类别级别累积平均准确率: 无数据")
        
        # 3. 收集任务级别平均准确率数据
        task_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'task_wise_accuracies' in seed_result and variant in seed_result['task_wise_accuracies']:
                task_wise_accs.append(seed_result['task_wise_accuracies'][variant])
        
        if task_wise_accs:
            mean_task_wise = np.mean(task_wise_accs)
            std_task_wise = np.std(task_wise_accs)
            variant_stats["task_wise_accuracy"] = {
                "mean": float(round(mean_task_wise, 2)),
                "std": float(round(std_task_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in task_wise_accs]
            }
            logging.info(f"  任务级别平均准确率: {mean_task_wise:.2f}% ± {std_task_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in task_wise_accs])}")
        else:
            variant_stats["task_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  任务级别平均准确率: 无数据")
        
        # 4. 收集类别级别平均准确率数据
        class_wise_accs = []
        for seed_key in seed_keys:
            seed_result = all_results[seed_key]
            if 'class_wise_accuracies' in seed_result and variant in seed_result['class_wise_accuracies']:
                class_wise_accs.append(seed_result['class_wise_accuracies'][variant])
        
        if class_wise_accs:
            mean_class_wise = np.mean(class_wise_accs)
            std_class_wise = np.std(class_wise_accs)
            variant_stats["class_wise_accuracy"] = {
                "mean": float(round(mean_class_wise, 2)),
                "std": float(round(std_class_wise, 2)),
                "raw_values": [float(round(acc, 2)) for acc in class_wise_accs]
            }
            logging.info(f"  类别级别平均准确率: {mean_class_wise:.2f}% ± {std_class_wise:.2f}%")
            logging.info(f"    详细数据: {', '.join([f'{acc:.2f}%' for acc in class_wise_accs])}")
        else:
            variant_stats["class_wise_accuracy"] = {"error": "无数据"}
            logging.info(f"  类别级别平均准确率: 无数据")
        
        # 5. 如果有per-task详细信息，也显示（从task_wise_accuracies中提取）
        # 注意：这里需要检查第一个种子的数据结构
        first_seed_result = all_results[seed_keys[0]]
        if ('task_wise_accuracies' in first_seed_result and 
            variant in first_seed_result['task_wise_accuracies'] and
            hasattr(first_seed_result['task_wise_accuracies'][variant], 'keys')):
            
            logging.info(f"  各任务详细准确率:")
            task_wise_data = first_seed_result['task_wise_accuracies'][variant]
            if isinstance(task_wise_data, dict):
                for task_id, acc in task_wise_data.items():
                    # 修复任务名称显示逻辑，适配增量拆分
                    if dataset_names and int(task_id) < len(dataset_names):
                        dataset_name = dataset_names[int(task_id)]
                        # 对于增量拆分场景，进一步清理数据集名称
                        if dataset_name.endswith('_split_0') or dataset_name.endswith('_split_1'):
                            dataset_name = dataset_name.split('_split_')[0]
                    else:
                        dataset_name = f"Task {task_id}"
                    logging.info(f"    {dataset_name}: {acc:.2f}%")
        
        statistics_results["variants"][variant] = variant_stats
    
    # 性能排名总结
    logging.info("\n" + "=" * 80)
    logging.info("🏆 性能排名总结")
    logging.info("=" * 80)
    
    # 按数据集级别准确率排名
    ranked_by_data_wise = []
    for variant in variant_names:
        if variant_stats := statistics_results["variants"][variant]:
            if "data_wise_accuracy" in variant_stats and "mean" in variant_stats["data_wise_accuracy"]:
                ranked_by_data_wise.append((variant, variant_stats["data_wise_accuracy"]["mean"]))
    
    if ranked_by_data_wise:
        ranked_by_data_wise.sort(key=lambda x: x[1], reverse=True)
        logging.info("📈 按数据集级别准确率排名:")
        for i, (variant, acc) in enumerate(ranked_by_data_wise, 1):
            logging.info(f"  {i:2d}. {variant:<30}: {acc:.2f}%")
    
    # 按累积平均准确率排名
    ranked_by_cumulative_avg = []
    for variant in variant_names:
        if variant_stats := statistics_results["variants"][variant]:
            if "cumulative_average_accuracy" in variant_stats and "mean" in variant_stats["cumulative_average_accuracy"]:
                ranked_by_cumulative_avg.append((variant, variant_stats["cumulative_average_accuracy"]["mean"]))
    
    if ranked_by_cumulative_avg:
        ranked_by_cumulative_avg.sort(key=lambda x: x[1], reverse=True)
        logging.info("\n📊 按累积平均准确率排名:")
        for i, (variant, acc) in enumerate(ranked_by_cumulative_avg, 1):
            logging.info(f"  {i:2d}. {variant:<30}: {acc:.2f}%")
    
    # 按任务级别累积平均准确率排名
    ranked_by_cumulative_task_wise = []
    for variant in variant_names:
        if variant_stats := statistics_results["variants"][variant]:
            if "cumulative_task_wise_accuracy" in variant_stats and "mean" in variant_stats["cumulative_task_wise_accuracy"]:
                ranked_by_cumulative_task_wise.append((variant, variant_stats["cumulative_task_wise_accuracy"]["mean"]))
    
    if ranked_by_cumulative_task_wise:
        ranked_by_cumulative_task_wise.sort(key=lambda x: x[1], reverse=True)
        logging.info("\n📊 按任务级别累积平均准确率排名:")
        for i, (variant, acc) in enumerate(ranked_by_cumulative_task_wise, 1):
            logging.info(f"  {i:2d}. {variant:<30}: {acc:.2f}%")
    
    # 按类别级别累积平均准确率排名
    ranked_by_cumulative_class_wise = []
    for variant in variant_names:
        if variant_stats := statistics_results["variants"][variant]:
            if "cumulative_class_wise_accuracy" in variant_stats and "mean" in variant_stats["cumulative_class_wise_accuracy"]:
                ranked_by_cumulative_class_wise.append((variant, variant_stats["cumulative_class_wise_accuracy"]["mean"]))
    
    if ranked_by_cumulative_class_wise:
        ranked_by_cumulative_class_wise.sort(key=lambda x: x[1], reverse=True)
        logging.info("\n📊 按类别级别累积平均准确率排名:")
        for i, (variant, acc) in enumerate(ranked_by_cumulative_class_wise, 1):
            logging.info(f"  {i:2d}. {variant:<30}: {acc:.2f}%")
    
    # 保存JSON文件
    if save_json:
        try:
            if not output_path:
                # 自动生成输出路径
                first_seed_log_path = all_results[seed_keys[0]].get('log_path', '')
                if first_seed_log_path:
                    parent_dir = Path(first_seed_log_path).parent
                    output_path = output_path = str(parent_dir / "multi_seed_statistics.json")
                else:
                    output_path = "multi_seed_statistics.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_results, f, ensure_ascii=False, indent=2)
            logging.info(f"\n💾 统计结果已保存到: {output_path}")
        except Exception as e:
            logging.warning(f"❌ 保存JSON文件失败: {e}")
    
    return statistics_results