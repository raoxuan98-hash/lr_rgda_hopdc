from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms

from models.base import BaseLearner
from classifier.classifier_builder import ClassifierReconstructor
from compensator.distribution_compensator import DistributionCompensator
from models.distillator import Distiller
from utils.inc_net import BaseNet
from lora import compute_covariances
import math

class GPUMemoryMonitor:
    """GPU显存监控器，用于跟踪和记录显存使用情况"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.memory_history = []
        
    def log_memory(self, stage: str):
        """记录当前显存使用情况"""
        if not self.enabled:
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        self.memory_history.append({
            'stage': stage,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        })
        
        logging.info(f"[GPU Memory] {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
        
    def get_peak_memory(self) -> float:
        """获取峰值显存使用量（GB）"""
        if not self.enabled or not self.memory_history:
            return 0.0
        return max(entry['max_allocated_gb'] for entry in self.memory_history)
        
    def clear_history(self):
        """清空历史记录"""
        self.memory_history.clear()

class EMASmooth:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.value = None
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value
    def get(self):
        return self.value if self.value is not None else 0.0

def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0) 
    label_one_hot = torch.nn.functional.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    total_loss = sce_a * ce_loss + sce_b * rce_loss
    return total_loss

@dataclass
class Timing:
    train: float = 0.0
    drift: float = 0.0
    total: float = 0.0

criterion = lambda logits, targets: symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5)

class SubspaceLoRA(BaseLearner):
    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = BaseNet(args, pretrained=True).to(self._device)
        self.args = args

        self._timings: Timing = Timing()
        self.time_history: List[Dict[str, float]] = []

        self.batch_size: int = args["batch_size"]
        self.iterations: int = args["iterations"]
        self.warmup_steps: int = int(args["warmup_ratio"] * self.iterations)

        self.lrate: float = args["lrate"]
        self.weight_decay: float = args["weight_decay"]
        self.optimizer_type: str = args["optimizer"]
        kd_type = args["kd_type"]

        self.use_kd: bool = args["gamma_kd"] > 0.0
        self.gamma_kd: float = args["gamma_kd"] if self.use_kd else 1.0

        self.use_aux_for_kd: bool = args["use_aux_for_kd"]
        self.update_teacher_each_task: bool = args["update_teacher_each_task"]
        
        self.aux_loader = None
        self.aux_iter = None

        self.covariances: Dict[str, torch.Tensor] | None = None
        self.drift_compensator = DistributionCompensator(
            auxiliary_data_size=args["auxiliary_data_size"],
            compensator_types=args['compensator_types'],
            feature_combination_type=args['feature_combination_type'],
            hopfield_temp=args.get('hopfield_temp', 0.1),
            hopfield_topk=args.get('hopfield_topk', 1000),
            rgda_num_centers=args.get(
                'rgda_mc_num_centers', args.get('rgda_num_centers', 4)),
            rgda_gmm_k=args.get('rgda_gmm_k', 4))
        
        self.classifier_reconstructor = ClassifierReconstructor(
            device=self._device,
            lda_reg_alpha=args['lda_reg_alpha'],
            qda_reg_alpha1=args['qda_reg_alpha1'],
            qda_reg_alpha2=args['qda_reg_alpha2'],
            qda_reg_alpha3=args['qda_reg_alpha3'],
            rgda_num_centers=args.get('rgda_num_centers', 4),
            rgda_train_iter=args.get('rgda_train_iter', 200),
            rgda_fit_lr=args.get('rgda_fit_lr', 0.01),
            rgda_fit_samples_per_class=args.get('rgda_fit_samples_per_class', 16),
            rgda_mc_num_centers=args.get(
                'rgda_mc_num_centers', args.get('rgda_num_centers', 4)),
            rgda_mc_train_iter=args.get(
                'rgda_mc_train_iter', args.get('rgda_train_iter', 200)),
            rgda_mc_fit_lr=args.get(
                'rgda_mc_fit_lr', args.get('rgda_fit_lr', 0.01)),
            rgda_mc_fit_samples_per_class=args.get(
                'rgda_mc_fit_samples_per_class',
                args.get('rgda_fit_samples_per_class', 16)),
            rgda_gmm_sample_mode=args.get('rgda_gmm_sample_mode', 'mean'),
            rgda_gmm_seed=args.get('rgda_gmm_seed', 42),
            rgda_rerank_topk=args.get('rgda_rerank_topk', 20))
        
        # 获取特征维度，处理不同模型类型
        feat_dim = getattr(self.network.vit, 'feature_dim', 768)
        self.distillator = Distiller(
            kd_type=kd_type,
            gamma_kd=args["gamma_kd"],
            update_teacher_each_task=args["update_teacher_each_task"],
            device=self._device,
            feat_dim=feat_dim,
            transform=args["distillation_transform"])
        
        self.get_aux_loader(self.args)
        self.drift_compensator.set_auxiliary_loader(self.aux_loader)

        self.teacher_network = None
        self.prev_network = None
        self.seed: int = args["seed"]
        self.task_count: int = 0
        self.current_task_id = 0
        
        logging.info(f"Optimizer instantiated: lrate={self.lrate}, wd={self.weight_decay}, optimizer={self.optimizer_type}")

        self.loss_smoother = EMASmooth(alpha=0.98)
        self.kd_smoother = EMASmooth(alpha=0.98)
        self.acc_smoother = EMASmooth(alpha=0.98)
        
        # 初始化GPU显存监控器
        self.gpu_monitor = GPUMemoryMonitor(enabled=True)

    def save_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        param_dict = {n: p.detach().cpu() for n, p in self.network.named_parameters() if p.requires_grad}
        payload = {"task": self.current_task_id, "model_state_dict": param_dict}
        path = f"{prefix}/after_task_{self.current_task_id}.pth"
        torch.save(payload, path)
        logging.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, prefix: str) -> None:
        """Save trainable parameters after the current task."""
        path = f"{prefix}/after_task_{self.current_task_id}.pth"
        param_dict = torch.load(path)['model_state_dict']
        self.network.load_state_dict(param_dict, strict=False)
        logging.info(f"Checkpoint loaded from: {path}")

    def equivalence_test(self, prefix: str) -> None:
        """Checkpoint Equivalence Test to ensure reproducibility."""
        self.network.eval()
        # Generate dummy input
        dummy_input = torch.randn(4, 3, 224, 224).to(self._device)
        
        with torch.no_grad():
            original_features = self.network.extract_vector(dummy_input)
        
        # Save and reload
        self.save_checkpoint(prefix)
        # We need to deepcopy the original network to restore if needed, 
        # but loading the checkpoint should perfectly restore it.
        self.load_checkpoint(prefix)
        
        with torch.no_grad():
            reloaded_features = self.network.extract_vector(dummy_input)
            
        max_error = torch.max(torch.abs(original_features - reloaded_features)).item()
        if max_error > 1e-6:
            logging.error(f"Equivalence test failed! Max error: {max_error}")
            raise ValueError(f"Equivalence test failed! Max error: {max_error}")
        else:
            logging.info(f"Equivalence test passed successfully. Max error: {max_error}")

    def handle_drift_compensation(self) -> None:
        """Handle the drift compensation and update classifiers."""
        drift_start = time.time()
        
        # 记录显存使用情况
        self.gpu_monitor.log_memory(f"before_drift_compensation_task_{self.current_task_id}")

        if self.args.get('classifier_only_eval', False):
            logging.info(
                "Classifier-only evaluation: collecting SeqFT statistics and skipping drift compensation.")
            self.drift_compensator.build_current_only(
                self.current_task_id,
                self.network.vit,
                self.train_loader_test_mode,
                low_rank=False)
        else:
            self.drift_compensator.build_all_variants(
                self.current_task_id,
                self.prev_network.vit,
                self.network.vit,
                self.train_loader_test_mode)

        self._timings.drift = time.time() - drift_start
        
        self.gpu_monitor.log_memory(f"after_drift_compensation_task_{self.current_task_id}")

    def refine_classifiers(self):
        logging.info(f"Building classifiers from {len(self.drift_compensator.variants)} variants...")
        variant_names = list(self.drift_compensator.variants.keys())
        logging.info(f"Available variants: {variant_names}")
        
        classifier_types = self.args.get('classifier_types', ['lda', 'qda'])
        self.fc_dict = self.classifier_reconstructor.build_classifiers(
            self.drift_compensator.variants,
            classifier_type=classifier_types)
            
        logging.info(f"Built {len(self.fc_dict)} classifiers: {list(self.fc_dict.keys())}")
        logging.info(f"Classifier types used: {classifier_types}")

    def after_task(self) -> None:
        """Update class counters after finishing a task."""
        self._known_classes = self._total_classes
        self.update_projection_matrices()
        
        self.task_count += 1

    def incremental_train(self, data_manager) -> None:
        start_time = time.time()
        task_id = self.current_task_id
        task_size = data_manager.get_task_size(task_id)

        # 只在第一个任务之后才更新教师网络，避免初始KD损失不为零
        if self.current_task_id > 0:
            self.distillator.update_teacher(self.network)
            print(f"[SubspaceLoRA] Teacher network updated for task {self.current_task_id}")
        else:
            print(f"[SubspaceLoRA] Skipping teacher update for first task {self.current_task_id} to avoid initial KD loss")
        
        self.prev_network = copy.deepcopy(self.network).cpu()

        # if hasattr(self.prev_network.vit, 'finalize_without_lora'):
        self.prev_network.vit.finalize_without_lora()
        self.network.vit.merge_lora_weights()

        self._total_classes = self._known_classes + task_size
        self.current_task_id += 1
        self.topk = min(self._total_classes, 5)

        train_set = data_manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="train")
        test_set = data_manager.get_incremental_subset(
            task=task_id, source="test", cumulative=True, mode="test")
        train_set_test_mode = data_manager.get_incremental_subset(
            task=task_id, source="train", cumulative=False, mode="test")

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=4, pin_memory=True, persistent_workers=False)
        
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size * 4, shuffle=False,
                                    num_workers=4, pin_memory=True, persistent_workers=False)

        dataset_size = len(train_set_test_mode)
        max_samples = getattr(self, 'max_train_test_samples', 5000)
        sampler = None

        # if dataset_size > max_samples:
        #     indices = torch.randperm(dataset_size)[:max_samples].tolist()
        #     sampler = SubsetRandomSampler(indices)
        #     print(f"⚠️ Dataset too large ({dataset_size}), sampling {max_samples} examples for test-mode training set.")

        self.train_loader_test_mode = DataLoader(
            train_set_test_mode,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            num_workers=3,
            pin_memory=True,
            persistent_workers=True)

        # === 初始化 DriftCompensator 所需的 loader，但不用于 KD ===
        self.network.update_fc(task_size)
        self.network.fc.to(self._device)

        # 获取当前任务的数据集信息，兼容增量拆分和普通模式
        try:
            dataset_info = data_manager.datasets[task_id]
            dataset_name = dataset_info.get('original_dataset_name', dataset_info['name'])
        except:
            dataset_name = data_manager.dataset_name
        
        logging.info(
            "System training on classes %d-%d (%s)",
            self._known_classes,
            self._total_classes,
            dataset_name.lower())

        if self.args.get('classifier_only_eval', False):
            logging.info(
                "Classifier-only evaluation: skipping backbone/LoRA training for task %d.",
                self.current_task_id)
            self._timings.train = 0.0
        elif self.args['eval_only']:
            self.load_checkpoint(self.args["log_path"])
        else:
            self.print_parameter_statistics(task_id)
            self.reset_loss_smoothers()
            self.system_training()
            self.equivalence_test(self.args["log_path"])
            # self.save_checkpoint(self.args["log_path"])

        self.handle_drift_compensation()
        self._timings.total = time.time() - start_time

        # 权重插值：只在启用插值且alpha < 1.0时执行
        if (self.args.get('enable_weight_interpolation', False) and
            self.args.get('interpolation_alpha', 1.0) < 1.0):
            alpha = self.args.get('interpolation_alpha', 1.0)
            logging.info(f"任务 {self.current_task_id} 完成后执行权重插值，alpha={alpha}")
            try:
                self.trigger_interpolation(alpha, self.current_task_id)
            except Exception as e:
                logging.error(f"权重插值失败: {str(e)}")

        logging.info(
            "Task %d finished total: %.2f s | train: %.2f s | drift: %.2f s", self.current_task_id, self._timings.total, self._timings.train, self._timings.drift)

        
    def make_optimizer(
        self,
        lora_params: List[torch.nn.Parameter],
        fc_params: List[torch.nn.Parameter]):

        """Create optimizer according to ``self.optimizer_type``."""
        distill_params = filter(lambda p: p.requires_grad, self.distillator.parameters())

        param_groups = [
            {"params": lora_params, "lr": self.lrate, "weight_decay": self.weight_decay},
            {"params": fc_params, "lr": 1e-3 if self.optimizer_type == "adamw" else 10 * self.lrate, "weight_decay": self.weight_decay},
            {"params": distill_params, "lr": 10 * self.lrate, "weight_decay": self.weight_decay},
        ]

        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.optimizer_type == "adamw":
            optimizer = optim.AdamW(param_groups)
        elif self.optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        def lora_lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.iterations - self.warmup_steps)
                initial_lr = self.lrate
                eta_min = getattr(self, 'eta_min', self.lrate / 3)
                lr_ratio = eta_min / initial_lr
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return lr_ratio + cosine_decay * (1.0 - lr_ratio)
        
        def const_lr_lambda(step):
            return 1.0
        
        lr_lambdas = [lora_lr_lambda, lora_lr_lambda, const_lr_lambda]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas, last_epoch=-1)

        return optimizer, scheduler


    def system_training(self) -> None:
        """Train ViT + new classifier head for ``self.epochs`` epochs."""
        if self.iterations <= 0:
            logging.info("Skipping system training because iterations <= 0.")
            self._timings.train = 0.0
            return

        fc_params = self.network.fc.parameters()
        # 处理不同模型类型的参数获取
        if hasattr(self.network.vit, 'get_param_groups'):
            lora_params = self.network.vit.get_param_groups()
        else:
            lora_params = [p for p in self.network.vit.parameters() if p.requires_grad]
        optimizer, scheduler = self.make_optimizer(
            lora_params,
            fc_params)
        
        start = time.time()
        self.network.train()
        
        step = 0
        done = False
        while True:
            for batch in self.train_loader:
                inputs, targets = batch[0], batch[1]
                loss, n_corr, kd_term = self.process_batch(inputs, targets, optimizer)

                batch_acc = n_corr / inputs.size(0)

                # 使用 EMASmooth 平滑
                smoothed_loss = self.loss_smoother.update(loss)
                smoothed_kd = self.kd_smoother.update(kd_term)
                smoothed_acc = self.acc_smoother.update(batch_acc)

                if (step + 1) % 50 == 0:
                    logging.info(
                        'step: %d, loss: %.4f, kd_loss: %.4f, acc: %.4f',
                        step, smoothed_loss, smoothed_kd, smoothed_acc)

                scheduler.step()
                step += 1

                if step == self.iterations:
                    done = True
                    break
            if done:
                break

        self._timings.train = time.time() - start

    def process_batch(self, inputs, targets, optimizer):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        
        feats = self.network.forward_features(inputs)
        kd_term = self.distillator(inputs, feats)
        logits = self.network.fc(feats)
        
        new_targets_rel = torch.where(targets - self._known_classes >= 0, targets - self._known_classes, -100)
        new_logits = logits[:, self._known_classes:]
        
        sce = criterion(new_logits, new_targets_rel)
        loss = sce + kd_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            n_correct = (pred == targets).sum().item()

        kd_raw = kd_term.item() / self.gamma_kd

        return loss.item(), n_correct, kd_raw

    def evaluate(
        self,
        loader: DataLoader,
        fc_dict):
        """
        Evaluate model on test data.
        For cross-domain scenarios, compute accuracy per dataset and average across datasets.
        For regular scenarios, compute overall accuracy.
        """
        self.network.eval()
        
        # Check if this is cross-domain scenario
        is_cross_domain = self.args['cross_domain']
        
        if is_cross_domain:
            result = self._evaluate_cross_domain(loader, fc_dict)
        else:
            result = self._evaluate_within_domain(loader, fc_dict)
           
        return result
    
    def _evaluate_within_domain(self, loader: DataLoader, fc_dict):
        """Regular evaluation: compute overall accuracy across all samples"""
        self.network.eval()
        total = 0
        corrects = {}
        for name, fc in fc_dict.items():
            corrects[name] = 0
            fc.to(self._device)

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self._device)
                targets = batch[1]
                
                feats = self.network.forward_features(inputs)

                for name, fc in fc_dict.items():
                    preds = fc(feats).argmax(dim=1).cpu()
                    corrects[name] += (preds == targets).sum().item()
                    
                total += targets.size(0)
        
        for name, correct in corrects.items():
            corrects[name] = float(np.around(100 * correct / total, 2))
        
        # 返回与cross_domain评估相同的数据结构，保持一致性
        return {
            'average_accs': corrects,
            'average_task_wise_accs': {name: {} for name in fc_dict.keys()},
            'average_class_wise_accs': {name: 0.0 for name in fc_dict.keys()}
        }
    

    def _evaluate_cross_domain(self, loader: DataLoader, fc_dict):
        self.network.eval()
        for name, fc in fc_dict.items():
            fc.to(self._device)
        
        targets_all = []
        preds_all = {}
        for name in fc_dict.keys():
            preds_all[name] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                inputs = batch[0].to(self._device)
                targets = batch[1]
                
                targets_all.append(targets)
                feats = self.network.forward_features(inputs)
                
                for name, fc in fc_dict.items():
                    preds = fc(feats).argmax(dim=1).cpu()
                    preds_all[name].append(preds)

            targets_all = torch.cat(targets_all, dim=0)
            for name in fc_dict.keys():
                preds_all[name] = torch.cat(preds_all[name], dim=0)
        
        """ Calculate overall accuracy across all samples """
        overall_accs = {}
        for name, preds in preds_all.items():
            overall_accs[name] = float(np.around(100 * (preds == targets_all).sum().item() / targets_all.size(0), 2))

        """ Calculate task-wise average accuracy"""
        overall_task_wise_avg_accs = {}
        for name in fc_dict.keys():
            overall_task_wise_avg_accs[name] = {}
        
        for task_id in range(self.current_task_id):
            task_start_label = self.data_manager.global_label_offset[task_id]
            # 对于最后一个任务，使用总类别数作为结束标签
            if task_id + 1 < len(self.data_manager.global_label_offset):
                task_end_label = self.data_manager.global_label_offset[task_id + 1]
            else:
                task_end_label = self._total_classes
            # print(task_start_label)
            # print(task_end_label)
            mask = (targets_all >= task_start_label) & (targets_all < task_end_label)
            for name, preds in preds_all.items():
                if mask.sum().item() > 0:
                    task_acc = float(np.around(100 * (preds[mask] == targets_all[mask]).sum().item() / mask.sum().item(), 2))
                else:
                    task_acc = 0.0
                overall_task_wise_avg_accs[name][task_id] = task_acc


        """ Calculate class-wise average accuracy """
        overall_class_wise_accs = {}
        for name in fc_dict.keys():
            overall_class_wise_accs[name] = {}
        unique_labels = torch.unique(targets_all)
        for label in unique_labels:
            mask = (targets_all == label)
            for name, preds in preds_all.items():
                class_wise_accs = float(np.around(100 * (preds[mask] == targets_all[mask]).sum().item() / mask.sum().item(), 2))
                overall_class_wise_accs[name][label.item()] = class_wise_accs
        overall_class_wise_avg_accs = {}
        for name in fc_dict.keys():
            overall_class_wise_avg_accs[name] = np.mean(list(overall_class_wise_accs[name].values()))

        task_stats = {
            'average_accs': overall_accs,
            'average_task_wise_accs': overall_task_wise_avg_accs,
            'average_class_wise_accs': overall_class_wise_avg_accs}
        
        return task_stats
    
    def eval_task(self):
        self.gpu_monitor.log_memory(f"before_evaluation_task_{self.current_task_id}")
        eval_start_time = time.time()
        
        logging.info(f"Evaluating with {len(self.fc_dict)} classifiers...")
        
        results = self.evaluate(
            self.test_loader,
            fc_dict=self.fc_dict)
        
        eval_end_time = time.time()
        eval_elapsed_time = eval_end_time - eval_start_time
        logging.info(f"[Time] Evaluation for task {self.current_task_id}: {eval_elapsed_time:.4f}s")

        self.gpu_monitor.log_memory(f"after_evaluation_task_{self.current_task_id}")
        self.all_task_results[self.current_task_id] = results

        peak_memory = self.gpu_monitor.get_peak_memory()
        logging.info(f"[GPU Memory] Peak memory usage up to task {self.current_task_id}: {peak_memory:.2f}GB")
        return results

    def update_projection_matrices(self):
        if hasattr(self.network.vit, 'use_projection') and self.network.vit.use_projection:
            if self.current_task_id >= 0:
                new_covs = compute_covariances(self.network.vit, self.train_loader_test_mode)
                # 将新的协方差矩阵移到CPU以节省GPU显存
                new_covs_cpu = {k: v.cpu() for k, v in new_covs.items()}
                
                if self.covariances is None:
                    self.covariances = new_covs_cpu
                else:
                    for k in self.covariances:
                        self.covariances[k] = 0.9 * self.covariances[k] + new_covs_cpu[k] + 1e-7 * torch.eye(self.covariances[k].size(0))
                # 只在需要时再将covariances移到GPU
                covariances_gpu = {k: v.to(self._device) for k, v in self.covariances.items()}
                self.network.update_projection_matrices(covariances_gpu)
                # 清理GPU上的临时covariances
                del covariances_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def loop(self, data_manager) -> Dict[str, Any]:
        self.data_manager = data_manager
        self.all_task_results = {}
        
        # 获取evaluate_final_only参数
        evaluate_final_only = self.args.get('evaluate_final_only', True)
        total_tasks = data_manager.nb_tasks
        final_analysis = None  # 初始化final_analysis
        
        for task_idx in range(total_tasks):
            self.incremental_train(data_manager)
            
            # 根据evaluate_final_only参数决定是否构建分类器和进行评估
            if not evaluate_final_only or (task_idx == total_tasks - 1):
                # 当evaluate_final_only=False时，每个任务后都构建分类器并评估
                # 当evaluate_final_only=True时，只在最后一个任务后构建分类器并评估
                self.refine_classifiers()
                logging.info(f"Evaluating after task {self.current_task_id}...")
                self.eval_task()
                print(self.all_task_results)
                print(self.all_task_results.keys())
                dataset_names = getattr(self.data_manager, 'dataset_names', None)
                final_analysis = self.analyze_task_results(self.all_task_results, dataset_names)
            else:
                # 跳过中间任务的分类器构建和评估，只记录基本信息
                logging.info(f"Task {self.current_task_id} completed (classifier building and evaluation skipped due to evaluate_final_only=True)")
                
            self.after_task()

        # 如果final_analysis为None（即evaluate_final_only=True且跳过了所有评估），
        # 返回一个空字典而不是None，以匹配函数的返回类型声明
        return final_analysis if final_analysis is not None else {}
    def analyze_task_results(self, all_task_results, dataset_names):
        task_ids = sorted(all_task_results.keys())
        last_task_id = task_ids[-1]
        # 提取 variant 名称
        variant_names = set()
        for task_dict in all_task_results.values():
            if 'average_accs' in task_dict:
                variant_names.update(task_dict['average_accs'].keys())
        variant_names = sorted(variant_names)
        # 计算数据集级别准确率（data-wise）
        data_wise_accuracies = {}
        if 'average_accs' in all_task_results[last_task_id]:
            data_wise_accuracies = all_task_results[last_task_id]['average_accs'].copy()
        # 计算累积平均准确率（data-wise）
        cumulative_average_accuracies = {}
        for variant in variant_names:
            accs = []
            for task_id in task_ids:
                task_dict = all_task_results[task_id]
                if 'average_accs' in task_dict and variant in task_dict['average_accs']:
                    acc_value = task_dict['average_accs'][variant]
                    if isinstance(acc_value, (int, float)):
                        accs.append(acc_value)
            
            if accs:
                cumulative_average_accuracies[variant] = float(np.mean(accs))
            else:
                cumulative_average_accuracies[variant] = 0.0
        
        # 计算任务级别累积平均准确率（task-wise）
        cumulative_task_wise_accuracies = {}
        for variant in variant_names:
            task_wise_accs = []
            for task_id in task_ids:
                task_dict = all_task_results[task_id]
                if 'average_task_wise_accs' in task_dict and variant in task_dict['average_task_wise_accs']:
                    task_accs = list(task_dict['average_task_wise_accs'][variant].values())
                    if task_accs:
                        task_wise_accs.append(np.mean(task_accs))
            
            if task_wise_accs:
                cumulative_task_wise_accuracies[variant] = float(np.mean(task_wise_accs))
            else:
                cumulative_task_wise_accuracies[variant] = 0.0
        
        # 计算类别级别累积平均准确率（class-wise）
        cumulative_class_wise_accuracies = {}
        for variant in variant_names:
            class_wise_accs = []
            for task_id in task_ids:
                task_dict = all_task_results[task_id]
                if 'average_class_wise_accs' in task_dict and variant in task_dict['average_class_wise_accs']:
                    class_acc = task_dict['average_class_wise_accs'][variant]
                    if isinstance(class_acc, (int, float)):
                        class_wise_accs.append(class_acc)
            
            if class_wise_accs:
                cumulative_class_wise_accuracies[variant] = float(np.mean(class_wise_accs))
            else:
                cumulative_class_wise_accuracies[variant] = 0.0
        is_cross_domain = self.args['cross_domain']
        
        if is_cross_domain:
            logging.info("📊 Cross-domain Evaluation Results:")
            logging.info("   ── Overall Performance Summary ──")
            
            # 创建统一的性能表格
            performance_data = []
            for variant in variant_names:
                current_acc = data_wise_accuracies.get(variant, 0.0)
                task_wise_avg = 0.0
                class_wise_avg = 0.0
                
                # 获取任务级别平均准确率
                if 'average_task_wise_accs' in all_task_results[last_task_id]:
                    task_wise_data = all_task_results[last_task_id]['average_task_wise_accs']
                    if variant in task_wise_data:
                        task_accs = list(task_wise_data[variant].values())
                        if task_accs:
                            task_wise_avg = float(np.mean(task_accs))
                
                # 获取类别级别平均准确率
                if 'average_class_wise_accs' in all_task_results[last_task_id]:
                    class_wise_data = all_task_results[last_task_id]['average_class_wise_accs']
                    if variant in class_wise_data:
                        class_wise_avg = class_wise_data[variant]
                
                performance_data.append({
                    'variant': variant,
                    'current': current_acc,
                    'task_wise': task_wise_avg,
                    'class_wise': class_wise_avg
                })
            
            # 按当前准确率排序
            performance_data.sort(key=lambda x: x['current'], reverse=True)
            
            # 输出统一表格
            logging.info("   Method                         │ Dataset │ Per-Task │ Per-Class")
            logging.info("   ───────────────────────────────┼─────────┼───────────┼───────────")
            for data in performance_data:
                logging.info(f"   {data['variant']:<30} │ {data['current']:6.2f}% │ {data['task_wise']:8.2f}% │ {data['class_wise']:9.2f}%")
            
            # 详细任务信息（可选，如果需要的话）
            if 'average_task_wise_accs' in all_task_results[last_task_id]:
                logging.info("")
                logging.info("   ── Detailed Per-Task Performance ──")
                task_wise_data = all_task_results[last_task_id]['average_task_wise_accs']
                
                # 按性能排序 variants
                sorted_variants = sorted(variant_names,
                                    key=lambda v: data_wise_accuracies.get(v, 0.0),
                                    reverse=True)
                
                # 按数据集名称分组统计结果
                for variant in sorted_variants:
                    if variant in task_wise_data:
                        # 第一步：按数据集名称分组所有任务结果
                        dataset_groups = {}  # dataset_name -> [(task_id, acc, test_samples), ...]
                        
                        for task_id in sorted(task_wise_data[variant].keys()):
                            # 获取数据集名称
                            if hasattr(self, 'data_manager') and self.data_manager:
                                dataset_info = self.data_manager.datasets[task_id] if task_id < len(self.data_manager.datasets) else None
                                if dataset_info and 'original_dataset_name' in dataset_info:
                                    dataset_name = dataset_info['original_dataset_name']
                                elif dataset_info and 'name' in dataset_info:
                                    dataset_name = dataset_info['name']
                                elif dataset_names and task_id < len(dataset_names):
                                    dataset_name = dataset_names[task_id]
                                else:
                                    dataset_name = f"Task {task_id}"
                            else:
                                dataset_name = dataset_names[task_id] if dataset_names and task_id < len(dataset_names) else f"Task {task_id}"
                            
                            # 清理数据集名称
                            if dataset_name.endswith('_split_0') or dataset_name.endswith('_split_1'):
                                dataset_name = dataset_name.split('_split_')[0]
                            elif dataset_name.endswith('_224'):
                                pass  # 保持_224后缀以区分不同分辨率
                            
                            acc = task_wise_data[variant][task_id]
                            
                            # 获取测试样本数量作为权重
                            if hasattr(self, 'data_manager') and self.data_manager:
                                dataset_info = self.data_manager.datasets[task_id] if task_id < len(self.data_manager.datasets) else None
                                test_samples = len(dataset_info['test_data']) if dataset_info and 'test_data' in dataset_info else 1
                            else:
                                test_samples = 1
                            
                            # 按数据集名称分组
                            if dataset_name not in dataset_groups:
                                dataset_groups[dataset_name] = []
                            dataset_groups[dataset_name].append((task_id, acc, test_samples))
                        
                        # 第二步：显示分组结果
                        logging.info(f"   📈 {variant}:")
                        for dataset_name, task_results in dataset_groups.items():
                            if len(task_results) > 1:
                                # 有多个子集，计算加权平均
                                accuracies = [result[1] for result in task_results]
                                weights = [result[2] for result in task_results]
                                weighted_avg = np.average(accuracies, weights=weights)
                                total_samples = sum(weights)
                                
                                # 显示每个子集的结果
                                for task_id, acc, samples in task_results:
                                    logging.info(f"        {dataset_name} (task {task_id}) : {acc:.2f}% ({samples} samples)")
                                
                                # 显示加权平均结果
                                logging.info(f"        {dataset_name} (weighted avg) : {weighted_avg:.2f}% (total: {total_samples} samples)")
                            else:
                                # 只有一个子集，直接显示
                                task_id, acc, samples = task_results[0]
                                logging.info(f"        {dataset_name:<20} : {acc:.2f}% ({samples} samples)")
            
            # 性能总结
            logging.info("")
            logging.info("   ── Performance Summary ──")
            best_variant = max(performance_data, key=lambda x: x['current'])
            logging.info(f"   🏆 Best Performing Method: '{best_variant['variant']}'")
            logging.info(f"   📊 Best Accuracy: {best_variant['current']:.2f}%")
            
            return {
                "last_task_id": last_task_id,
                "last_task_accuracies": data_wise_accuracies,  # 保持向后兼容性
                "data_wise_accuracies": data_wise_accuracies,
                "average_accuracies": cumulative_average_accuracies,  # 保持向后兼容性
                "cumulative_average_accuracies": cumulative_average_accuracies,
                "cumulative_task_wise_accuracies": cumulative_task_wise_accuracies,
                "cumulative_class_wise_accuracies": cumulative_class_wise_accuracies,
                "task_wise_accuracies": {data['variant']: data['task_wise'] for data in performance_data},
                "class_wise_accuracies": {data['variant']: data['class_wise'] for data in performance_data}}
             
        else:
            # For within-domain datasets: output only average_accuracy
            logging.info("📊 Within-domain Evaluation Results:")
            logging.info("   ── Average Accuracy Across Tasks (%) ──")
            for variant in variant_names:
                logging.info(f"      {variant:<20} : {cumulative_average_accuracies[variant]:.2f}%")
            
            logging.info("   ── Final Task Accuracy (%) ──")
            for variant in variant_names:
                logging.info(f"      {variant:<20} : {data_wise_accuracies[variant]:.2f}%")
            
            # Optional: Identify best variants and log summary
            best_last = max(data_wise_accuracies.items(), key=lambda x: x[1])[0] if data_wise_accuracies else None
            best_avg = max(cumulative_average_accuracies.items(), key=lambda x: x[1])[0] if cumulative_average_accuracies else None

            if best_last and best_avg:
                if best_last == best_avg:
                    summary = f" Variant '{best_last}' is best in both final task and average performance."
                else:
                    summary = f" Best in Final Task: '{best_last}' | Best Average: '{best_avg}'"
            else:
                summary = " No valid variants found for comparison."

            logging.info("   ── Summary ──")
            logging.info(f"      {summary}")
            
            return {
                "last_task_id": last_task_id,
                "last_task_accuracies": data_wise_accuracies,  # 保持向后兼容性
                "data_wise_accuracies": data_wise_accuracies,
                "average_accuracies": cumulative_average_accuracies,  # 保持向后兼容性
                "cumulative_average_accuracies": cumulative_average_accuracies,
                "cumulative_task_wise_accuracies": cumulative_task_wise_accuracies,
                "cumulative_class_wise_accuracies": cumulative_class_wise_accuracies}
    
    def get_aux_loader(self, args):
        aux_dataset_type = args.get('aux_dataset', 'flickr8k')
        num_samples =args['auxiliary_data_size']

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        if aux_dataset_type == 'imagenet':
            dataset = datasets.ImageFolder(args['auxiliary_data_path'] + '/ImageNet-2012/train', transform=transform)
        elif aux_dataset_type == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        elif aux_dataset_type == 'svhn':
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        elif aux_dataset_type == 'flickr8k':
            dataset = datasets.ImageFolder(args['auxiliary_data_path'] + '/flickr8k', transform=transform)
        else:
            raise ValueError(f"不支持的 aux_dataset_type: {aux_dataset_type}")

        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        train_subset = Subset(dataset, indices)

        self.aux_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.aux_trainset = train_subset
        return self.aux_loader

    def count_trainable_parameters(self) -> Dict[str, int]:
        """统计各部分的训练参数数量"""
        param_counts = {}
        
        # 获取模型参数，处理不同模型类型
        if hasattr(self.network.vit, 'get_param_groups'):
            lora_params = self.network.vit.get_param_groups()
        else:
            # 对于全参数微调模型，获取所有可训练参数
            lora_params = [p for p in self.network.vit.parameters() if p.requires_grad]
        
        lora_count = sum(p.numel() for p in lora_params)
        param_counts["lora"] = lora_count
        
        # 分类头参数
        fc_count = sum(p.numel() for p in self.network.fc.parameters())
        param_counts["classifier"] = fc_count
        total_count = lora_count + fc_count
        param_counts["total"] = total_count
        
        return param_counts

    def count_total_parameters(self) -> int:
        """统计模型总参数数量（包括冻结参数）"""
        return sum(p.numel() for p in self.network.parameters())

    def print_parameter_statistics(self, task_id: int) -> None:
        """打印参数统计信息"""
        trainable_params = self.count_trainable_parameters()
        total_params = self.count_total_parameters()
        
        logging.info(f"=== 任务 {task_id} 参数统计 ===")
        logging.info(f"总模型参数: {total_params:,}")
        logging.info(f"可训练参数: {trainable_params['total']:,}")
        logging.info(f"  - LoRA参数: {trainable_params['lora']:,}")
        logging.info(f"  - 分类头参数: {trainable_params['classifier']:,}")
        
        # 计算参数效率
        efficiency = (trainable_params['total'] / total_params) * 100
        logging.info(f"参数效率: {efficiency:.2f}%")

    def reset_loss_smoothers(self) -> None:
        """重置所有损失平滑器"""
        self.loss_smoother = EMASmooth(alpha=0.98)
        self.kd_smoother = EMASmooth(alpha=0.98)
        self.acc_smoother = EMASmooth(alpha=0.98)
        
        logging.debug("损失平滑器已重置")

    def interpolate_weights_with_prev_net(self, alpha: float = 0.5) -> None:
        """
        将当前网络的可训练参数与prev_net的可训练参数进行权重插值。
        
        Args:
            alpha (float): 插值权重，范围[0,1]。0表示完全使用prev_net参数，1表示完全使用当前网络参数。
                          默认0.5表示对等权重。
        
        Returns:
            None
        
        Raises:
            ValueError: 当prev_network为None或参数不匹配时抛出异常
        """
        if self.prev_network is None:
            raise ValueError("prev_network is None. Cannot perform interpolation without previous network.")
        
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha should be in range [0, 1], got {alpha}")
        
        logging.info(f"开始权重插值，alpha={alpha}")
        
        # 获取当前网络和prev_network的可训练参数
        curr_params = dict(self.network.named_parameters())
        prev_params = dict(self.prev_network.named_parameters())
        
        # 检查参数名称是否匹配
        curr_param_names = set(curr_params.keys())
        prev_param_names = set(prev_params.keys())
        
        if curr_param_names != prev_param_names:
            missing_in_prev = curr_param_names - prev_param_names
            missing_in_curr = prev_param_names - curr_param_names
            
            if missing_in_prev:
                logging.warning(f"当前网络有但prev_network没有的参数: {missing_in_prev}")
            if missing_in_curr:
                logging.warning(f"prev_network有但当前网络没有的参数: {missing_in_curr}")
        
        # 进行参数插值
        interpolated_count = 0
        for name, curr_param in curr_params.items():
            if name in prev_params:
                prev_param = prev_params[name]
                
                # 只对可训练参数进行插值
                if curr_param.requires_grad and prev_param.requires_grad:
                    # 确保参数在同一设备上
                    if curr_param.device != prev_param.device:
                        prev_param = prev_param.to(curr_param.device)
                    
                    # 检查参数形状是否匹配
                    if curr_param.shape != prev_param.shape:
                        logging.warning(f"参数 {name} 形状不匹配: {curr_param.shape} vs {prev_param.shape}")
                        continue
                    
                    # 执行插值: new_param = alpha * curr_param + (1-alpha) * prev_param
                    new_param = alpha * curr_param.data + (1 - alpha) * prev_param.data
                    curr_param.data.copy_(new_param)
                    
                    interpolated_count += 1
                else:
                    logging.debug(f"跳过参数 {name}，因为它不是可训练参数")
            else:
                logging.warning(f"参数 {name} 在prev_network中不存在，跳过插值")
        
        logging.info(f"权重插值完成，共处理 {interpolated_count} 个可训练参数")
        
        # 可选：将插值后的参数移回原设备
        self.network.to(self._device)

    def trigger_interpolation(self, alpha: float = 0.5, task_id = None) -> None:
        """
        触发权重插值的便捷方法。
        
        Args:
            alpha (float): 插值权重，默认0.5
            task_id (int, optional): 指定任务ID，如果为None则使用当前任务ID
        
        Returns:
            None
        """
        if task_id is None:
            task_id = self.current_task_id
            
        logging.info(f"触发任务 {task_id} 的权重插值，alpha={alpha}")
        
        try:
            self.interpolate_weights_with_prev_net(alpha)
            logging.info(f"任务 {task_id} 权重插值成功完成")
        except Exception as e:
            logging.error(f"任务 {task_id} 权重插值失败: {str(e)}")
            raise
