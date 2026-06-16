# classifier/sgd_classifier_builder.py
import torch
import torch.nn as nn
import time
import logging
from classifier.base_classifier_builder import BaseClassifierBuilder
import tqdm


def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")

class SGDClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, device="cuda", max_steps=5000, steps_per_class=5, lr=1e-3):
        self.device = device
        self.base_steps = max_steps
        self.steps_per_class = steps_per_class
        self.lr = lr

    def build(self, stats_dict, linear=True, alpha1=1.0, alpha2=0.0, alpha3=0.0):
        start_time = time.time()
        
        D = list(stats_dict.values())[0].mean.size(0)
        num_classes = len(stats_dict)
        
        # 动态计算最大训练步数：基础4000 + 每类3步
        max_steps = self.base_steps + self.steps_per_class * num_classes
        
        if linear:
            fc = nn.Sequential(
                nn.Linear(D, num_classes)
            ).to(self.device)
        else:
            fc = nn.Sequential(
                nn.Linear(D, 768),
                nn.ReLU(),
                nn.Linear(768, num_classes)
            ).to(self.device)
        
        # 定义优化器和学习率调度器
        opt = torch.optim.AdamW(fc.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-4)
        class_ids = sorted(stats_dict.keys())
        means = []
        covs = []
        for cid in class_ids:
            s = stats_dict[cid]
            means.append(s.mean.float().to(self.device))
            covs.append(s.cov.float().to(self.device))
        means = torch.stack(means)   # [C, D]
        covs = torch.stack(covs)     # [C, D, D]

        # 计算全局协方差
        global_cov = covs.mean(0)
        global_cov = 0.5 * (global_cov + global_cov.T)

        # 应用正则化过程（与RGDA相同）
        C, D, _ = covs.shape
        covs_sym = 0.5 * (covs + covs.transpose(-1, -2))  # 数值对称化
        covs_reg = alpha1 * covs_sym + alpha2 * global_cov + alpha3 * torch.eye(D, device=self.device).unsqueeze(0)
        covs_reg = covs_reg + 1e-3 * torch.eye(D, device=self.device).unsqueeze(0)

        # 计算增强协方差的Cholesky分解
        enhanced_L = torch.linalg.cholesky(covs_reg)  # [C, D, D]

        # === 定义采样函数 ===
        initial_samples_per_class = 256
        
        def resample_pseudo_data():
            """封装采样逻辑，以便重复调用"""
            samples = []
            labels = []
            for i, cid in enumerate(class_ids):
                mu = means[i]
                L_enhanced = enhanced_L[i]
                # 生成样本
                Z = torch.randn(initial_samples_per_class, D, device=self.device)
                X = mu + Z @ L_enhanced.t()
                y = torch.full((initial_samples_per_class,), int(cid), device=self.device)
                samples.append(X)
                labels.append(y)
            return torch.cat(samples), torch.cat(labels)

        # === 在训练前进行初始采样 ===
        print(f"Sampling initial pseudo-samples from each distribution...")
        print(f"Number of classes: {num_classes}, Max training steps: {max_steps}")
        X_initial, Y_initial = resample_pseudo_data()
        print(f"Generated {X_initial.size(0)} initial pseudo-samples")
        
        criterion = nn.CrossEntropyLoss()
        
        # 收敛检测参数
        convergence_threshold = 1e-3
        patience = 100  # 连续多少个step没有显著改善就停止
        best_loss = float('inf')
        patience_counter = 0
        converged = False
        final_step = 0
        
        # 动态采样和训练
        steps_pbar = tqdm.tqdm(
            range(max_steps),
            desc="Training steps",
            leave=False)
        
        current_loss = None
        for step in steps_pbar:
            # 使用当前的样本集进行批次采样
            perm_initial = torch.randperm(X_initial.size(0), device=self.device)
            batch_size = min(128, X_initial.size(0))
            X = X_initial[perm_initial[:batch_size]]
            Y = Y_initial[perm_initial[:batch_size]]
            opt.zero_grad()
            output = fc(X)
            loss = criterion(output, Y)
            loss.backward()
            opt.step()
            sch.step()
            
            if current_loss is None:
                current_loss = loss.item()
            else:
                current_loss = 0.01 * loss.item() + 0.99 * current_loss
            
            # 每1000步进行检查与处理
            if step % 1000 == 0:
                # 1. 收敛检测
                if current_loss < best_loss - convergence_threshold:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        converged = True
                        print(f"Convergence reached at step {step+1} (patience exceeded)")
                        break
                
                print(f"Step {step+1}/{max_steps}, Loss: {current_loss:.6f}, Patience: {patience_counter}/{patience}")

                # 2. 重新采样样本 (Step 0除外，因为Step 0开始前已经初始化过了)
                if step > 0:
                    X_initial, Y_initial = resample_pseudo_data()
                    # 可选：打印重新采样信息
                    # print(f"Resampled pseudo-samples at step {step+1}")
            
            steps_pbar.set_postfix(
                loss=f"{current_loss:.4f}",
                patience=f"{patience_counter}/{patience}")
            
            final_step = step + 1
    
        
        steps_pbar.close()
        
        if converged:
            print(f"Training converged after {final_step} steps")
        else:
            print(f"Training completed after {final_step} steps (max steps reached)")

        end_time = time.time()
        log_time_usage("SGD Classifier build", start_time, end_time)
        return fc.cpu()
