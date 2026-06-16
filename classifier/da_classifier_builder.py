# classifier/gaussian_classifier_builder.py
import torch
import torch.nn as nn
import time
import logging
from classifier.gaussian_classifier import RegularizedGaussianDA, LinearLDAClassifier, LowRankGaussianDA
from classifier.base_classifier_builder import BaseClassifierBuilder


def log_time_usage(operation_name: str, start_time: float, end_time: float):
    """记录时间损耗情况"""
    elapsed_time = end_time - start_time
    logging.info(f"[Time] {operation_name}: {elapsed_time:.4f}s")

class LDAClassifierBuilder(BaseClassifierBuilder):
    def __init__(self, reg_alpha=0.3, device="cuda"):
        self.reg_alpha = reg_alpha
        self.device = device

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        model = LinearLDAClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            lda_reg_alpha=self.reg_alpha
        ).to(self.device)
        
        end_time = time.time()
        log_time_usage("LDA Classifier build", start_time, end_time)
        
        return model


class LRRGDAClassifierBuilder(BaseClassifierBuilder):
    """Build the low-rank RGDA classifier used in the paper."""

    def __init__(
        self,
        rgda_alpha1=0.2,
        rgda_alpha2=0.2,
        rgda_alpha3=0.2,
        qda_reg_alpha1=None,
        qda_reg_alpha2=None,
        qda_reg_alpha3=None,
        low_rank=True,
        rank = 64,
        device="cuda",
        num_centers=1,
        train_iter=0,
        fit_lr=0.01,
        fit_samples_per_class=0,
        fit_sample_mode="mean",
        fit_seed=42,
        fit_verbose=True,
    ):
        self.rgda_alpha1 = rgda_alpha1 if qda_reg_alpha1 is None else qda_reg_alpha1
        self.rgda_alpha2 = rgda_alpha2 if qda_reg_alpha2 is None else qda_reg_alpha2
        self.rgda_alpha3 = rgda_alpha3 if qda_reg_alpha3 is None else qda_reg_alpha3
        self.qda_reg_alpha1 = self.rgda_alpha1
        self.qda_reg_alpha2 = self.rgda_alpha2
        self.qda_reg_alpha3 = self.rgda_alpha3
        self.device = device
        self.low_rank = low_rank
        self.rank = rank
        self.num_centers = max(1, int(num_centers))
        self.train_iter = max(0, int(train_iter))
        self.fit_lr = fit_lr
        self.fit_samples_per_class = max(0, int(fit_samples_per_class))
        self.fit_sample_mode = fit_sample_mode
        self.fit_seed = int(fit_seed)
        self.fit_verbose = fit_verbose

    def _center_means_from_stats(self, stats_dict):
        if self.num_centers <= 1:
            return None
        center_means = {}
        for cid, stats in stats_dict.items():
            centers = getattr(stats, "centers", None)
            if centers is None:
                centers = stats.mean.unsqueeze(0).repeat(self.num_centers, 1)
            elif centers.size(0) != self.num_centers:
                repeat = (self.num_centers + centers.size(0) - 1) // centers.size(0)
                centers = centers.repeat((repeat, 1))[:self.num_centers]
            center_means[cid] = centers
        return center_means

    def _sample_fit_data(self, stats_dict):
        if self.train_iter <= 0 or self.fit_samples_per_class <= 0:
            return None, None
        features, labels = [], []
        for cid in sorted(stats_dict.keys()):
            stat = stats_dict[cid]
            if getattr(stat, "gmm_means", None) is not None:
                samples = stat.sample_gmm(
                    n_samples=self.fit_samples_per_class,
                    mode=self.fit_sample_mode,
                    seed=self.fit_seed + int(cid),
                ).cpu()
            else:
                logging.warning(
                    "Class %s has no stored GMM replay statistics; falling back to single Gaussian sampling.",
                    cid,
                )
                samples = stat.sample(n_samples=self.fit_samples_per_class).cpu()
            features.append(samples)
            labels.append(torch.full((samples.size(0),), int(cid), dtype=torch.long))
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def build(self, stats_dict):
        start_time = time.time()
        
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        if self.low_rank:
            model = LowRankGaussianDA(
                stats_dict=stats_dict,
                class_priors=priors,
                rank = self.rank,
                qda_reg_alpha1=self.qda_reg_alpha1,
                qda_reg_alpha2=self.qda_reg_alpha2,
                qda_reg_alpha3=self.qda_reg_alpha3,
                device=self.device,
                center_means=self._center_means_from_stats(stats_dict),
            ).to(self.device)
        else:
            model = RegularizedGaussianDA(
                stats_dict=stats_dict,
                class_priors=priors,
                qda_reg_alpha1=self.qda_reg_alpha1,
                qda_reg_alpha2=self.qda_reg_alpha2,
                qda_reg_alpha3=self.qda_reg_alpha3,
            ).to(self.device)

        if self.low_rank and self.train_iter > 0:
            features, labels = self._sample_fit_data(stats_dict)
            if features is None:
                logging.warning(
                    "Skipping LR-RGDA fit because rgda_fit_samples_per_class is 0.")
            else:
                model.fit(
                    features.to(self.device),
                    labels.to(self.device),
                    iterations=self.train_iter,
                    lr=self.fit_lr,
                    verbose=self.fit_verbose,
                )
        
        end_time = time.time()
        log_time_usage("LR-RGDA Classifier build", start_time, end_time)
        
        return model


class LDATopKLRRGDARerankClassifier(nn.Module):
    """Use lightweight LDA for coarse top-k retrieval and LR-RGDA for reranking."""

    def __init__(self, lda_model: nn.Module, rgda_model: nn.Module, topk: int = 50):
        super().__init__()
        self.lda_model = lda_model
        self.rgda_model = rgda_model
        self.topk = max(1, int(topk))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse_logits = self.lda_model(x)
        k = min(self.topk, coarse_logits.size(1))
        selected = coarse_logits.topk(k, dim=1).indices

        if hasattr(self.rgda_model, "forward_selected"):
            selected_logits = self.rgda_model.forward_selected(x, selected)
        else:
            selected_logits = self.rgda_model(x).gather(1, selected)

        reranked = torch.full_like(coarse_logits, float("-inf"))
        reranked.scatter_(1, selected, selected_logits)
        return reranked

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor):
        return torch.softmax(self.forward(x), dim=1)


class LDATopKLRRGDARerankBuilder(BaseClassifierBuilder):
    """Build the appendix-style LDA top-k + LR-RGDA reranking classifier."""

    def __init__(
        self,
        lda_reg_alpha=0.3,
        topk=50,
        rgda_alpha1=0.2,
        rgda_alpha2=0.2,
        rgda_alpha3=0.2,
        low_rank=True,
        rank=64,
        device="cuda",
        num_centers=1,
        train_iter=0,
        fit_lr=0.01,
        fit_samples_per_class=0,
        fit_sample_mode="mean",
        fit_seed=42,
        fit_verbose=True,
    ):
        self.lda_reg_alpha = lda_reg_alpha
        self.topk = max(1, int(topk))
        self.rgda_builder = LRRGDAClassifierBuilder(
            rgda_alpha1=rgda_alpha1,
            rgda_alpha2=rgda_alpha2,
            rgda_alpha3=rgda_alpha3,
            low_rank=low_rank,
            rank=rank,
            device=device,
            num_centers=num_centers,
            train_iter=train_iter,
            fit_lr=fit_lr,
            fit_samples_per_class=fit_samples_per_class,
            fit_sample_mode=fit_sample_mode,
            fit_seed=fit_seed,
            fit_verbose=fit_verbose,
        )
        self.device = device

    def build(self, stats_dict):
        start_time = time.time()
        priors = {cid: 1.0 / len(stats_dict) for cid in stats_dict}
        lda_model = LinearLDAClassifier(
            stats_dict=stats_dict,
            class_priors=priors,
            lda_reg_alpha=self.lda_reg_alpha,
            device=self.device,
        ).to(self.device)
        rgda_model = self.rgda_builder.build(stats_dict).to(self.device)
        model = LDATopKLRRGDARerankClassifier(
            lda_model=lda_model,
            rgda_model=rgda_model,
            topk=self.topk,
        ).to(self.device)
        end_time = time.time()
        log_time_usage("LDA top-k + LR-RGDA rerank classifier build", start_time, end_time)
        return model


class QDAClassifierBuilder(LRRGDAClassifierBuilder):
    """Backward-compatible alias for the historical `qda` interface."""

    pass
