from typing import Optional

import torch

def cholesky_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    if matrix.dim() == 3:
        batch_size, n, _ = matrix.shape
        reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        reg_eye = reg_eye.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.linalg.cholesky(matrix + reg_eye)
    
    elif matrix.dim() == 2:
        reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return torch.linalg.cholesky(matrix + reg_eye)
    else:
        raise ValueError(f"不支持的矩阵维度: {matrix.dim()}。支持2D或3D张量。")

def cholesky_manual_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    """保留原始手动实现以供参考（已被优化的PyTorch版本替代）"""
    n = matrix.size(0)
    L = torch.zeros_like(matrix)
    reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + reg_eye

    for j in range(n):
        s_diag = torch.sum(L[j, :j] ** 2, dim=0)
        diag = matrix[j, j] - s_diag
        L[j, j] = torch.sqrt(torch.clamp(diag, min=1e-8))

        if j < n - 1:
            s_off = L[j + 1:, :j] @ L[j, :j]
            L[j + 1:, j] = (matrix[j + 1:, j] - s_off) / L[j, j]
    return L

def cholesky_stable_with_fallback(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    try:
        return cholesky_stable(matrix, reg)
    
    except RuntimeError as e:
        return cholesky_manual_stable(matrix, reg)

def kmeans_centers(x: torch.Tensor, num_centers: int, n_iter: int = 20, seed: int = 42) -> torch.Tensor:
    """Return a fixed number of per-class centers for compact multi-centroid RGDA."""
    unique_x = torch.unique(x, dim=0)
    if unique_x.size(0) <= num_centers:
        if unique_x.size(0) == num_centers:
            return unique_x.clone()
        repeat = (num_centers + unique_x.size(0) - 1) // unique_x.size(0)
        return unique_x.repeat((repeat, 1))[:num_centers].clone()

    generator = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.size(0), generator=generator, device=x.device)[:num_centers]
    centers = x[idx].clone()

    for _ in range(n_iter):
        dists = torch.cdist(x, centers)
        labels = dists.argmin(dim=1)
        new_centers = centers.clone()
        for center_idx in range(num_centers):
            mask = labels == center_idx
            if mask.any():
                new_centers[center_idx] = x[mask].mean(dim=0)
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return centers


class GaussianStatistics:
    """Container for per-class Gaussian statistics."""
    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        reg: float = 1e-4,
        cholesky = False,
        centers: Optional[torch.Tensor] = None,
        gmm_means: Optional[torch.Tensor] = None,
        gmm_diag_vars: Optional[torch.Tensor] = None,
        gmm_weights: Optional[torch.Tensor] = None,
    ):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        if mean.dim() != 1:
            raise AssertionError("GaussianStatistics.mean 必须是 1D 向量")

        self.mean = mean
        self.cov = cov
        self.reg = reg
        self.centers = centers
        self.gmm_means = gmm_means
        self.gmm_diag_vars = gmm_diag_vars
        self.gmm_weights = gmm_weights

        if cholesky:
            self.L = cholesky_stable_with_fallback(cov, reg=reg)
        else:
            self.L = None

    def to(self, device):
        """Move statistics to the requested device."""

        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        if self.centers is not None:
            self.centers = self.centers.to(device)
        if self.gmm_means is not None:
            self.gmm_means = self.gmm_means.to(device)
        if self.gmm_diag_vars is not None:
            self.gmm_diag_vars = self.gmm_diag_vars.to(device)
        if self.gmm_weights is not None:
            self.gmm_weights = self.gmm_weights.to(device)
        if self.L is not None:
            self.L = self.L.to(device)
        return self

    def sample(
        self,
        n_samples = None,
        cached_eps = None,
    ) -> torch.Tensor:
        """Draw samples from the Gaussian distribution."""

        if self.L is None:
            self.L = cholesky_stable_with_fallback(self.cov, reg=self.reg)

        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            if n_samples is None:
                raise ValueError("n_samples 必须在未提供 cached_eps 时给定")
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        samples = self.mean.unsqueeze(0) + eps @ self.L.t()
        return samples

    def sample_gmm(
        self,
        n_samples: int,
        mode: str = "mean",
        seed: Optional[int] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Draw compact replay samples from stored diagonal-GMM statistics."""
        if self.gmm_means is None or self.gmm_weights is None:
            return self.sample(n_samples=n_samples)

        mode = str(mode).lower()
        if mode not in {"mean", "sample"}:
            raise ValueError(f"Unsupported GMM sample mode: {mode}")

        device = self.mean.device
        means = self.gmm_means.to(device)
        weights = self.gmm_weights.to(device).float()
        weights = weights / weights.sum().clamp(min=1e-12)
        k = means.size(0)
        if k == 0:
            raise ValueError("Stored GMM has zero components")

        # Match project_clip_continual_learning/main_joint.py: allocate by
        # rounded mixture weights, keep every component represented, and put the
        # residual on the largest-weight component.
        counts = torch.clamp(torch.round(weights * int(n_samples)).long(), min=1)
        diff = int(n_samples - counts.sum().item())
        max_idx = int(torch.argmax(weights).item())
        counts[max_idx] += diff
        if counts[max_idx] < 0:
            raise ValueError(
                "GMM replay count allocation became negative; increase n_samples "
                "or reduce the number of GMM components."
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        chunks = []
        diag_vars = None
        if self.gmm_diag_vars is not None:
            diag_vars = self.gmm_diag_vars.to(device).clamp(min=1e-8)

        for comp_idx in range(k):
            count = int(counts[comp_idx].item())
            if count <= 0:
                continue
            mean = means[comp_idx].unsqueeze(0)
            if mode == "mean" or diag_vars is None:
                samples = mean.repeat(count, 1)
            else:
                std = torch.sqrt(diag_vars[comp_idx]).unsqueeze(0)
                noise = torch.randn(
                    count,
                    mean.size(1),
                    device=device,
                    generator=generator,
                )
                samples = mean + noise * std
            if normalize:
                samples = torch.nn.functional.normalize(samples, dim=-1)
            chunks.append(samples)

        if not chunks:
            raise ValueError("GMM replay generated no samples")
        return torch.cat(chunks, dim=0)


def make_gaussian_statistics_like(
    stat: GaussianStatistics,
    mean: torch.Tensor,
    cov: torch.Tensor,
    centers: Optional[torch.Tensor] = None,
    gmm_means: Optional[torch.Tensor] = None,
    gmm_diag_vars: Optional[torch.Tensor] = None,
) -> GaussianStatistics:
    """Create a transformed statistic while preserving optional compact replay fields."""
    return GaussianStatistics(
        mean,
        cov,
        stat.reg,
        centers=centers,
        gmm_means=gmm_means,
        gmm_diag_vars=(
            gmm_diag_vars
            if gmm_diag_vars is not None
            else getattr(stat, "gmm_diag_vars", None)
        ),
        gmm_weights=getattr(stat, "gmm_weights", None),
    )


def fit_diag_gmm_statistics(
    x: torch.Tensor,
    num_components: int,
    seed: int = 42,
    reg: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate compact diagonal-GMM statistics with deterministic torch k-means.

    sklearn's diagonal GaussianMixture can be very slow or unstable for
    high-dimensional frozen-backbone features. For LR-RGDA replay we only need a
    compact multi-modal summary, so k-means component assignments plus per
    component diagonal variances are the pragmatic default.
    """
    x = x.detach().float().cpu()
    actual_k = max(1, min(int(num_components), x.size(0)))

    if actual_k == 1:
        var = x.var(dim=0, unbiased=False).clamp(min=reg)
        return x.mean(dim=0, keepdim=True), var.unsqueeze(0), torch.ones(1)

    centers = kmeans_centers(x, actual_k, seed=seed)
    labels = torch.cdist(x, centers).argmin(dim=1)
    diag_vars = []
    weights = []
    for comp_idx in range(actual_k):
        mask = labels == comp_idx
        feats = x[mask] if mask.any() else centers[comp_idx].unsqueeze(0)
        diag_vars.append(feats.var(dim=0, unbiased=False).clamp(min=reg))
        weights.append(float(feats.size(0)) / float(x.size(0)))
    weights_t = torch.tensor(weights).float()
    weights_t = weights_t / weights_t.sum().clamp(min=1e-12)
    return centers, torch.stack(diag_vars), weights_t


def fit_spherical_gmm_statistics(
    x: torch.Tensor,
    num_components: int,
    seed: int = 42,
    reg: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit sklearn spherical GMM statistics following project_clip main_joint.py.

    This stores one scalar variance per component. ``GaussianStatistics.sample_gmm``
    broadcasts that scalar over the feature dimension during stochastic replay.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "rgda_gmm_backend=sklearn_spherical requires scikit-learn. "
            "Install scikit-learn or use --rgda_gmm_backend kmeans_diag."
        ) from exc

    x_cpu = x.detach().float().cpu()
    actual_k = max(1, min(int(num_components), x_cpu.size(0)))
    if actual_k == 1:
        var = x_cpu.var(dim=0, unbiased=False).mean().clamp(min=reg)
        return x_cpu.mean(dim=0, keepdim=True), var.unsqueeze(0), torch.ones(1)

    gmm = GaussianMixture(
        n_components=actual_k,
        covariance_type="spherical",
        random_state=int(seed),
        reg_covar=float(reg),
    )
    gmm.fit(x_cpu.numpy())

    means = torch.from_numpy(gmm.means_).float()
    variances = torch.from_numpy(gmm.covariances_).float().clamp(min=reg)
    weights = torch.from_numpy(gmm.weights_).float()
    weights = weights / weights.sum().clamp(min=1e-12)
    return means, variances, weights

class LowRankGaussianStatistics:
    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        rank: int = 512,
        reg: float = 1e-8, 
        device=None
    ):
        if mean.dim() != 1:
            raise ValueError("mean must be a 1D vector")
        
        d = mean.size(0)
        if cov.shape != (d, d):
            raise ValueError("cov shape mismatch")

        self.mean = mean
        self.d = d
        self.rank = rank or min(100, d)  # default max rank=100
        self.reg = reg

        U, S, Vh = torch.svd_lowrank(cov, q=rank, niter=4, M=None)
        U = U[:, :self.rank]
        S = S[:self.rank]

        self.U = U
        self.S = torch.clamp(S, min=0.0)

        if device:
            self.to(device)
    @property
    def L(self):
        return self.U * torch.sqrt(self.S).unsqueeze(0)

    @property
    def cov(self) -> torch.Tensor:
        """Reconstruct full covariance (use sparingly for high d!)"""
        return self.L @ self.L.T



    def to(self, device):
        self.mean = self.mean.to(device)
        self.L = self.L.to(device)
        return self

    def sample(self, n_samples: int) -> torch.Tensor:
        """Efficient sampling without forming full covariance"""
        eps = torch.randn(n_samples, self.L.size(1), device=self.L.device, dtype=self.L.dtype)
        return self.mean.unsqueeze(0) + eps @ self.L.T  # (n_samples, d)
