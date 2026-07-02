import numpy as np
import torch

JITTERS = [10**exp for exp in range(-15, 0, 1)]


def compute_inv_covariance(centroids, train_features, jitters=None):
    r"""
    Computes the inverse covariance matrix required by Mahalanobis distance:
    MD = \sqrt((h(x) - \mu)^{T} \Sigma^{-1} (h(x) - \mu))
    """
    if jitters is None:
        jitters = JITTERS
    jitter = 0
    jitter_eps = None

    if torch.cuda.is_available():
        centroids = centroids.cuda()
        train_features = train_features.cuda()

    cov_scaled = torch.cov(train_features.T)

    # Increase jitter until the covariance matrix is positive semi-definite
    # (required for a valid inverse).
    for i, jitter_eps in enumerate(jitters):
        jitter = jitter_eps * torch.eye(
            cov_scaled.shape[1],
            device=cov_scaled.device,
        )
        cov_scaled_update = cov_scaled + jitter
        eigenvalues = torch.linalg.eigh(cov_scaled_update).eigenvalues
        if (eigenvalues >= 0).all():
            break
    cov_scaled = cov_scaled + jitter

    cov_inv = torch.inverse(cov_scaled.to(torch.float64)).float()
    return cov_inv, jitter_eps


def mahalanobis_distance_with_known_centroids_sigma_inv(
    centroids, centroids_mask, sigma_inv, eval_features
):
    """Returns a tensor of Mahalanobis distances between eval_features and centroids."""
    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(
        0
    )  # bs (b), num_labels (c / s), dim (d / a)

    dists = torch.sqrt(torch.einsum("bcd,da,bsa->bcs", diff, sigma_inv, diff))
    device = dists.device

    dists = torch.stack([torch.diag(dist).cpu() for dist in dists], dim=0)

    if centroids_mask is not None:
        dists = dists.masked_fill_(centroids_mask, float("inf")).to(device)
    return dists  # np.min(dists, axis=1)


def create_cuda_tensor_from_numpy(array, device="cuda"):
    if isinstance(array, list):
        array = np.stack(array)
    if not isinstance(array, torch.Tensor):
        array = torch.from_numpy(array)
    if torch.cuda.is_available() and (device == "cuda"):
        array = array.cuda()
    return array
