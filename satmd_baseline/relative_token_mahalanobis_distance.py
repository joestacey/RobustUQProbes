import numpy as np

from typing import Dict

from lm_polygraph_lite.estimators.estimator import Estimator
from lm_polygraph_lite.estimators.mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    create_cuda_tensor_from_numpy,
)

from .token_mahalanobis_distance import TokenMahalanobisDistance


class RelativeTokenMahalanobisDistance(Estimator):
    """
    Relative Mahalanobis Distance (Ren et al., 2023): adjusts the per-token MD score
    by subtracting a background MD computed on a large general-purpose corpus.
    RMD(x) = MD(x) - MD_0(x)
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        metric = None,
        aggregated: bool = False,
        metric_name: str = "",
        hidden_layer: int = -1,
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        self.hidden_layer = hidden_layer
        train_greedy_tokens = f"train_greedy_tokens"
        if self.hidden_layer == -1:
            super().__init__([f"token_embeddings", f"train_token_embeddings", f"background_train_token_embeddings", train_greedy_tokens, "train_target_texts"], "sequence")
        else:
            super().__init__([f"token_embeddings_{self.hidden_layer}", f"train_token_embeddings_{self.hidden_layer}", f"background_train_token_embeddings_{self.hidden_layer}", train_greedy_tokens, "train_target_texts"], "sequence")
        self.centroid_0 = None
        self.sigma_inv_0 = None
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.metric_name = metric_name
        self.MD = TokenMahalanobisDistance(
            embeddings_type, normalize=False, metric_thr=metric_thr, metric=metric, metric_name=metric_name, aggregation="none", hidden_layer=self.hidden_layer, aggregated=aggregated, device=device, storage_device=storage_device,
        )
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.metric = metric
        self.device = device
        self.storage_device = storage_device

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"RelativeTokenMahalanobisDistance_{self.embeddings_type}{hidden_layer} ({self.aggregation}, {self.metric_name}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}{hidden_layer}"]
        )

        if not self.is_fitted:
            train_greedy_texts = stats[f"train_greedy_texts"]
            centroid_key = f"background_centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if centroid_key in stats.keys():  # shared cache across estimators using the same training data
                self.centroid_0 = stats[centroid_key]
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()
                elif self.storage_device == "cuda":
                    self.centroid_0 = self.centroid_0.cuda()
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"background_train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                self.centroid_0 = background_train_embeddings.mean(axis=0)
                
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()

                if save_data:
                    stats[centroid_key] = self.centroid_0

        if not self.is_fitted:
            covariance_key = f"background_covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if covariance_key in stats.keys():  # shared cache across estimators using the same training data
                self.sigma_inv_0 = stats[covariance_key]
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()
                elif self.storage_device == "cuda":
                    self.sigma_inv_0 = self.sigma_inv_0.cuda()
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"background_train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                self.sigma_inv_0, _ = compute_inv_covariance(
                    self.centroid_0.unsqueeze(0), background_train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()

                if save_data:
                    stats[covariance_key] = self.sigma_inv_0

            self.is_fitted = True

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # CPU is faster for small batches, avoiding GPU transfer overhead
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.cpu().float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.cuda().float(),
                        None,
                        self.sigma_inv_0.cuda().float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        else:
            raise NotImplementedError

        md = self.MD(stats, save_data=save_data)

        dists = md - dists_0
        
        agg_dists = []
        k = 0
        greedy_tokens = stats[f"greedy_tokens"]
        for tokens in greedy_tokens:
            dists_i = dists[k:k+len(tokens)]
            k += len(tokens)
            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))
            elif self.aggregation == "sum":
                agg_dists.append(np.sum(dists_i))
        if self.aggregation == "none":
            agg_dists = dists
            
        agg_dists = np.array(agg_dists)
        
        if self.max < agg_dists.max():
            self.max = agg_dists.max()
        if self.min > agg_dists.min():
            self.min = agg_dists.min()

        if self.normalize:
            agg_dists = np.clip(
                (self.max - agg_dists) / (self.max - self.min), a_min=0, a_max=1
            )

        return agg_dists
