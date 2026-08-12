import numpy as np

from typing import Dict

from lm_polygraph_lite.estimators.estimator import Estimator
from lm_polygraph_lite.generation_metrics.aggregated_metric import AggregatedMetric

from lm_polygraph_lite.estimators.mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    create_cuda_tensor_from_numpy,
)


class TokenMahalanobisDistance(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        hidden_layer: int = -1,
        metric = None,
        metric_name: str = "",
        aggregated: bool = False,
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        self.hidden_layer = hidden_layer
        train_greedy_tokens = f"train_greedy_tokens"
        if self.hidden_layer == -1:
            super().__init__([f"token_embeddings", f"train_token_embeddings", train_greedy_tokens, "train_target_texts"], "sequence")
        else:
            super().__init__([f"token_embeddings_{self.hidden_layer}", f"train_token_embeddings_{self.hidden_layer}", train_greedy_tokens, "train_target_texts"], "sequence")
        self.centroid = None
        self.sigma_inv = None
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.device = device
        self.storage_device = storage_device
        self.aggregated = aggregated
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)

    def __str__(self):
        hidden_layer = "" if self.hidden_layer==-1 else f"_{self.hidden_layer}"
        return f"TokenMahalanobisDistance_{self.embeddings_type}{hidden_layer} ({self.aggregation}, {self.metric_name}, {self.metric_thr})"

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
        if self.hidden_layer == -1:
            hidden_layer = ""
        else:
            hidden_layer = f"_{self.hidden_layer}"
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"token_embeddings_{self.embeddings_type}{hidden_layer}"]
        )

        # compute centroids if not given
        if not self.is_fitted:
            train_greedy_texts = stats[f"train_greedy_texts"]
            centroid_key = f"centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if centroid_key in stats.keys():  # shared cache across estimators using the same training data
                self.centroid = stats[centroid_key]
                if self.storage_device == "cpu":
                    self.centroid = self.centroid.cpu()
                elif self.storage_device == "cuda":
                    self.centroid = self.centroid.cuda()
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                if self.metric_thr > 0:
                    train_greedy_tokens = stats[f"train_greedy_tokens"]
                    train_target_texts = stats[f"train_target_texts"]
                    
                    metric_key = f"train_{self.metric_name}_{len(train_greedy_texts)}"

                    if metric_key in stats.keys():
                        self.train_token_metrics = stats[metric_key]
                    else:
                        metrics = []

                        for x, y, x_t in zip(train_greedy_texts, train_target_texts, train_greedy_tokens):

                            if isinstance(y, list) and (not self.aggregated):
                                y_ = y[0]
                            elif isinstance(y, str) and (self.aggregated):
                                y_ = [y]
                            else:
                                y_ = y

                            metrics.append([self.metric({"greedy_texts": [x], "target_texts": [y_]}, [y_])[0]] * len(x_t))

                        self.train_token_metrics = np.concatenate(metrics)
                        stats[metric_key] = self.train_token_metrics

                    if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]

                self.centroid = train_embeddings.mean(axis=0)
                
                if self.storage_device == "cpu":
                    self.centroid = self.centroid.cpu()
                if save_data:
                    stats[centroid_key] = self.centroid

        if not self.is_fitted:
            covariance_key = f"covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_greedy_texts)}"
            if covariance_key in stats.keys():  # shared cache across estimators using the same training data
                self.sigma_inv = stats[covariance_key]
                if self.storage_device == "cpu":
                    self.sigma_inv = self.sigma_inv.cpu()
                elif self.storage_device == "cuda":
                    self.sigma_inv = self.sigma_inv.cuda()
            else:
                train_embeddings = create_cuda_tensor_from_numpy(
                    stats[f"train_token_embeddings_{self.embeddings_type}{hidden_layer}"]
                )
                if self.metric_thr > 0:
                    if (self.train_token_metrics >= self.metric_thr).sum() > 10:
                        train_embeddings = train_embeddings[self.train_token_metrics >= self.metric_thr]
                self.sigma_inv, _ = compute_inv_covariance(
                    self.centroid.unsqueeze(0), train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv = self.sigma_inv.cpu()

                if save_data:
                    stats[covariance_key] = self.sigma_inv
            self.is_fitted = True

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # CPU is faster for small batches, avoiding GPU transfer overhead
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.float(),
                    None,
                    self.sigma_inv.float(),
                    embeddings.cpu().float(),
                )[:, 0]
            else:
                dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                    self.centroid.cuda().float(),
                    None,
                    self.sigma_inv.cuda().float(),
                    embeddings.float(),
                )[:, 0]
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists = mahalanobis_distance_with_known_centroids_sigma_inv(
                self.centroid.float(),
                None,
                self.sigma_inv.float(),
                embeddings.float(),
            )[:, 0]
        else:
            raise NotImplementedError
        
        k = 0
        agg_dists = []
        greedy_tokens = stats[f"greedy_tokens"]
        
        for tokens in greedy_tokens:
            dists_i = dists[k:k+len(tokens)].cpu().detach().numpy()
            k += len(tokens)
            if self.aggregation == "mean":
                agg_dists.append(np.mean(dists_i))
            elif self.aggregation == "sum":
                agg_dists.append(np.sum(dists_i))
        if self.aggregation == "none":
            agg_dists = dists.cpu().detach().numpy()
        else:
            agg_dists = np.array(agg_dists)
    
        if self.max < agg_dists.max():
            self.max = agg_dists.max()
        if self.min > agg_dists.min():
            self.min = agg_dists.min()

        if self.normalize:
            agg_dists = np.clip((self.max - agg_dists) / (self.max - self.min), 0, 1)

        return agg_dists
