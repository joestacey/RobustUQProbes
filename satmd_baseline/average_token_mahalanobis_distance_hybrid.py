import numpy as np

from typing import Dict, List

from lm_polygraph_lite.estimators.estimator import Estimator
from lm_polygraph_lite.generation_metrics.aggregated_metric import AggregatedMetric

from lm_polygraph_lite.estimators.max_probability import MaximumSequenceProbability
from lm_polygraph_lite.stat_calculators.entropy import EntropyCalculator

from .token_mahalanobis_distance import TokenMahalanobisDistance
from .relative_token_mahalanobis_distance import RelativeTokenMahalanobisDistance

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn.decomposition import PCA


class LinRegTokenMahalanobisDistance_Hybrid(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        hidden_layers: List[int] = [0, -1],
        metric = None,
        metric_name: str = "",
        metric_router = None,

        metric_md = None,
        metric_md_name: str = "",

        aggregated: bool = False,
        positive: bool = True,
        ue: str = "TokenMahalanobis",

        meta_model: str = "LinReg",
        norm: str = "norm",

        tgt_norm: bool = False,
        remove_corr: bool = False,
        remove_alg: int = 2,

        device: str = "cuda",
        storage_device: str = "cuda",
    
        sim_pca: bool = False,
        n_components: int = 10,
    ):
        self.ue = ue
        self.hidden_layers = hidden_layers
        self.device = device
        self.storage_device = storage_device
        self.tmds = {}

        self.n_components = n_components
        self.sim_pca = sim_pca
        self.sim_pca_name = f", sim_pca" if self.sim_pca else ""

        train_greedy_tokens = f"train_greedy_tokens"
        dependencies = [train_greedy_tokens, "train_target_texts"]

        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += [f"token_embeddings", f"train_token_embeddings"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings", f"background_train_token_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings_{layer}"]
            if ue == "TokenMahalanobis":
                self.tmds[layer] = TokenMahalanobisDistance(
                    embeddings_type, normalize=False, metric_thr=metric_thr, metric=metric_md, metric_name=metric_md_name, metric_router=metric_router, aggregation="none", hidden_layer=layer, aggregated=aggregated, device=self.device, storage_device=self.storage_device,
                )
            elif ue == "RelativeTokenMahalanobis":
                self.tmds[layer] = RelativeTokenMahalanobisDistance(
                    embeddings_type, normalize=False, metric_thr=metric_thr, metric=metric_md, metric_name=metric_md_name, metric_router=metric_router, aggregation="none", hidden_layer=layer, aggregated=aggregated, device=self.device, storage_device=self.storage_device,
                )
        super().__init__(dependencies, "sequence")
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregated=aggregated
        self.metric_router = metric_router
        if metric is not None:
            self.metric = metric
            if aggregated:
                self.metric = AggregatedMetric(base_metric=self.metric)
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.metric_md_name = metric_md_name
        self.embeddings_type=embeddings_type
        self.positive=positive
        self.meta_model=meta_model
        self.remove_corr=remove_corr
        self.remove_alg=remove_alg
        self.msp = MaximumSequenceProbability()
        self.ent = EntropyCalculator()

    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        positive = "pos" if self.positive else ""
        remove_corr = f"remove_corr_{self.remove_alg}_{self.n_components}_comp" if self.remove_corr else ""
        return f"Hybrid{self.meta_model}{self.ue}Distance_{self.embeddings_type}{hidden_layers} ({self.aggregation}, {self.metric_name}, {self.metric_md_name}, {self.metric_thr}, {positive}, {remove_corr}{self.sim_pca_name})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        
        if not self.is_fitted: 
            train_greedy_texts = stats[f"train_greedy_texts"]
            train_greedy_tokens = stats[f"train_greedy_tokens"]
            train_target_texts = stats[f"train_target_texts"]
            train_greedy_log_probs = stats[f"train_greedy_log_probs"]
            train_greedy_log_likelihoods = stats[f"train_greedy_log_likelihoods"]

            train_source_ids = stats["train_source_ids"] if self.metric_router is not None else [None] * len(train_greedy_texts)
            metric_key = f"train_seq_{self.metric_name}_{len(train_greedy_texts)}"
            if metric_key in stats.keys():
                self.train_seq_metrics = stats[metric_key]
            else:
                metrics = []
                for x, y, x_t, src in zip(train_greedy_texts, train_target_texts, train_greedy_tokens, train_source_ids):
                    metric = self.metric_router(src) if self.metric_router is not None else self.metric

                    if isinstance(y, list) and (not self.aggregated):
                        y_ = y[0]
                    elif isinstance(y, str) and (self.aggregated):
                        y_ = [y]
                    else:
                        y_ = y
                    metrics.append(metric({"greedy_texts": [x], "target_texts": [y_]}, [y_])[0])
                self.train_seq_metrics = np.array(metrics)
                stats[metric_key] = self.train_seq_metrics

            train_mds = []
            dev_size = 0.5 
            train_idx, dev_idx = train_test_split(list(range(len(train_greedy_texts))), test_size=dev_size, shuffle=True, random_state=42)
            lens = np.array([0]+[len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            token_train_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in train_idx])
            token_dev_idx = np.concatenate([np.arange(tokens_before[i], tokens_before[i+1]) for i in dev_idx])

            for layer in tqdm(self.hidden_layers):
                if layer == -1:
                    train_token_embeddings = stats[f"train_token_embeddings_{self.embeddings_type}"]
                    train_stats = {f"train_tokens": [train_greedy_tokens[k] for k in train_idx],
                                   f"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx],
                                   "train_greedy_texts":[train_greedy_texts[k] for k in train_idx],
                                   f"tokens": [train_greedy_tokens[k] for k in dev_idx],
                                   f"greedy_tokens": [train_greedy_tokens[k] for k in dev_idx],
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   "train_source_ids": [train_source_ids[k] for k in train_idx],
                                   f"train_token_embeddings_{self.embeddings_type}": [train_token_embeddings[k] for k in token_train_idx],
                                   f"token_embeddings_{self.embeddings_type}": [train_token_embeddings[k] for k in token_dev_idx],
                                  }
                    if "relative" in self.ue.lower(): 
                        train_stats[f"background_train_token_embeddings_{self.embeddings_type}"] = stats[f"background_train_token_embeddings_{self.embeddings_type}"]
                else:
                    train_token_embeddings = stats[f"train_token_embeddings_{self.embeddings_type}_{layer}"]
                    train_stats = {f"train_tokens": [train_greedy_tokens[k] for k in train_idx],
                                   f"train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx],
                                   "train_greedy_texts": [train_greedy_texts[k] for k in train_idx],
                                   f"tokens": [train_greedy_tokens[k] for k in dev_idx],
                                   f"greedy_tokens": [train_greedy_tokens[k] for k in dev_idx],
                                   "train_target_texts": [train_target_texts[k] for k in train_idx],
                                   "train_source_ids": [train_source_ids[k] for k in train_idx],
                                   f"train_token_embeddings_{self.embeddings_type}_{layer}": [train_token_embeddings[k] for k in token_train_idx],
                                   f"token_embeddings_{self.embeddings_type}_{layer}": [train_token_embeddings[k] for k in token_dev_idx],
                                  }
                    if "relative" in self.ue.lower(): 
                        train_stats[f"background_train_token_embeddings_{self.embeddings_type}_{layer}"] = stats[f"background_train_token_embeddings_{self.embeddings_type}_{layer}"]
                    
                metric_key = f"train_{self.metric_md_name}_{len(train_greedy_texts)}"
                if metric_key in stats.keys():
                    train_stats[f"train_{self.metric_md_name}_{len(train_idx)}"] = stats[metric_key][token_train_idx]

                if layer == -1:
                    hidden_layer = ""
                else:
                    hidden_layer = f"_{layer}"
                    
                centroid_key_ = f"centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"
                covariance_key_ = f"covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"

                background_centroid_key_ = f"background_centroid{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"
                background_covariance_key_ = f"background_covariance{hidden_layer}_{self.metric_name}_{self.metric_thr}_{len(train_idx)}"

                if centroid_key_ in stats.keys():
                    train_stats[centroid_key_] = stats[centroid_key_]
                if covariance_key_ in stats.keys():
                    train_stats[covariance_key_] = stats[covariance_key_]
                if background_centroid_key_ in stats.keys():
                    train_stats[background_centroid_key_] = stats[background_centroid_key_]
                if background_covariance_key_ in stats.keys():
                    train_stats[background_covariance_key_] = stats[background_covariance_key_]
                
                md = self.tmds[layer](train_stats, save_data=False).reshape(-1)

                if "Relative" in self.ue:
                    if centroid_key_ not in stats.keys():
                        stats[centroid_key_] = self.tmds[layer].MD.centroid
                    if covariance_key_ not in stats.keys():
                        stats[covariance_key_] = self.tmds[layer].MD.sigma_inv
                    if background_centroid_key_ not in stats.keys():
                        stats[background_centroid_key_] = self.tmds[layer].centroid_0
                    if background_covariance_key_ not in stats.keys():
                        stats[background_covariance_key_] = self.tmds[layer].sigma_inv_0
                else:
                    if centroid_key_ not in stats.keys():
                        stats[centroid_key_] = self.tmds[layer].centroid
                    if covariance_key_ not in stats.keys():
                        stats[covariance_key_] = self.tmds[layer].sigma_inv

                self.tmds[layer].is_fitted = False
                k = 0
                mean_md = []
                for tokens in [train_greedy_tokens[k] for k in dev_idx]:
                    dists_i = md[k:k+len(tokens)]
                    k += len(tokens)
                    mean_md.append(np.mean(dists_i))
                train_mds.append(mean_md)
            train_dists = np.array(train_mds).T
            train_dists[np.isnan(train_dists)] = 0
            self.regressor = Ridge(positive=self.positive)

            X = train_dists

            if self.remove_corr:
                self.pca = PCA(n_components=self.n_components)
                X = self.pca.fit_transform(X)

            msp = np.array(self.msp({"greedy_log_likelihoods": [train_greedy_log_likelihoods[i] for i in dev_idx]}))
            ent = np.array([np.mean(x) for x in self.ent({"greedy_log_probs": [train_greedy_log_probs[i] for i in dev_idx]})["entropy"]])
            X = np.hstack([X, msp.reshape(-1, 1), ent.reshape(-1, 1)])

            target = self.train_seq_metrics[dev_idx]
            target[np.isnan(target)] = 0
            y = 1 - target

            self.regressor.fit(X, y)
            self.is_fitted = True


        eval_mds = []
        greedy_tokens = stats[f"greedy_tokens"]
        for layer in self.tmds.keys():
            md = self.tmds[layer](stats).reshape(-1)
            k = 0
            mean_md = []
            for tokens in greedy_tokens:
                dists_i = md[k:k+len(tokens)]
                k += len(tokens)
                mean_md.append(np.mean(dists_i))
            eval_mds.append(mean_md)
        eval_dists = np.array(eval_mds).T
        eval_dists[np.isnan(eval_dists)] = 0

        if self.remove_corr:
            eval_dists = self.pca.transform(eval_dists)

        msp = np.array(self.msp({"greedy_log_likelihoods": stats["greedy_log_likelihoods"]}))
        ent = np.array([np.mean(x) for x in self.ent({"greedy_log_probs": stats["greedy_log_probs"]})["entropy"]])

        eval_dists = np.hstack([eval_dists, msp.reshape(-1, 1), ent.reshape(-1, 1)])

        ues = self.regressor.predict(eval_dists)
        return ues

