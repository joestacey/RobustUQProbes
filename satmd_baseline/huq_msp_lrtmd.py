import numpy as np

from typing import Dict, List

from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from lm_polygraph_lite.estimators.estimator import Estimator
from lm_polygraph_lite.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph_lite.estimators.max_probability import MaximumSequenceProbability
from .average_token_mahalanobis_distance import LinRegTokenMahalanobisDistance

prr = PredictionRejectionArea()


def total_uncertainty_linear_step(
    epistemic, aleatoric, threshold_min=0.1, threshold_max=0.9, alpha=0.1
):
    n_preds = len(aleatoric)
    n_lowest = int(n_preds * threshold_min)
    n_max = int(n_preds * threshold_max)

    aleatoric_rank = rankdata(aleatoric)
    epistemic_rank = rankdata(epistemic)

    total_rank = (1 - alpha) * epistemic_rank + alpha * aleatoric_rank

    total_rank[epistemic_rank <= n_lowest] = rankdata(
        aleatoric[epistemic_rank <= n_lowest]
    )
    total_rank[
        (aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)
    ] = aleatoric_rank[(aleatoric_rank > n_max) & (epistemic_rank <= n_lowest)]

    return total_rank


def grid_search_hp(
    epistemic,
    aleatoric,
    metrics,
    t_min_min=0.0,
    t_min_max=0.3,
    t_max_min=0.8,
    t_max_max=1.0,
    alpha_min=0.0,
    alpha_max=1.0,
    target_metric=PredictionRejectionArea()
):
    t_min_best = 0
    t_max_best = 1
    alpha_best = 0

    eps = 0.01
    best_prr = target_metric(epistemic, metrics)
    for t_min in np.arange(t_min_min, t_min_max + eps, 0.05):
        for t_max in np.arange(t_max_min, t_max_max + eps, 0.05):
            for alpha in np.arange(alpha_min, alpha_max + eps, 0.1):
                unc = total_uncertainty_linear_step(
                    epistemic, aleatoric, t_min, t_max, alpha
                )
                new_prr = target_metric(unc, metrics)
                if new_prr > best_prr:
                    best_prr = new_prr  # Fixed from new_prr = new_prr in original satmd repo
                    t_min_best = t_min
                    t_max_best = t_max
                    alpha_best = alpha

    return best_prr, t_min_best, t_max_best, alpha_best


class HUQ_LRTMD(Estimator):
    def __init__(
        self,
        embeddings_type: str = "decoder",
        normalize: bool = False,
        metric_thr: float = 0.0,
        aggregation: str = "mean",
        hidden_layers: List[int] = [0, -1],
        metric=None,
        metric_name: str = "",
        metric_md=None,
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
        self.sim_pca = sim_pca
        self.sim_pca_name = f", sim_pca" if self.sim_pca else ""
        self.n_components = n_components

        dependencies = ["train_greedy_tokens", "train_target_texts"]
        for layer in self.hidden_layers:
            if layer == -1:
                dependencies += [f"token_embeddings", f"train_token_embeddings"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings"]
            else:
                dependencies += [f"token_embeddings_{layer}", f"train_token_embeddings_{layer}"]
                if "relative" in ue.lower():
                    dependencies += [f"background_train_token_embeddings_{layer}"]

        super().__init__(dependencies, "sequence")
        self.is_fitted = False
        self.metric_thr = metric_thr
        self.aggregation = aggregation
        self.metric_name = metric_name
        self.metric_md_name = metric_md_name
        self.embeddings_type = embeddings_type
        self.positive = positive
        self.meta_model = meta_model
        self.remove_corr = remove_corr
        self.remove_alg = remove_alg
        self.md = LinRegTokenMahalanobisDistance(
            embeddings_type,
            metric=metric, metric_name=metric_name,
            metric_md=metric_md, metric_md_name=metric_md_name,
            aggregated=aggregated,
            hidden_layers=hidden_layers, metric_thr=metric_thr,
            aggregation=aggregation,
            ue=ue, positive=positive, meta_model=meta_model, norm=norm,
            n_components=n_components,
            remove_corr=remove_corr, remove_alg=remove_alg,
            device=device, storage_device=storage_device, sim_pca=sim_pca,
        )
        self.msp = MaximumSequenceProbability()

    def __str__(self):
        hidden_layers = ",".join([str(x) for x in self.hidden_layers])
        positive = "pos" if self.positive else ""
        remove_corr = f"remove_corr_{self.remove_alg}_{self.n_components}_comp" if self.remove_corr else ""
        return (
            f"HUQ-{self.meta_model}{self.ue}Distance_{self.embeddings_type}"
            f"{hidden_layers} ({self.aggregation}, {self.metric_name}, "
            f"{self.metric_md_name}, {self.metric_thr}, {positive}, "
            f"{remove_corr}{self.sim_pca_name})"
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            train_greedy_texts = stats["train_greedy_texts"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            dev_size = 0.5
            train_idx, dev_idx = train_test_split(
                list(range(len(train_greedy_texts))), test_size=dev_size, random_state=42
            )
            lens = np.array([0] + [len(tokens) for tokens in train_greedy_tokens])
            tokens_before = np.cumsum(lens)
            # token_train_idx and token_dev_idx unused here but dev_idx must match self.md's split
            _ = np.concatenate([np.arange(tokens_before[i], tokens_before[i + 1]) for i in train_idx])
            _ = np.concatenate([np.arange(tokens_before[i], tokens_before[i + 1]) for i in dev_idx])

            md_eval = self.md(stats)
            self.train_md = self.md.y_preds
            self.train_msp = np.array(
                self.msp({"greedy_log_likelihoods": [stats["train_greedy_log_likelihoods"][i] for i in dev_idx]})
            )
            metrics = self.md.train_seq_metrics[dev_idx]

            best_prr, self.t_min_best, self.t_max_best, self.alpha_best = grid_search_hp(
                self.train_md, self.train_msp, metrics, target_metric=prr
            )
            self.is_fitted = True
        else:
            md_eval = self.md(stats)

        msp_eval = np.array(self.msp({"greedy_log_likelihoods": stats["greedy_log_likelihoods"]}))
        msp_eval_plus = np.concatenate([np.array(msp_eval), self.train_msp])
        md_eval_plus = np.concatenate([np.array(md_eval), self.train_md])

        ues = total_uncertainty_linear_step(
            md_eval_plus, msp_eval_plus,
            threshold_min=self.t_min_best, threshold_max=self.t_max_best, alpha=self.alpha_best,
        )
        return ues[:len(msp_eval)]
