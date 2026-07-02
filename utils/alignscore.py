import numpy as np
from lm_polygraph_lite.generation_metrics.alignscore_utils import AlignScorer

import torch
from typing import List, Dict
from lm_polygraph_lite.generation_metrics.generation_metric import GenerationMetric


class AlignScore(GenerationMetric):
    """
    Calculates AlignScore metric (https://aclanthology.org/2023.acl-long.634/)
    between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        lang="en",
        ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
        batch_size=16,
        return_mean=False,
        return_inverse=False,
    ):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = AlignScorer(
            model="roberta-large",
            batch_size=batch_size,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
            verbose=False,
        )
        self.return_mean = return_mean
        self.return_inverse = return_inverse

    def __str__(self):
        if self.return_mean:
            return "AlignScoreMean"
        if self.return_inverse:
            return "AlignScoreInv"
        return "AlignScore"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """Returns AlignScore between stats['greedy_texts'] and target_texts."""

        greedy_texts = stats["greedy_texts"]

        # Expand multiref examples: each (output, single_reference) pair becomes one row.
        exp_targets: List[str] = []
        exp_outputs: List[str] = []
        example_indices: List[int] = []
        for i, (tgt, out) in enumerate(zip(target_texts, greedy_texts)):
            out_filt = out if len(out.strip()) else "(empty)"
            refs = tgt if isinstance(tgt, list) else [tgt]
            for r in refs:
                exp_targets.append(r if len(r.strip()) else "(empty)")
                exp_outputs.append(out_filt)
                example_indices.append(i)

        exp_scores = np.array(
            self.scorer.score(claims=exp_targets, contexts=exp_outputs)
        )
        if self.return_mean or self.return_inverse:
            exp_scores_ = np.array(
                self.scorer.score(claims=exp_outputs, contexts=exp_targets)
            )

        def _aggregate(values: np.ndarray) -> np.ndarray:
            agg = np.full(len(target_texts), -np.inf)
            for idx, v in zip(example_indices, values):
                if v > agg[idx]:
                    agg[idx] = v
            return agg

        if self.return_mean:
            return _aggregate((exp_scores + exp_scores_) / 2)
        if self.return_inverse:
            return _aggregate(exp_scores_)
        return _aggregate(exp_scores)
