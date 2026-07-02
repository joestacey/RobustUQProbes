import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph_lite.utils.model import WhiteboxModel

class AttentionMapCalculator(StatCalculator):
    def __init__(self, stage: str = "train"):
        
        self.stage = stage
        self.consider_token_outputs_independently = False

        if stage == "train":
            self.stage += "_"

        deps = ['greedy_tokens']

        if stage == "train":
            stats = ['train_stats_full_att_map']
        else:
            stats = ['stats_full_att_map']
        
        super().__init__(
            stats,
            deps,
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:

        batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        assert batch['input_ids'].shape[0] == 1
        assert len(dependencies['greedy_tokens']) == 1

        expected_output = torch.cat(
                [batch['input_ids'].squeeze(), torch.tensor(dependencies['greedy_tokens'][0])]).unsqueeze(0)

        batch = {k: v.to(model.device()) for k, v in batch.items()}

        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=True,
                output_hidden_states=False,
                num_return_sequences=1,
                do_sample=False
            )

        assert torch.equal(out.sequences.cpu().detach(), expected_output), str(out.sequences) + " " + str(expected_output)

        results = {
                "full_attentions": out.attentions
                }

        return results

