import random
import torch
import numpy as np
import logging

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph_lite.utils.model import WhiteboxModel

from luh.utils import load_feature_extractor

log = logging.getLogger(__name__)


class CompiledFeatures(StatCalculator):
    def __init__(self, stage: str = "train"):

        self.stage = stage
        self.consider_token_outputs_independently = False

        self.features_to_dependencies = {
                "luh.feature_extractors.basic_hidden_states": ["hidden_states"],
                "luh.feature_extractors.basic_attention": ["attentions"],
                "luh.feature_extractors.token_probabilities": ["scores"],
                "luh.feature_extractors.lookback_lens": ["attentions"],
                }

        if stage == "train":
            self.stage += "_"

        deps = ['greedy_tokens', 'input_tokens', 'input_texts', 'greedy_texts']

        if stage == "train":
            stats = ['train_stats_compiled_probe_features']
        else:
            stats = ['stats_compiled_probe_features']

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
        feature_extractors=None,
        head_cfg=None
    ) -> Dict[str, np.ndarray]:

        batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        assert batch['input_ids'].shape[0] == 1
        assert len(dependencies['greedy_tokens']) == 1

        greedy_tokens = dependencies['greedy_tokens']

        expected_output = torch.cat(
                [batch['input_ids'].squeeze(), torch.tensor(greedy_tokens[0])]).unsqueeze(0)

        batch = {k: v.to(model.device()) for k, v in batch.items()}

        features_required = []
        for fe in feature_extractors:
            assert fe['name'] in self.features_to_dependencies, str(fe)
            for dep in self.features_to_dependencies[fe['name']]:
                if dep not in features_required:
                    features_required.append(dep)

        for feature in features_required:
            assert feature in ['attentions', 'hidden_states', 'scores'], str(feature)

        feature_extractor = load_feature_extractor(feature_extractors, model.model)

        require_attention = 'attentions' in features_required
        require_hidden_states = 'hidden_states' in features_required
        require_scores = 'scores' in features_required

        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=require_scores,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=require_attention,
                output_hidden_states=require_hidden_states,
                num_return_sequences=1,
                do_sample=False
            )

        assert not model.model.training
        assert len(texts) == 1
        assert len(model.tokenizer.batch_decode(greedy_tokens)) == 1

        if not torch.equal(out.sequences.cpu().detach(), expected_output):
            log.warning("Compiled features sequence mismatch: %s vs %s", out.sequences, expected_output)

        assert len(greedy_tokens) == 1

        input_tokens = dependencies['input_tokens']
        device = model.device()

        assert len(input_tokens) == 1

        llm_inputs = {}
        llm_outputs = {}

        full_tokens = [torch.tensor(input_tokens[0] + greedy_tokens[0]).to(device) for i in range(len(greedy_tokens))]
        output_mask = torch.zeros(len(full_tokens[0]))
        output_mask[len(input_tokens[0]):] = 1

        if head_cfg['token_aggregation'] == 'average_all':
            output_mask[:] = 1
        elif head_cfg['token_aggregation'] == 'context_average':
            output_mask = torch.zeros(len(full_tokens[0]))
            output_mask[:len(input_tokens[0])] = 1
        elif 'train' not in self.stage and head_cfg['token_aggregation'] == 'ablation_random_keep':
            output_mask = torch.zeros(len(full_tokens[0]))
            random_idx = random.choice(range(len(greedy_tokens[0])))
            output_mask[len(input_tokens[0]) + random_idx] = 1

        output_mask = output_mask[1:].to(device).to(dtype=torch.long)
        llm_inputs['output_mask'] = output_mask.unsqueeze(0)

        prompt_attention = torch.ones(len(input_tokens[0])).to(device).to(dtype=torch.long)
        llm_inputs['attention_mask'] = prompt_attention.unsqueeze(0)

        assert len(full_tokens) == 1

        att_mask = torch.ones(len(full_tokens[0])).to(device).to(dtype=torch.long)
        llm_outputs['full_attention_mask'] = att_mask.unsqueeze(0)

        context_length = torch.tensor(len(input_tokens[0])).to(device)
        llm_outputs['context_lengths'] = context_length.unsqueeze(0)
        llm_outputs['sequences'] = full_tokens[0].unsqueeze(0)

        if 'attentions' in features_required:
            llm_outputs['attentions'] = out.attentions
        if 'hidden_states' in features_required:
            llm_outputs['hidden_states'] = out.hidden_states
        if 'scores' in features_required:
            llm_outputs['scores'] = out.scores

        features = feature_extractor(llm_inputs, llm_outputs)

        output_mask = llm_inputs["output_mask"][0].unsqueeze(dim=0)
        sum_entity_embeds = (features[0, :] * output_mask.unsqueeze(-1)).sum(dim=1)
        count_entity_tokens = output_mask.sum(dim=1).clamp(min=1)
        entity_representation = sum_entity_embeds / count_entity_tokens.unsqueeze(-1)

        if head_cfg['token_aggregation'] == 'last_token':
            entity_representation = features[0, -1, :].unsqueeze(0)
        elif head_cfg['token_aggregation'] == 'first_output_token':
            entity_representation = features[0, len(input_tokens[0]), :].unsqueeze(0)
        elif head_cfg['token_aggregation'] == 'last_context_token':
            entity_representation = features[0, len(input_tokens[0]) - 1, :].unsqueeze(0)

        entity_representation = entity_representation.to('cpu')

        return {"features": entity_representation}
