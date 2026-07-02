import numpy as np
import torch
from omegaconf import OmegaConf

from datasets import Dataset

from typing import Dict

from lm_polygraph_lite.estimators.estimator import Estimator

from lm_polygraph_lite.generation_metrics.aggregated_metric import AggregatedMetric
from luh.heads.utils import CausalLMWithUncertaintyLayer

from transformers import Trainer, TrainingArguments


class DataCollator():
    def __init__(self, base_model):
        self._base_model = base_model

    def __call__(self, examples):
        train_input_texts = [e["train_input_texts"] for e in examples]
        batch = self._base_model.tokenize(train_input_texts)

        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'greedy_texts': [e["train_greedy_texts"] for e in examples],
            'greedy_tokens': [e["train_greedy_tokens"] for e in examples],
            'targets': [e["targets"] for e in examples],
        }


class FeatureSupervision(Estimator):
    """Trains a luh uncertainty head against a generation metric on first call, then runs inference on subsequent calls."""

    def __init__(self,
                 device: str = "cuda",
                 storage_device: str = "cuda",
                 metric=None,
                 metric_name: str = "",
                 metric_router=None,
                 aggregated: bool = False,
                 config=None,
                 head=None,
                 pre_compile_features: bool = False,
                 base_model=None,
                 ):

        assert config is not None
        assert (metric is not None) or (metric_router is not None)

        for key in ['head_type', 'feature_extractor', 'uncertainty_head']:
            assert key in config.keys()

        self.base_model = base_model
        self.cfg = OmegaConf.create(config)
        self.device = device
        self.storage_device = storage_device
        self.head = head.to(self.device)
        self.is_fitted = False
        self.pre_compile_features = pre_compile_features

        self.metric = (AggregatedMetric(base_metric=metric) if aggregated else metric) if metric is not None else None
        self.metric_router = metric_router
        self.aggregated = aggregated
        self.metric_name = metric_name

        self.features_to_dependencies = {
            "luh.feature_extractors.basic_hidden_states": ["stats_full_hidden_states"],
            "luh.feature_extractors.basic_attention": ["stats_full_att_map"],
            "luh.feature_extractors.token_probabilities": ["stats_scores"],
            "luh.feature_extractors.lookback_lens": ["stats_full_att_map"],
        }

        dependencies = ["train_greedy_tokens", "train_target_texts", "input_tokens"]

        for feature_extractor in self.cfg.feature_extractor:
            assert feature_extractor['name'] in self.features_to_dependencies, str(feature_extractor)

            for feature_dep in self.features_to_dependencies[feature_extractor['name']]:
                if feature_dep not in dependencies:
                    dependencies.append(feature_dep)

                if self.cfg.head_type != 'full_sequence_uhead':
                    # uhead re-runs the model on train data via HF Trainer, so pre-computed train stats aren't needed
                    if "train_" + feature_dep not in dependencies:
                        dependencies.append("train_" + feature_dep)

        self.require_attention = 'stats_full_att_map' in dependencies
        self.require_hidden_states = 'stats_full_hidden_states' in dependencies
        self.require_scores = 'stats_scores' in dependencies

        self.feature_extractors = self.cfg.feature_extractor

        if self.pre_compile_features:
            # non-uhead heads use CompiledFeatures which handles feature extraction itself; replace raw stat dependencies
            dependencies = ["train_greedy_tokens", "train_target_texts", "input_tokens", "stats_compiled_probe_features", "train_stats_compiled_probe_features"]

        super().__init__(dependencies, "sequence")

    def __str__(self):
        return f"FeatureSupervision_{self.cfg.head_type}_{self.cfg.feature_extractor}"

    def _get_head_inputs(self, stats, head_type, train=False):
        train_prefix = "train_" if train else ""

        if self.pre_compile_features:
            compiled_features = stats[train_prefix + 'features']
            return 'pre_compiled', {'features': compiled_features}

        llm_inputs = {}
        llm_outputs = {}

        assert head_type == 'full_sequence_uhead'

        if 'stats_full_att_map' in self.stats_dependencies:
            llm_outputs['attentions'] = stats['full_attentions']
        if 'stats_full_hidden_states' in self.stats_dependencies:
            llm_outputs['hidden_states'] = stats['full_hidden_state']
        if 'stats_scores' in self.stats_dependencies:
            llm_outputs['scores'] = stats['out_scores']

        greedy_tokens = stats['greedy_tokens']
        input_tokens = stats['input_tokens']

        assert len(greedy_tokens) == len(input_tokens)
        assert len(input_tokens) == 1

        full_tokens = [torch.tensor(input_tokens[0] + greedy_tokens[0]).to(self.device)]
        output_mask = torch.zeros(len(full_tokens[0]))
        output_mask[len(input_tokens[0]):] = 1
        assert sum(output_mask).item() == len(greedy_tokens[0])
        output_mask = output_mask[1:].to(self.device).to(dtype=torch.long)
        att_mask = torch.ones(len(full_tokens[0])).to(self.device).to(dtype=torch.long)

        orig_att_mask = torch.ones(len(input_tokens[0])).to(self.device)
        assert orig_att_mask.shape[0] == len(input_tokens[0])

        context_length = torch.tensor(len(input_tokens[0])).to(self.device)

        llm_outputs['context_lengths'] = context_length.unsqueeze(0)
        llm_outputs['sequences'] = full_tokens[0].unsqueeze(0)
        llm_outputs['full_attention_mask'] = att_mask.unsqueeze(0)
        llm_inputs['output_mask'] = output_mask.unsqueeze(0)
        llm_inputs['attention_mask'] = orig_att_mask.unsqueeze(0)

        return llm_inputs, llm_outputs

    def __call__(self, stats: Dict[str, np.ndarray], max_no_tokens) -> np.ndarray:

        if not self.is_fitted:

            assert not self.head.is_fitted

            metrics = []
            train_greedy_texts = stats['train_greedy_texts']
            train_target_texts = stats['train_target_texts']
            train_greedy_tokens = stats['train_greedy_tokens']
            train_input_texts = stats['train_input_texts']
            train_source_ids = stats["train_source_ids"] if self.metric_router is not None else [None] * len(train_greedy_texts)

            for x, y, i_t, src in zip(train_greedy_texts, train_target_texts, train_input_texts, train_source_ids):
                metric = self.metric_router(src) if self.metric_router is not None else self.metric

                if isinstance(y, list) and (not self.aggregated):
                    y_ = y[0]
                elif isinstance(y, str) and (self.aggregated):
                    y_ = [y]
                else:
                    y_ = y
                metrics.append(metric({"greedy_texts": [x], "target_texts": [y_], "input_texts": [i_t]}, [y_])[0])

            targets = 1 - np.array(metrics)  # invert quality scores: high metric = good output = low uncertainty

            if self.cfg.head_type == 'full_sequence_uhead':
                model_w_head = CausalLMWithUncertaintyLayer(
                    base_model=self.base_model,
                    ue_head=self.head,
                    ue_pos_weight=self.head.ue_pos_weight,
                    require_attentions=self.require_attention,
                    require_hidden_states=self.require_hidden_states,
                    require_scores=self.require_scores,
                    max_no_tokens=max_no_tokens
                )

                train_args = TrainingArguments(
                    num_train_epochs=self.cfg.uncertainty_head.num_train_epochs,
                    per_device_train_batch_size=self.cfg.uncertainty_head.train_batch_size,
                    gradient_accumulation_steps=4,
                    learning_rate=self.cfg.uncertainty_head.learning_rate,
                    weight_decay=self.cfg.uncertainty_head.weight_decay,
                    max_grad_norm=1.0,
                    warmup_ratio=self.cfg.uncertainty_head.warmup_ratio,
                    lr_scheduler_type="linear",
                    fp16=True,
                    fp16_full_eval=False,
                    output_dir="outputs",
                    include_num_input_tokens_seen=True,
                    gradient_checkpointing=False,
                    dataloader_num_workers=1,
                    remove_unused_columns=False,
                )

                train_data = Dataset.from_dict({
                    'train_input_texts': train_input_texts,
                    'train_greedy_texts': train_greedy_texts,
                    'train_greedy_tokens': train_greedy_tokens,
                    'targets': targets,
                })

                trainer = Trainer(
                    model=model_w_head,
                    train_dataset=train_data,
                    args=train_args,
                    data_collator=DataCollator(self.base_model),
                )

                trainer.train()
                model_w_head.ue_head.is_fitted = True

                assert model_w_head.ue_head.is_fitted == self.head.is_fitted

                self.head.eval()

            else:
                llm_inputs, llm_outputs = self._get_head_inputs(stats, self.cfg.head_type, train=True)
                self.head.train(llm_inputs, llm_outputs, targets)

        assert self.head.is_fitted
        self.is_fitted = self.head.is_fitted

        llm_inputs, llm_outputs = self._get_head_inputs(stats, self.cfg.head_type, train=False)

        results = self.head(llm_inputs, llm_outputs)

        assert isinstance(results.item(), float)

        return [results.item()]
