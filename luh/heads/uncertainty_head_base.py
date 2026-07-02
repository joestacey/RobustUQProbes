from abc import abstractmethod

import torch
import torch.nn as nn

import logging

log = logging.getLogger()

class UncertaintyHeadBase(nn.Module):
    def __init__(
        self,
        feature_extractor,
        cfg=None,
    ):
        super().__init__()

        self.feature_extractor = feature_extractor

    @abstractmethod
    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        pass

    def _get_attn_mask(self, llm_inputs, llm_outputs):
        return llm_outputs["full_attention_mask"][:, 1:]

    def forward(self, llm_inputs, llm_outputs):

        if llm_inputs == 'pre_compiled':
            features = llm_outputs['features']
            features_attn_mask = None
        else:
            features = self.feature_extractor(llm_inputs, llm_outputs)
            features_attn_mask = self._get_attn_mask(llm_inputs, llm_outputs)

        return self._compute_tensors(llm_inputs, features, features_attn_mask)
