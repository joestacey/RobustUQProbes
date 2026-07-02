
import torch

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_layer_nums


class FeatureExtractorBasicHiddenStates(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums=[-1], **kwargs):
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        self._feature_dim = orig_base_model.config.hidden_size * len(self._layer_nums)

    def __call__(self, llm_inputs, llm_outputs):
        """ output shape: (batch_size x num_generation_steps x hidden_size * len(layer_nums)) """
        return torch.cat(
            [
                torch.cat([t[layer] for layer in self._layer_nums], dim=-1)
                for t in llm_outputs["hidden_states"]
            ],
            dim=1,
        )

    def feature_dim(self):
        return self._feature_dim

def load_extractor(config, base_model):
    return FeatureExtractorBasicHiddenStates(base_model, **config)
