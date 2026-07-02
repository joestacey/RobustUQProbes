
import torch

from .feature_extractor_base import FeatureExtractorBase
from .lookback_lens import process_attentions
from .utils import get_layer_nums, get_head_nums


class FeatureExtractorBasicAttention(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums, attn_history_sz, offset, pool, pool_type=None, head_nums='all', **kwargs):
        """layer_nums = {all, [-1, -2, -5, ...]}"""

        self.pool = pool
        self.pool_type = pool_type
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        self._head_nums = get_head_nums(head_nums, self._layer_nums, orig_base_model)
        self._input_size = (
            sum(len(h) for h in self._head_nums.values())
            if not pool
            else len(list(self._head_nums.values())[0])
        )
        if pool:
            n_heads = len(self._head_nums[list(self._head_nums.keys())[0]])
            assert all(len(v) == n_heads for v in self._head_nums.values())
        self._attn_history_sz = attn_history_sz

        self.offset = offset

    def feature_dim(self):
        return self._input_size * self._attn_history_sz

    def __call__(self, llm_inputs, llm_outputs):

        attentions_all: list[tuple] = process_attentions(llm_outputs['attentions'])

        batch_size, seq_len = llm_inputs['attention_mask'].shape
        all_features = []  # [seq_len](batch_sz, attn_hist, heads, layers)
        for i in range(len(attentions_all)):
            i_features = []  # [layers_num](batch_sz, attn_hist, heads)
            for layer_num in self._layer_nums:
                cur_attn = attentions_all[i][layer_num]  # (batch_sz, n_heads, 1, prev_seq_len)
                attn_index = torch.LongTensor([a for a in range(
                    cur_attn.shape[-1] - 1,
                    cur_attn.shape[-1] - self._attn_history_sz - 1 + self.offset,
                    -1,
                )])

                cur_features = cur_attn[:, :, 0, attn_index.clamp(0)].permute(0, 2, 1)  # (batch_sz, attn_hist, n_heads)

                cur_features = cur_features[:, :, self._head_nums[layer_num]]  # (batch_sz, attn_hist, heads)
                
                cur_features[:, attn_index < 0, :] = 0.0
                i_features.append(cur_features)

            all_features.append(torch.stack(i_features, dim=-1))  # (batch_sz, attn_hist, heads, layers)
        all_features = torch.stack(all_features, dim=1)  # (batch_sz, seq_len, attn_hist, heads, layers)

        if self.pool:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                all_features = torch.amax(all_features, dim=-1)  # (batch_sz, seq_len, attn_hist, heads)
            elif self.pool_type == 'mean':
                all_features = torch.mean(all_features, dim=-1)
        
        return all_features.reshape(batch_size, len(attentions_all), -1)

    def output_attention(self):
        return True


def load_extractor(config, base_model):
    return FeatureExtractorBasicAttention(base_model, **config)

