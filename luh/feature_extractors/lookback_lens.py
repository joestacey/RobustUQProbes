

import torch

from .feature_extractor_base import FeatureExtractorBase

from .utils import get_layer_nums

def process_attentions(attentions_all) -> list[tuple]:
    # Returns list[tuple[Tensor]]: one entry per sequence position; each tuple
    # holds one Tensor per layer of shape (batch_sz, heads, 1, attended_len).
    layer = 0

    attn_inp = [
        tuple(
            attentions_all[0][l][:, :, i:i + 1, :i + 1]
            for l in range(len(attentions_all[0]))
        )
        for i in range(attentions_all[0][layer].shape[-2])
    ]
    inp_len = attentions_all[0][layer].shape[-2]
    outp_len = len(attentions_all[1:])
    attn_outp = [
        tuple(
            a[l][:, :, :, :i + 1]
            for l in range(len(attentions_all[0]))
        )
        for i, a in zip(range(inp_len, inp_len + outp_len), attentions_all[1:])
    ]

    return attn_inp + attn_outp


class FeatureExtractorLookbackLens(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums, pool, pool_type=None, **kwargs):

        self._n_heads = orig_base_model.config.num_attention_heads
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)

        if pool:
            self._input_size = len(self._layer_nums)
        else:
            self._input_size = len(self._layer_nums) * self._n_heads

        self.pool = pool
        self.pool_type = pool_type

    def feature_dim(self):
        return self._input_size

    def __call__(self, llm_inputs, llm_outputs):

        attentions_all = process_attentions(llm_outputs['attentions'])
        batch_sz = llm_inputs['attention_mask'].shape[0]
        context_bounds = torch.as_tensor(llm_outputs['context_lengths'], device=attentions_all[0][0].device)
        context_lengths = [
            llm_inputs['attention_mask'][i][:context_bounds[i]].sum().item()
            for i in range(batch_sz)
        ]

        all_features = []
        for seq_idx, attentions in enumerate(attentions_all):
            features = []
            assert attentions[0].shape[2] == 1

            attn_ctx, attn_new = [], []
            for l in self._layer_nums:
                a = attentions[l]  # shape: (batch_sz, H, 1, seq_len)
                seq_range = torch.arange(a.shape[-1], device=a.device).unsqueeze(0)  # shape: (1, seq_len)
                ctx_mask = seq_range < context_bounds.unsqueeze(1)  # shape: (batch_sz, seq_len)
                new_mask = ~ctx_mask  # Complement of the context mask
                attn_ctx_layer = (a[:, :, 0, :] * ctx_mask.unsqueeze(1)).sum(-1)
                attn_new_layer = (a[:, :, 0, :] * new_mask.unsqueeze(1)).sum(-1)
                attn_ctx.append(attn_ctx_layer)
                attn_new.append(attn_new_layer)
            attn_ctx = torch.stack(attn_ctx)  # shape: (L, batch_sz, H)
            attn_new = torch.stack(attn_new)  # shape: (L, batch_sz, H)

            for batch_i in range(batch_sz):
                ctx_len = context_lengths[batch_i]
                ctx_bound = context_bounds[batch_i]
                if seq_idx > ctx_bound:  # output token
                    mean_attn_ctx = attn_ctx[:, batch_i, :] / ctx_len
                    mean_attn_new = attn_new[:, batch_i, :] / (seq_idx - ctx_bound)
                    lb_ratio = mean_attn_new / (mean_attn_ctx + mean_attn_new)
                else:  # context/prompt position
                    lb_ratio = torch.ones_like(attn_ctx[:, batch_i, :])  # L x H
                
                if self.pool:
                    assert self.pool_type in ['max', 'mean']
                    if self.pool_type == 'max':
                        lb_ratio = torch.amax(lb_ratio, dim=-1).unsqueeze(-1)
                    elif self.pool_type == 'mean':
                        lb_ratio = torch.mean(lb_ratio, dim=-1).unsqueeze(-1)

                features.append(lb_ratio.reshape(-1))
            features = torch.stack(features)  # batch_size x feature_vector (feature vector is L x H)
            all_features.append(features)

        # Output: batch_size x sequence_length x feature_vector
        result = torch.stack(all_features, dim=1)

        return result

    def output_attention(self):
        return True


def load_extractor(config, base_model, *args, **kwargs):
    return FeatureExtractorLookbackLens(base_model, **config)

