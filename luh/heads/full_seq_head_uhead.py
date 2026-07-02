
import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()

class FullSeqHeadUHead(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        num_train_epochs: int = 7,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        dropout: float = 0.1,
        ue_pos_weight: float = None,
        train_batch_size: int = 4,
        cfg = None,
    ):
        super().__init__(feature_extractor, cfg=cfg)

        self.ue_pos_weight = ue_pos_weight

        self.feature_extractor = feature_extractor
        log.info(f"Feature size: {feature_extractor.feature_dim()}")

        self.is_fitted = False

        self.proj = nn.Sequential(
                nn.Linear(feature_extractor.feature_dim(), head_dim * 2),
                nn.LayerNorm(head_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim * 2, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
            )

        # Embedding distinguishing generated tokens (1) from context tokens (0)
        self.entity_embedding = nn.Embedding(2, head_dim)

        encoder_layer = nn.TransformerEncoderLayer(
                d_model=head_dim, nhead=n_heads, dropout=dropout, activation="gelu", batch_first=True
            )

        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

        self.classifier = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(head_dim, 1)
            )

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total number of parameters {total_params}")

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):

        output_masks_all = llm_inputs["output_mask"]

        features = X.to(torch.float32)
        features = self.proj(features)


        src_key_padding_mask = (X_attn_mask == 0)
        results = []
        batch_size = len(llm_inputs['output_mask'])

        for i in range(batch_size):
            output_mask = output_masks_all[i].unsqueeze(dim=0)

            # Output mask should be something like torch.tensor([0, 0, 0, 1, 1, 1, 1])
            # .. with 0's for context and 1's for the generated tokens
            ent_embeds = self.entity_embedding(output_mask)

            assert ent_embeds.shape[0] == 1
            
            out = features[i].unsqueeze(0)  + ent_embeds

            # cuDNN's SDPA backend raises "No execution plans support the graph" for this
            # op's shapes on at least one dev GPU (RTX 4060 Ti, cuDNN 9.1); disabling it
            # only changes which kernel computes the (mathematically identical) attention,
            # not the architecture or training procedure.
            with torch.backends.cuda.sdp_kernel(enable_cudnn=False):
                out = self.transformer_encoder(
                    out, src_key_padding_mask=src_key_padding_mask[i].unsqueeze(0).repeat(ent_embeds.shape[0], 1))

            sum_output_embeds = (out * output_mask.unsqueeze(-1)).sum(dim=1)
            count_output_tokens = output_mask.sum(dim=1)
            assert count_output_tokens >= 1

            output_representation = sum_output_embeds / count_output_tokens.unsqueeze(-1)

            out = self.classifier(output_representation)
            results.append(out)

        # Padding to ensure uniform output shape
        max_entities_per_batch = max(o.shape[0] for o in results)
        assert max_entities_per_batch == 1

        padded_results = [F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100) for o in results]

        return torch.stack(padded_results)

