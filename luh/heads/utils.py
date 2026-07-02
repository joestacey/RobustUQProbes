
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

import torch
from torch.nn import BCEWithLogitsLoss

import torch.nn.init as init

from typing import Optional
from dataclasses import dataclass
@dataclass
class CausalLMWithUncertaintyOutput(CausalLMOutput):
    loss: Optional[torch.FloatTensor] = None
    uncertainty: torch.FloatTensor = None


class CausalLMWithUncertaintyLayer(PreTrainedModel):
    def __init__(
        self,
        base_model,
        ue_head,
        ue_pos_weight: float,
        require_attentions: bool,
        require_hidden_states: bool,
        require_scores: bool,
        max_no_tokens = None
    ):
        super().__init__(PretrainedConfig())

        self._base_model = base_model
        self.ue_head = ue_head
        self._ue_pos_weight = ue_pos_weight

        self.require_scores = require_scores
        self.require_attentions= require_attentions
        self.require_hidden_states = require_hidden_states

        self.max_no_tokens = max_no_tokens

        self.ue_head.apply(self.reinitialize_weights)

    def reinitialize_weights(self, module):
        
        if hasattr(module, "weight") and module.weight is not None and (not(hasattr(module, "name") and "positional_encoding" in module.name)):
            if module.weight.ndim >= 2:
                init.xavier_uniform_(module.weight)
            else:
                init.uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            init.zeros_(module.bias)

        return

    def get_features_from_llm_output(self, out, input_ids, attention_mask, padding_token):

        device = input_ids.device

        llm_inputs = {}
        llm_outputs = {}

        full_tokens_tensor = out.sequences

        for i in range(len(input_ids)):
            assert torch.equal(full_tokens_tensor[i][:len(input_ids[i])], input_ids[i])

        output_mask = torch.zeros(full_tokens_tensor.shape)
        for i, seq in enumerate(input_ids):
            output_mask[i, len(input_ids[i]):len(full_tokens_tensor[i])] = 1

        output_mask[full_tokens_tensor == padding_token] = 0

        att_mask = output_mask.clone()
        output_mask = output_mask[:, 1:].to(device).to(dtype=torch.long)

        llm_inputs['output_mask'] = output_mask
        llm_inputs['attention_mask'] = attention_mask

        for i, seq in enumerate(input_ids):
            att_mask[i, :len(attention_mask[i])] = attention_mask[i]

        llm_outputs['full_attention_mask'] = att_mask.to(device)

        context_length = torch.tensor([len(x) for x in input_ids]).to(device)
        llm_outputs['context_lengths'] = context_length

        llm_outputs['sequences'] = full_tokens_tensor

        if self.require_attentions:
            llm_outputs['attentions'] = out.attentions
        if self.require_hidden_states:
            llm_outputs['hidden_states'] = out.hidden_states
        if self.require_scores:
            llm_outputs['scores'] = out.scores

        return llm_inputs, llm_outputs
        

    def forward(
        self,
        input_ids,
        attention_mask,
        targets=None,
        **kwargs
    ):
        
        padding_token = self._base_model.tokenizer.pad_token_id

        input_ids = input_ids.to(self._base_model.model.device)
        attention_mask = attention_mask.to(self._base_model.model.device)

        torch.cuda.empty_cache()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                outputs = self._base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_scores=self.require_scores,
                    return_dict_in_generate=True,
                    max_new_tokens=self.max_no_tokens,
                    min_new_tokens=2,
                    output_attentions=self.require_attentions,
                    output_hidden_states=self.require_hidden_states,
                    num_return_sequences=1,
                    do_sample=False,
                    )

        assert self._base_model.model.training == False

        llm_inputs, llm_outputs = self.get_features_from_llm_output(outputs, input_ids, attention_mask, padding_token)

        uncertainty = self.ue_head(llm_inputs, llm_outputs)

        uncertainty_raveled = uncertainty.reshape(-1)
        uncertainty_targets = torch.tensor(targets).reshape(-1).to(
                uncertainty_raveled.device)

        if self._ue_pos_weight:

            loss_fct = BCEWithLogitsLoss(
                pos_weight=torch.Tensor([self._ue_pos_weight]).to(
                    uncertainty_raveled.device
                )
            )
        else:
            loss_fct = BCEWithLogitsLoss()
         
        loss = loss_fct(
            uncertainty_raveled.to(torch.float32),
            uncertainty_targets.to(torch.float32),
        )

        return CausalLMWithUncertaintyOutput(
            loss=loss,
            uncertainty=uncertainty,
        )

