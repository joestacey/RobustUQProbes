import torch
import torch.nn.functional as F

from .feature_extractor_base import FeatureExtractorBase


class FeatureExtractorTokenProbabilities(FeatureExtractorBase):
    def __init__(self, orig_base_model, top_n, temperature=1.0, **kwargs):
        """
        Extracts features based on token probabilities with optional temperature scaling.

        :param orig_base_model: The original base model, used to infer the number of tokens and other properties.
        :param top_n: Number of top tokens to keep.
        :param temperature: Scaling factor for logits before applying softmax. Temperature < 1 makes probabilities sharper, while temperature > 1 makes them smoother.
        """
        self.temperature = temperature
        self.top_n = top_n

    def _instance_feature(self, logits):
        """
        Extracts token-level probabilities for a single instance (sequence) with temperature scaling.

        :param logits: Logits from the model (before applying softmax), shape (sequence_length x vocab_size).
        :return: Top-n token-level probabilities, shape (sequence_length x top_n).
        """
        mask = logits.sum(dim=-1) != 0.
        top_probas = torch.zeros(*logits.shape[:-1], self.top_n, device=logits.device, dtype=logits.dtype)
        
        def get_top_n_probas(inpt_logits):
            res_p = F.softmax(inpt_logits / self.temperature, dim=-1)
            res_p = res_p.topk(self.top_n, dim=-1)[0]
            return res_p.to(top_probas.dtype)

        top_probas[mask] = get_top_n_probas(logits[mask])
        return top_probas

    def __call__(self, llm_inputs, llm_outputs):
        """
        Extracts token probabilities from model output.

        :param llm_inputs: Model inputs.
        :param llm_outputs: Model outputs containing 'sequences', 'scores', and 'context_lengths'.
        :return: Tensor of top-n token probabilities, shape (batch_size x sequence_length x top_n).
        """
        batch_size = llm_outputs['sequences'].shape[0]
        seq_len = llm_outputs['sequences'].shape[1] - 1
        logits = torch.zeros(size=(batch_size, seq_len, llm_outputs['scores'][0].shape[-1]), device=llm_outputs['scores'][0].device)
        for i in range(batch_size):
            for j in range(llm_outputs['context_lengths'][i] - 1, seq_len):
                logits[i, j, :] = llm_outputs['scores'][j - llm_outputs['context_lengths'][i] + 1][i]

        # Logits shape: (batch_size x sequence_length x vocab_size)

        all_token_features = self._instance_feature(logits)
        return all_token_features

    def feature_dim(self):
        """
        Returns the dimensionality of the token-level feature vector.

        :return: Number of top tokens kept (top_n).
        """
        return self.top_n


def load_extractor(config, base_model):
    """
    Loads the token probability feature extractor with optional temperature scaling.

    :param config: A dictionary-like configuration object.
    :param base_model: The original model from which to extract logits and probabilities.
    :return: Instance of FeatureExtractorTokenProbabilities.
    """
    return FeatureExtractorTokenProbabilities(base_model, **config)

