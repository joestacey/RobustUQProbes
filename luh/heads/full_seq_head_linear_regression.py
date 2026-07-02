import numpy as np

from .uncertainty_head_base import UncertaintyHeadBase

class FullSeqHeadLinearRegression(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        cfg = None,
    ):
        from sklearn.linear_model import LinearRegression
        super().__init__(feature_extractor, cfg=cfg)
        self.classifier = LinearRegression()
        self.is_fitted = False

    def train(self, llm_inputs, llm_outputs, targets):
        assert not self.is_fitted
        assert llm_inputs == 'pre_compiled'
        # features: list of (1, feature_dim) tensors, one per training example
        features = llm_outputs['features']
        X = np.array([f.squeeze(0).cpu().numpy() for f in features])  # (n_train, feature_dim)
        self.classifier.fit(X, targets)
        self.is_fitted = True

    def _compute_tensors(self, llm_inputs, features, features_attn_mask):
        assert llm_inputs == 'pre_compiled'
        assert self.is_fitted
        # features: (1, feature_dim) tensor for the current example
        return self.classifier.predict(features.cpu().numpy())
