import numpy as np

import tensorflow as tf

tf.config.optimizer.set_jit(False)
tf.config.set_visible_devices([], "GPU") # forces TF to use CPU, avoiding sharing GPU with pytorch

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()

class FullSeqHeadSAPLMA(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        cfg = None,
    ):
        super().__init__(feature_extractor, cfg=cfg)
        self.is_fitted = False

        # Using https://github.com/sisinflab/HidingInTheHiddenStates model code
        self.model = Sequential()
        self.model.add(Dense(256, activation='relu', input_dim=feature_extractor.feature_dim()))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, llm_inputs, llm_outputs, targets):
        assert not self.is_fitted
        assert llm_inputs == 'pre_compiled'
        features = llm_outputs['features']
        model_features = np.array([f.squeeze(0).float().cpu() for f in features])
        self.model.fit(model_features, targets, epochs=5, batch_size=1)
        self.is_fitted = True

    def _compute_tensors(self, llm_inputs, features, features_attn_mask):
        assert llm_inputs == 'pre_compiled'
        assert self.is_fitted
        claim_ues = self.model.predict(features.float().cpu().numpy())
        assert len(claim_ues) == 1
        return claim_ues[0]
