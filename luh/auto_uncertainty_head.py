
from .heads.full_seq_head_uhead import FullSeqHeadUHead
from .heads.full_seq_head_saplma import FullSeqHeadSAPLMA
from .heads.full_seq_head_linear_regression import FullSeqHeadLinearRegression

from .utils import load_feature_extractor


class AutoUncertaintyHead:
    MODEL_MAPPING = {
        "full_sequence_uhead": FullSeqHeadUHead,
        "full_sequence_saplma": FullSeqHeadSAPLMA,
        "full_sequence_linear_regression": FullSeqHeadLinearRegression,
    }

    @classmethod
    def from_config(cls, config, base_model):
        uq_head_type = cls.MODEL_MAPPING[config.head_type]

        feature_extractor = load_feature_extractor(
            config.feature_extractor, base_model
        )

        ue_head_cfg = dict() if config.uncertainty_head is None else config.uncertainty_head
        uq_head = uq_head_type(
            feature_extractor,
            cfg=config,
            **ue_head_cfg,
        )

        return uq_head
