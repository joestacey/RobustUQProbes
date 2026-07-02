
import luh.feature_extractors.combined

def load_feature_extractor(config, base_model):
    return luh.feature_extractors.combined.load_extractor(config, base_model)
