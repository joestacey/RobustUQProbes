import os
import logging

from lm_polygraph_lite.stat_calculators import *
from lm_polygraph_lite.utils.model import Model

from typing import Dict, List, Tuple

log = logging.getLogger("lm_polygraph_lite")


def register_stat_calculators(
    language: str = "en",
    n_ccp_alternatives: int = 10,
    cache_path=os.path.expanduser("~") + "/.cache",
    model: Model = None,
) -> Tuple[Dict[str, "StatCalculator"], Dict[str, List[str]]]:
    stat_calculators: Dict[str, "StatCalculator"] = {}
    stat_dependencies: Dict[str, List[str]] = {}

    def _register(calculator_class: StatCalculator):
        for stat in calculator_class.stats:
            if stat in stat_calculators.keys():
                raise ValueError(
                    "A statistic is supposed to be processed by a single calculator only."
                )
            stat_calculators[stat] = calculator_class
            stat_dependencies[stat] = calculator_class.stat_dependencies

    _register(InitialStateCalculator())
    _register(GreedyProbsCalculator(n_alternatives=n_ccp_alternatives))
    _register(EntropyCalculator())
    _register(HSCalculator())
    _register(HSCalculator(stage=""))
    _register(AttentionMapCalculator())
    _register(AttentionMapCalculator(stage=""))
    _register(TokenProbsCalculator())
    _register(TokenProbsCalculator(stage=""))
    _register(CompiledFeatures())
    _register(CompiledFeatures(stage=""))

    if "gemma-3" in model.model_path:
        hidden_layers = list(range(model.model.config.text_config.num_hidden_layers - 1)) + [-1]
    else:
        hidden_layers = list(range(model.model.config.num_hidden_layers - 1)) + [-1]

    _register(EmbeddingsCalculator(hidden_layers=hidden_layers, stage="train"))
    _register(EmbeddingsCalculator(hidden_layers=hidden_layers, stage=""))

    return stat_calculators, stat_dependencies
