from typing import Dict, List
import numpy as np

from .stat_calculator import StatCalculator
from lm_polygraph_lite.utils.model import Model


class InitialStateCalculator(StatCalculator):
    """Passes input texts through unchanged, satisfying the 'input_texts' dependency."""

    def __init__(self):
        super().__init__(["input_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        return {"input_texts": texts}
