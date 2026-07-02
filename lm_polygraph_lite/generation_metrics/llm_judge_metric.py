import json
import numpy as np

from typing import List, Dict
from .generation_metric import GenerationMetric


class LLMJudgeMetric(GenerationMetric):
    """
    Returns pre-computed LLM-as-judge scores for each sample, looked up by input_text.
    Scores are produced by run_llm_judge.py.

    File format (JSON array):
        [{"input_text": "...", "target": "...", "model_answer": "...", "judge_score": 0.7}, ...]

    Raises KeyError if any sample's input_text is missing from the file.
    """

    def __init__(self, judge_file: str):
        super().__init__([], "sequence")
        with open(judge_file) as f:
            data = json.load(f)
        self._scores = {item["input_text"]: float(item["judge_score"]) for item in data}

    def __str__(self):
        return "LLMJudge"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        result = []
        for input_text in stats["input_texts"]:
            if input_text not in self._scores:
                raise KeyError(
                    f"LLMJudgeMetric: no judge score found for input_text: {input_text[:100]!r}"
                )
            result.append(self._scores[input_text])
        return np.array(result)
