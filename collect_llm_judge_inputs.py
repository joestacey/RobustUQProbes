#!/usr/bin/env python3
"""
Collect model text outputs for LLM-as-judge scoring.

Runs the model on either the evaluation or training split and saves per-sample
outputs in a format suitable for run_llm_judge.py.

Usage:
    HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python collect_llm_judge_inputs.py \
        +split=eval +output_file=judge_inputs/eval_sciq_llama.json
    HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python collect_llm_judge_inputs.py \
        +split=train +output_file=judge_inputs/train_sciq_llama.json

The config must be identical to the one used in run_polygraph.py so that model,
dataset, and subsampling match exactly.
"""

import hydra
import os
import json
import logging
import transformers
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger()

from utils.pipeline import get_cache_kwargs, load_model, load_dataset

hydra_config = Path(os.environ["HYDRA_CONFIG"])


def _format_target(target):
    if isinstance(target, list):
        return " / ".join(str(t) for t in target)
    return str(target)


@hydra.main(
    version_base=None,
    config_path=str(hydra_config.parent),
    config_name=str(hydra_config.name),
)
def main(args):
    os.chdir(hydra.utils.get_original_cwd())

    if not hasattr(args, 'split') or args.split not in ('eval', 'train'):
        raise ValueError("Must specify +split=eval or +split=train")
    if not hasattr(args, 'output_file'):
        raise ValueError("Must specify +output_file=<path>")

    output_file = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    cache_kwargs = get_cache_kwargs(args)

    log.info(f"Loading model {args.model.path}...")
    transformers.set_seed(1)
    model = load_model(args, hydra_config, cache_kwargs)

    log.info(f"Loading {args.split} dataset...")
    dataset = load_dataset(args, args.split, cache_kwargs)
    log.info(f"Collecting outputs for {len(dataset.x)} samples...")

    results = []
    for inp_texts, target_texts, max_new_tokens in tqdm(dataset):
        model_answers = model.generate_texts(inp_texts, max_new_tokens=max(max_new_tokens))
        for input_text, target, model_answer in zip(inp_texts, target_texts, model_answers):
            results.append({
                "input_text": input_text,
                "target": _format_target(target),
                "model_answer": model_answer.replace("\n", "").strip(),
            })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(results)} samples to {output_file}")


if __name__ == "__main__":
    main()
