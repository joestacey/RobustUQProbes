#!/usr/bin/env python3
"""
Collect model text outputs for LLM-as-judge scoring.

Runs the model on either the evaluation or training split (loaded via
ProbeDrift, matching run_polygraph.py's dataset loading) and saves
per-sample outputs in a format suitable for run_llm_judge.py.

Usage:
    python collect_llm_judge_inputs.py --eval_dataset sciq --ood_setting ID \
        --split eval --output_file judge_inputs/eval_sciq_llama.json \
        --model_path meta-llama/Meta-Llama-3.1-8B
    python collect_llm_judge_inputs.py --eval_dataset sciq --ood_setting ID \
        --split train --output_file judge_inputs/train_sciq_llama.json \
        --model_path meta-llama/Meta-Llama-3.1-8B

--eval_dataset/--ood_setting/--instruct/--model_path (or --model_config) must
match the run_polygraph.py invocation being judged, so model and dataset
match exactly.
"""

import argparse
import os
import json
import logging
from pathlib import Path

import transformers
from tqdm import tqdm

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

from utils.pipeline import get_cache_kwargs, load_model, load_datasets_via_probe_drift
from run_polygraph import build_model_config


def _format_target(target):
    if isinstance(target, list):
        return " / ".join(str(t) for t in target)
    return str(target)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", required=True,
                         choices=["sciq", "trivia_qa", "qa", "pubmed_qa", "xsum", "cnn_dailymail"])
    parser.add_argument("--ood_setting", default="ID",
                         choices=["ID", "OOD_LEAVE_ONE_OUT", "OOD_ONE_DATASET_SAME_TASK",
                                  "OOD_DIFF_TASK", "OOD_ONE_DATASET_DIFF_TASK"])
    parser.add_argument("--instruct", action="store_true", default=False)
    parser.add_argument("--split", required=True, choices=["eval", "train"])
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--cache_path", default="./workdir/output")

    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--loadin4bit", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = get_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    cache_kwargs = get_cache_kwargs(args)

    model_cfg, anchor = build_model_config(args)
    model_args = argparse.Namespace(model=model_cfg, loadin4bit=args.loadin4bit)

    log.info(f"Loading model {model_cfg.path}...")
    transformers.set_seed(1)
    model = load_model(model_args, anchor, cache_kwargs)

    log.info(f"Loading {args.split} dataset for {args.eval_dataset!r} "
             f"(ood_setting={args.ood_setting!r}) via ProbeDrift...")
    train_dataset, eval_dataset = load_datasets_via_probe_drift(args)
    dataset = eval_dataset if args.split == "eval" else train_dataset
    log.info(f"Collecting outputs for {len(dataset.x)} samples...")

    results = []
    for inp_texts, target_texts in tqdm(dataset):
        model_answers = model.generate_texts(inp_texts, max_new_tokens=args.max_new_tokens)
        for input_text, target, model_answer in zip(inp_texts, target_texts, model_answers):
            results.append({
                "input_text": input_text,
                "target": _format_target(target),
                "model_answer": model_answer.replace("\n", "").strip(),
            })

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(results)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
