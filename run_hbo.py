"""
Combines MSP, SAPLMA-probe, and SATMD outputs (all produced by
run_polygraph.py; see scripts/README.md) into one HBO score per eval example.

Inputs:
  --msp_file          output_file from a --method msp run.
  --probe_file        output_file from a --method feature_supervision
                      --probe_head_type full_sequence_saplma run.
  --md_save_prefix    md_save_file prefix from a --method satmd run.
  --layer_index       Which saved layer to use. Default -1 (last saved).

All three inputs must come from runs using the same eval dataset config and
subsample_eval_dataset value (fixed seed=1 keeps the eval order aligned).

Output: one JSON line per eval example, {"input_text": ..., "score": ...},
written to --output.

Usage:
    python run_hbo.py --msp_file hbo_outputs/msp_sciq.jsonl \\
        --probe_file hbo_outputs/saplma_sciq_id.jsonl \\
        --md_save_prefix hbo_outputs/satmd_sciq_id \\
        --layer_index -1 \\
        --output hbo_outputs/hbo_sciq_id.jsonl
"""
import argparse
import json

import numpy as np
from scipy.stats import rankdata


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msp_file", required=True, help="output_file from a --method msp run")
    parser.add_argument("--probe_file", required=True, help="output_file from a --method feature_supervision (SAPLMA) run")
    parser.add_argument("--md_save_prefix", required=True, help="md_save_file prefix from a --method satmd run")
    parser.add_argument("--layer_index", type=int, default=-1, help="position within the saved layers list (default: last)")
    parser.add_argument("--output", required=True, help="where to write the combined predictions")
    return parser.parse_args()


def load_score_jsonl(path):
    input_texts, scores = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            input_texts.append(row["input_text"])
            scores.append(row["score"])
    return input_texts, np.array(scores, dtype=float)


def main():
    args = get_args()

    msp_input_texts, msp_scores = load_score_jsonl(args.msp_file)
    probe_input_texts, probe_scores = load_score_jsonl(args.probe_file)

    assert msp_input_texts == probe_input_texts, (
        "msp_file and probe_file evaluated different examples (or different "
        "order) - re-run both with the same --eval_dataset"
    )

    with open(args.md_save_prefix + ".jsonl") as f:
        test_md = np.array([json.loads(line) for line in f], dtype=float)  # (n_eval, n_layers)
    dev_md = np.load(args.md_save_prefix + "_train.npy")  # (n_dev, n_layers)

    assert test_md.shape[0] == len(msp_input_texts), (
        f"md_save_prefix has {test_md.shape[0]} eval examples but msp_file/"
        f"probe_file have {len(msp_input_texts)} - re-run all three with the "
        f"same --eval_dataset"
    )

    test_md_layer = test_md[:, args.layer_index]
    dev_md_layer = dev_md[:, args.layer_index].tolist()

    percentiles = np.array([
        rankdata(dev_md_layer + [x])[-1] / (len(dev_md_layer) + 1)
        for x in test_md_layer
    ])

    msp_weight = np.minimum(1.0, percentiles + 0.5)
    probe_weight = 1.0 - msp_weight

    msp_rank = rankdata(msp_scores)
    probe_rank = rankdata(probe_scores)

    final_scores = msp_weight * msp_rank + probe_weight * probe_rank

    with open(args.output, "w") as f:
        for input_text, score in zip(msp_input_texts, final_scores):
            json.dump({"input_text": input_text, "score": float(score)}, f)
            f.write("\n")

    print(f"Wrote {len(final_scores)} combined predictions to {args.output}")


if __name__ == "__main__":
    main()
