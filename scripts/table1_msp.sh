#!/usr/bin/env bash
# Table 1: MSP, Llama-3.1-8B
#
# MSP is training-free, so one run per dataset covers all OOD settings.
#
# Set HF_HUB_TOKEN in your environment before running if models are gated.

set -e
PYTHON="${PYTHON:-python}"

EVAL_DATASETS=(sciq trivia_qa qa pubmed_qa xsum cnn_dailymail)
declare -A MAX_NEW_TOKENS=([sciq]=20 [trivia_qa]=20 [qa]=20 [xsum]=56 [pubmed_qa]=128 [cnn_dailymail]=128)

COMMON=(
  --method msp
  --batch_size 1
  --model_path meta-llama/Meta-Llama-3.1-8B
  --attn_implementation eager
)

for ds in "${EVAL_DATASETS[@]}"; do
  $PYTHON run_polygraph.py "${COMMON[@]}" --eval_dataset "$ds" --max_new_tokens "${MAX_NEW_TOKENS[$ds]}"
done
