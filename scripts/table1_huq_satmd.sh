#!/usr/bin/env bash
# Table 1: HUQ-SATMD, Llama-3.1-8B
#
# Set HF_HUB_TOKEN in your environment before running if models are gated.

set -e
PYTHON="${PYTHON:-python}"

EVAL_DATASETS=(sciq trivia_qa qa pubmed_qa xsum cnn_dailymail)
OOD_SETTINGS=(ID OOD_LEAVE_ONE_OUT OOD_ONE_DATASET_SAME_TASK OOD_DIFF_TASK OOD_ONE_DATASET_DIFF_TASK)
declare -A MAX_NEW_TOKENS=([sciq]=20 [trivia_qa]=20 [qa]=20 [xsum]=56 [pubmed_qa]=128 [cnn_dailymail]=128)

COMMON=(
  --method huq_satmd
  --batch_size 1
  --model_path meta-llama/Meta-Llama-3.1-8B
  --attn_implementation eager
  --metric_thr 0.3
  --clean_md_device cuda
)

for ds in "${EVAL_DATASETS[@]}"; do
  for setting in "${OOD_SETTINGS[@]}"; do
    $PYTHON run_polygraph.py "${COMMON[@]}" --eval_dataset "$ds" --ood_setting "$setting" --max_new_tokens "${MAX_NEW_TOKENS[$ds]}"
  done
done
