#!/usr/bin/env bash
# SAPLMA architecture ablations, Llama-3.1-8B: lin_reg / uhead_v1 / uhead_v2.
#
# Set HF_HUB_TOKEN in your environment before running if models are gated.

set -e
PYTHON="${PYTHON:-python}"

EVAL_DATASETS=(sciq trivia_qa qa pubmed_qa xsum cnn_dailymail)
OOD_SETTINGS=(ID OOD_LEAVE_ONE_OUT OOD_ONE_DATASET_SAME_TASK OOD_DIFF_TASK OOD_ONE_DATASET_DIFF_TASK)
declare -A MAX_NEW_TOKENS=([sciq]=20 [trivia_qa]=20 [qa]=20 [xsum]=56 [pubmed_qa]=128 [cnn_dailymail]=128)

for ARCH in lin_reg uhead_v1 uhead_v2; do

case $ARCH in
  lin_reg)
    EXTRA=(
      --probe_head_type full_sequence_linear_regression
    )
    ;;
  uhead_v1)
    EXTRA=(
      --probe_head_type full_sequence_uhead
      --probe_head_dim 768
      --probe_n_layers 1
      --probe_n_heads 16
      --probe_dropout 0.05
      --probe_learning_rate 0.0002
      --probe_weight_decay 0.1
      --probe_warmup_ratio 0.1
      --probe_num_train_epochs 6
      --probe_train_batch_size 1
    )
    ;;
  uhead_v2)
    EXTRA=(
      --probe_head_type full_sequence_uhead
      --probe_head_dim 768
      --probe_n_layers 2
      --probe_n_heads 4
      --probe_dropout 0.2
      --probe_learning_rate 0.0002
      --probe_weight_decay 0.1
      --probe_warmup_ratio 0.1
      --probe_num_train_epochs 7
      --probe_train_batch_size 1
    )
    ;;
esac

COMMON=(
  --method feature_supervision
  --probe_feature_extractor_setting hs_middle
  --token_aggregation average
  --batch_size 1
  --model_path meta-llama/Meta-Llama-3.1-8B
  --attn_implementation eager
  --clean_md_device cuda
  "${EXTRA[@]}"
)

  for ds in "${EVAL_DATASETS[@]}"; do
    for setting in "${OOD_SETTINGS[@]}"; do
      $PYTHON run_polygraph.py "${COMMON[@]}" --eval_dataset "$ds" --ood_setting "$setting" --max_new_tokens "${MAX_NEW_TOKENS[$ds]}"
    done
  done

done
