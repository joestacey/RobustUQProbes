#!/usr/bin/env bash
# Table 6: HBO, Llama-3.1-8B. ID setting, all 6 Table 1 eval datasets.
#
# Set HF_HUB_TOKEN in your environment before running if models are gated.

set -e
PYTHON="${PYTHON:-python}"
OUT=hbo_outputs
mkdir -p "$OUT"

EVAL_DATASETS=(sciq trivia_qa qa pubmed_qa xsum cnn_dailymail)
OOD_SETTINGS=(ID)
declare -A MAX_NEW_TOKENS=([sciq]=20 [trivia_qa]=20 [qa]=20 [xsum]=56 [pubmed_qa]=128 [cnn_dailymail]=128)

MSP_COMMON=(
  --method msp
  --batch_size 1
  --model_path meta-llama/Meta-Llama-3.1-8B
  --attn_implementation eager
)

SAPLMA_COMMON=(
  --method feature_supervision
  --probe_head_type full_sequence_saplma
  --probe_feature_extractor_setting hs_middle
  --token_aggregation average
  --batch_size 1
  --model_path meta-llama/Meta-Llama-3.1-8B
  --attn_implementation eager
  --clean_md_device cuda
)

SATMD_COMMON=(
  --method satmd
  --batch_size 1
  --model_path meta-llama/Meta-Llama-3.1-8B
  --attn_implementation eager
  --metric_thr 0.3
  --clean_md_device cuda
)

# Index 15 = middle layer of Llama-3.1-8B (32 layers, math.ceil(32/2)-1=15),
# matching hs_middle used by SAPLMA_COMMON.
LAYER_INDEX=15

for ds in "${EVAL_DATASETS[@]}"; do

  # Step 1: extract MSP predictions (one run per dataset, no training, reused across all OOD settings).
  msp_file="$OUT/msp_${ds}.jsonl"
  if [ ! -f "$msp_file" ]; then
    $PYTHON run_polygraph.py "${MSP_COMMON[@]}" --eval_dataset "$ds" --output_file "$msp_file" --max_new_tokens "${MAX_NEW_TOKENS[$ds]}"
  fi

  for setting in "${OOD_SETTINGS[@]}"; do
    saplma_file="$OUT/saplma_${ds}_${setting}.jsonl"
    md_prefix="$OUT/satmd_${ds}_${setting}"

    # Step 1: extract SAPLMA predictions.
    $PYTHON run_polygraph.py "${SAPLMA_COMMON[@]}" \
      --eval_dataset "$ds" --ood_setting "$setting" --output_file "$saplma_file" --max_new_tokens "${MAX_NEW_TOKENS[$ds]}"

    # Step 2: extract MD (SATMD) predictions.
    $PYTHON run_polygraph.py "${SATMD_COMMON[@]}" \
      --eval_dataset "$ds" --ood_setting "$setting" --md_save_file "$md_prefix" --max_new_tokens "${MAX_NEW_TOKENS[$ds]}"

    # Step 3: run HBO.
    $PYTHON run_hbo.py \
      --msp_file "$msp_file" \
      --probe_file "$saplma_file" \
      --md_save_prefix "$md_prefix" \
      --layer_index "$LAYER_INDEX" \
      --output "$OUT/hbo_${ds}_${setting}.jsonl"
  done
done
