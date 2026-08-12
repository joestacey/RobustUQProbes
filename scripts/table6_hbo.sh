#!/usr/bin/env bash
# Table 6 — HBO, Llama-3.1-8B. ID setting, all 6 Table 1 eval datasets.
# See scripts/README.md's "Table 6 / HBO" section for the full recipe.
#
# Set HF_HUB_TOKEN in your environment before running if models are gated.

set -e
PYTHON="${PYTHON:-python}"
OUT=hbo_outputs
mkdir -p "$OUT"

MSP_COMMON=(
  use_seq_ue=True
  use_density_based_ue=False
  batch_size=1
  subsample_eval_dataset=2000
  model.path=meta-llama/Meta-Llama-3.1-8B
  +model.attn_implementation=eager
  +method=msp
  +loadin4bit=False
)

SAPLMA_COMMON=(
  ignore_exceptions=False
  use_density_based_ue=True
  batch_size=1
  subsample_eval_dataset=2000
  model.path=meta-llama/Meta-Llama-3.1-8B
  +model.attn_implementation=eager
  '+metric_thrs=[0.3]'
  '+layers=[-1]'
  +clean_md_device=cuda
  +loadin4bit=False
  +method=feature_supervision
  +probe.head_type=full_sequence_saplma
  +probe.feature_extractor_setting=hs_middle
  +token_aggregation=average
  +pre_compile_features=True
)

SATMD_COMMON=(
  ignore_exceptions=False
  use_density_based_ue=True
  batch_size=1
  subsample_eval_dataset=2000
  subsample_background_train_dataset=1
  model.path=meta-llama/Meta-Llama-3.1-8B
  +model.attn_implementation=eager
  '+metric_thrs=[0.3]'
  '+layers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,-1]'
  +clean_md_device=cuda
  +loadin4bit=False
  +method=satmd
  +token_aggregation=average
  +pre_compile_features=True
)

# SATMD_COMMON's +layers list has 32 entries (indices 0-31); index 15 is the
# "middle" layer (math.ceil(32/2)-1), matching SAPLMA_COMMON's hs_middle
# feature_extractor_setting. Keep these in sync if either COMMON block's
# layer list changes.
LAYER_INDEX=15

declare -A CONFIGS=(
  [sciq]=configs/polygraph_eval_sciq.yaml
  [triviaqa]=configs/polygraph_eval_triviaqa.yaml
  [coqa]=configs/polygraph_eval_coqa.yaml
  [pubmedqa]=configs/polygraph_eval_pubmedqa.yaml
  [xsum]=configs/polygraph_eval_xsum.yaml
  [cnn]=configs/polygraph_eval_cnn.yaml
)

for dataset in sciq triviaqa coqa pubmedqa xsum cnn; do
  cfg="${CONFIGS[$dataset]}"

  # MSP: one run per dataset, no training, reused across all OOD settings.
  msp_file="$OUT/msp_${dataset}.jsonl"
  if [ ! -f "$msp_file" ]; then
    HYDRA_CONFIG="$cfg" $PYTHON run_polygraph.py "${MSP_COMMON[@]}" +output_file="$msp_file"
  fi

  # ─── ID ────────────────────────────────────────────────────────────────
  saplma_file="$OUT/saplma_${dataset}_id.jsonl"
  md_prefix="$OUT/satmd_${dataset}_id"

  HYDRA_CONFIG="$cfg" $PYTHON run_polygraph.py "${SAPLMA_COMMON[@]}" \
    subsample_train_dataset=1800 +output_file="$saplma_file"

  HYDRA_CONFIG="$cfg" $PYTHON run_polygraph.py "${SATMD_COMMON[@]}" \
    subsample_train_dataset=1800 +md_save_file="$md_prefix"

  $PYTHON run_hbo.py \
    --msp_file "$msp_file" \
    --probe_file "$saplma_file" \
    --md_save_prefix "$md_prefix" \
    --layer_index "$LAYER_INDEX" \
    --output "$OUT/hbo_${dataset}_id.jsonl"
done
