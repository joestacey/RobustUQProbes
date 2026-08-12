#!/usr/bin/env bash
# Table 1 — MSP, Llama-3.1-8B
#
# MSP has no training step (no estimator fit), so there is no LOO /
# 1D-SameTask / DiffTask / 1D-DiffTask variation to run — just one eval
# per Table 1 eval dataset.
#
# Set HF_HUB_TOKEN in your environment before running if models are gated.

set -e
PYTHON="${PYTHON:-python}"

COMMON=(
  use_seq_ue=True
  use_density_based_ue=False
  batch_size=1
  subsample_eval_dataset=2000
  model.path=meta-llama/Meta-Llama-3.1-8B
  +model.attn_implementation=eager
  +method=msp
  +loadin4bit=False
)

HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml      $PYTHON run_polygraph.py "${COMMON[@]}"
HYDRA_CONFIG=configs/polygraph_eval_triviaqa.yaml  $PYTHON run_polygraph.py "${COMMON[@]}"
HYDRA_CONFIG=configs/polygraph_eval_coqa.yaml      $PYTHON run_polygraph.py "${COMMON[@]}"
HYDRA_CONFIG=configs/polygraph_eval_pubmedqa.yaml  $PYTHON run_polygraph.py "${COMMON[@]}"
HYDRA_CONFIG=configs/polygraph_eval_xsum.yaml      $PYTHON run_polygraph.py "${COMMON[@]}"
HYDRA_CONFIG=configs/polygraph_eval_cnn.yaml       $PYTHON run_polygraph.py "${COMMON[@]}"
