# scripts/

## Table 1 — baselines

| Script | Method |
|---|---|
| `table1_msp.sh` | MSP |
| `table1_satmd.sh` | Token Mahalanobis distance |
| `table1_satrmd.sh` | Relative token Mahalanobis distance |
| `table1_satmd_msp.sh` | MD + MSP hybrid |
| `table1_satrmd_msp.sh` | Relative MD + MSP hybrid |
| `table1_huq_satmd.sh` | HUQ (MD + MSP) |
| `table1_huq_satrmd.sh` | HUQ (relative MD + MSP) |
| `table1_saplma_middle.sh` | SAPLMA probe, middle hidden layer |
| `table1_saplma_last.sh` | SAPLMA probe, last hidden layer |

Each loops over `--eval_dataset` in `{sciq, trivia_qa, qa, pubmed_qa, xsum, cnn_dailymail}` and `--ood_setting` in `{ID, OOD_LEAVE_ONE_OUT, OOD_ONE_DATASET_SAME_TASK, OOD_DIFF_TASK, OOD_ONE_DATASET_DIFF_TASK}` (`table1_msp.sh` only loops over `--eval_dataset`).

```bash
bash scripts/table1_msp.sh
```

## Tables 3-5 — ablations

| Script | Sweeps |
|---|---|
| `feature_ablations.sh` | `--probe_feature_extractor_setting` in `{att_2, att_1, token_probabilities, lookback_lens}` |
| `aggregation_ablations.sh` | `--token_aggregation` in `{context_average, average_all, last_token, last_context_token, ablation_random_keep}` |
| `architecture_ablations.sh` | `--probe_head_type` in `{full_sequence_linear_regression, full_sequence_uhead (v1), full_sequence_uhead (v2)}` |

```bash
bash scripts/feature_ablations.sh
```

## Table 6 — HBO

```bash
bash scripts/table6_hbo.sh
```

Manual recipe:

```bash
python run_polygraph.py --method msp --eval_dataset sciq \
    --model_path meta-llama/Meta-Llama-3.1-8B --attn_implementation eager \
    --max_new_tokens 20 --output_file hbo_outputs/msp_sciq.jsonl

python run_polygraph.py --method feature_supervision \
    --probe_head_type full_sequence_saplma \
    --probe_feature_extractor_setting hs_middle \
    --token_aggregation average \
    --eval_dataset sciq --ood_setting ID \
    --model_path meta-llama/Meta-Llama-3.1-8B --attn_implementation eager \
    --clean_md_device cuda --max_new_tokens 20 \
    --output_file hbo_outputs/saplma_sciq_ID.jsonl

python run_polygraph.py --method satmd \
    --eval_dataset sciq --ood_setting ID \
    --model_path meta-llama/Meta-Llama-3.1-8B --attn_implementation eager \
    --metric_thr 0.3 --clean_md_device cuda --max_new_tokens 20 \
    --md_save_file hbo_outputs/satmd_sciq_ID

python run_hbo.py \
    --msp_file    hbo_outputs/msp_sciq.jsonl \
    --probe_file  hbo_outputs/saplma_sciq_ID.jsonl \
    --md_save_prefix hbo_outputs/satmd_sciq_ID \
    --layer_index 15 \
    --output      hbo_outputs/hbo_sciq_ID.jsonl
```
