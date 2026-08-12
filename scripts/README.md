# scripts/

Reproduction scripts for the paper's tables. All scripts run `run_polygraph.py`
against `meta-llama/Meta-Llama-3.1-8B` (set `HF_HUB_TOKEN` in your environment
first if the model is gated).

## Table 1 — baselines

| Script | Method |
|---|---|
| `table1_msp.sh` | MSP (no training step) |
| `table1_satmd.sh` / `table1_satrmd.sh` | raw / relative token Mahalanobis distance |
| `table1_satmd_msp.sh` / `table1_satrmd_msp.sh` | MD + MSP/entropy hybrid |
| `table1_huq_satmd.sh` / `table1_huq_satrmd.sh` | HUQ (MD + MSP, tuned combination) |
| `table1_saplma_middle.sh` / `table1_saplma_last.sh` | SAPLMA probe, middle/last hidden layer |

Each (except `table1_msp.sh`, which has no training step) implements all 5
OOD settings:

- **ID** — train on the same dataset, `subsample_train_dataset=1800`.
- **LOO** — leave-one-out: the other 9 datasets, 200 examples each.
- **1D-SameTask** — single same-task dataset (MedQuAD for QA evals, SAMSum
  for summarization evals), 1800 examples.
- **DiffTask** — multiple different-task datasets (SAMSum+XSum+CNN for QA
  evals, sciq+mmlu+trivia_qa+coqa+truthful_qa+med_quad for summarization
  evals), 300-600 each.
- **1D-DiffTask** — single different-task dataset, 1800 examples.

## Tables 3-5 — ablations

- `feature_ablations.sh` — att_2 / att_1 / token_probabilities / lookback_lens
  feature extractors (SAPLMA, ID setting).
- `aggregation_ablations.sh` — context_average / average_all / last_token /
  last_context_token / ablation_random_keep token-aggregation strategies
  (SAPLMA, hs_middle, ID setting).
- `architecture_ablations.sh` — `full_sequence_linear_regression` (lin_reg)
  and two `full_sequence_uhead` configs (uhead_v1, uhead_v2), all ID setting.

## Table 6 — HBO

`table6_hbo.sh` combines MSP (aleatoric), a supervised SAPLMA-middle probe,
and raw SATMD (epistemic), where SATMD's percentile rank sets the per-example
weighting between MSP and the probe.

Three separate `run_polygraph.py` runs per (dataset, OOD setting) — MSP,
SAPLMA-middle, and SATMD — must evaluate the same eval examples in the same
order; `run_polygraph.py`'s eval subsample uses a fixed `seed=1`, so this
holds automatically given the same `HYDRA_CONFIG` and
`subsample_eval_dataset`. `run_hbo.py` combines the three saved outputs:

1. **MSP** (`+method=msp`, no training) — run once per dataset, reused
   across all 5 OOD settings, with `+output_file=PATH.jsonl`.
2. **SAPLMA-middle** (`+method=feature_supervision
   +probe.head_type=full_sequence_saplma
   +probe.feature_extractor_setting=hs_middle`) — one run per (dataset, OOD
   setting), same `+output_file=PATH.jsonl` mechanism.
3. **SATMD** (`+method=satmd`, all layers) — one run per (dataset, OOD
   setting), with `+md_save_file=PREFIX`. Produces `PREFIX.jsonl` (per-layer
   mean MD values per eval example) and `PREFIX_train.npy` (dev-split MD
   matrix used as the percentile reference).

Combination (`run_hbo.py --msp_file ... --probe_file ... --md_save_prefix
... --layer_index 15 --output ...`):

1. Each test example's SATMD percentile rank against the dev-split
   distribution, at `--layer_index` (default -1/last; `table6_hbo.sh` uses
   15, the middle layer, to match SAPLMA-middle).
2. `msp_weight = min(1.0, percentile + 0.5)`; `probe_weight = 1 - msp_weight`.
3. Rank-transform the MSP and SAPLMA scores separately across the eval set.
4. `final_score = msp_weight * msp_rank + probe_weight * probe_rank`.

`table6_hbo.sh` covers the ID setting for all 6 Table 1 eval datasets. To
extend to LOO / 1D-SameTask / DiffTask / 1D-DiffTask, copy the matching
`train_dataset_N` override block from `table1_saplma_middle.sh` (SAPLMA run)
and `table1_satmd.sh` (SATMD run) for that dataset+setting; the MSP run does
not change across OOD settings.
