# RobustUQProbes

Code for the paper [**Hidden Failures in Robustness: Why Supervised Uncertainty Quantification Needs Better Evaluation**](https://arxiv.org/pdf/2604.11662).

The [**ProbeDrift**](https://github.com/joestacey/ProbeDrift) evaluation framework introduced in this paper lives in an external repo containing all of the dataset loading and evaluation splits.

This implementation adapts code from [**token_mahalanobis_distance**](https://github.com/ArtemVazh/token_mahalanobis_distance/tree/main) and [**llm-uncertainty-head**](https://github.com/IINemo/llm-uncertainty-head).

---

## Installation

```bash
pip install -r requirements.txt
pip install git+https://github.com/joestacey/ProbeDrift.git
```

The main model (`meta-llama/Meta-Llama-3.1-8B`) is gated. Set your token before running:

```bash
export HF_HUB_TOKEN=your_token_here
```

---

## Running the baselines (Table 1 and ablations)

All scripts live in `scripts/` and loop over all 6 eval datasets and all 5 OOD settings automatically. Activate your environment, then run the relevant script:

```bash
bash scripts/table1_msp.sh
bash scripts/table1_satmd.sh
bash scripts/table1_saplma_middle.sh
# etc. : see scripts/README.md for the full table
```

Results are written to `workdir/output/` by default.

---

## HBO (Table 6)

HBO combines three separately saved signals. `table6_hbo.sh` runs all steps end-to-end for the ID setting:

```bash
bash scripts/table6_hbo.sh
```

If you want to run the steps manually (e.g. to reuse an existing MSP run), the recipe is:

**Step 1 : MSP** (training-free; one run per dataset, reused across all OOD settings):
```bash
python run_polygraph.py --method msp --eval_dataset sciq \
    --model_path meta-llama/Meta-Llama-3.1-8B --attn_implementation eager \
    --max_new_tokens 20 --output_file hbo_outputs/msp_sciq.jsonl
```

**Step 2 : SAPLMA-middle** (one run per dataset + OOD setting):
```bash
python run_polygraph.py --method feature_supervision \
    --probe_head_type full_sequence_saplma \
    --probe_feature_extractor_setting hs_middle \
    --token_aggregation average \
    --eval_dataset sciq --ood_setting ID \
    --model_path meta-llama/Meta-Llama-3.1-8B --attn_implementation eager \
    --clean_md_device cuda --max_new_tokens 20 \
    --output_file hbo_outputs/saplma_sciq_ID.jsonl
```

**Step 3 : SATMD** (one run per dataset + OOD setting; produces `.jsonl` + `_train.npy`):
```bash
python run_polygraph.py --method satmd \
    --eval_dataset sciq --ood_setting ID \
    --model_path meta-llama/Meta-Llama-3.1-8B --attn_implementation eager \
    --metric_thr 0.3 --clean_md_device cuda --max_new_tokens 20 \
    --md_save_file hbo_outputs/satmd_sciq_ID
```

**Step 4 : combine:**
```bash
python run_hbo.py \
    --msp_file    hbo_outputs/msp_sciq.jsonl \
    --probe_file  hbo_outputs/saplma_sciq_ID.jsonl \
    --md_save_prefix hbo_outputs/satmd_sciq_ID \
    --layer_index 15 \
    --output      hbo_outputs/hbo_sciq_ID.jsonl
```

---

## LLM-as-judge

LLM-as-judge scores replace the automatic metric (ROUGE/accuracy) in `run_polygraph.py`. The workflow is:

**Step 1 : collect model outputs:**
```bash
python collect_llm_judge_inputs.py \
    --eval_dataset sciq --ood_setting ID --split eval \
    --model_path meta-llama/Meta-Llama-3.1-8B \
    --output_file judge_inputs/eval_sciq.json
```

**Step 2 : score with the LLM judge** (requires `OPENAI_API_KEY`):
```bash
export OPENAI_API_KEY=your_key_here
python run_llm_judge.py \
    --input_file  judge_inputs/eval_sciq.json \
    --output_file judge_scores/eval_sciq.json
```

**Step 3 : run the probe experiment using the judge scores:**
```bash
python run_polygraph.py --method feature_supervision ... \
    --eval_dataset sciq --ood_setting ID \
    --llm_judge_file judge_scores/eval_sciq.json
```

`--eval_dataset`, `--ood_setting`, `--instruct`, and `--model_path` must match between `collect_llm_judge_inputs.py` and `run_polygraph.py` so the examples align.

---

## Contact

Questions about the paper or this repo: j.stacey@sheffield.ac.uk
