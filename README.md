# RobustUQProbes

Code for the paper [**Hidden Failures in Robustness: Why Supervised Uncertainty Quantification Needs Better Evaluation**](https://arxiv.org/pdf/2604.11662).

This implementation adapts code from [**token_mahalanobis_distance**](https://github.com/ArtemVazh/token_mahalanobis_distance/tree/main) and [**llm-uncertainty-head**](https://github.com/IINemo/llm-uncertainty-head).

This branch replicates the code used to run the paper experiments, with any differences to the main branch (updated code) explained in [differences_to_main.txt](differences_to_main.txt).

---

## Installation

```bash
pip install -r requirements.txt
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
HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python run_polygraph.py \
    use_seq_ue=True use_density_based_ue=False batch_size=1 \
    subsample_eval_dataset=2000 \
    model.path=meta-llama/Meta-Llama-3.1-8B +model.attn_implementation=eager \
    +method=msp +loadin4bit=False \
    +output_file=hbo_outputs/msp_sciq.jsonl
```

**Step 2 : SAPLMA-middle** (one run per dataset + OOD setting):
```bash
HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python run_polygraph.py \
    use_density_based_ue=True batch_size=1 \
    subsample_train_dataset=1800 subsample_eval_dataset=2000 \
    model.path=meta-llama/Meta-Llama-3.1-8B +model.attn_implementation=eager \
    +metric_thrs='[0.3]' +layers='[-1]' \
    +method=feature_supervision +probe.head_type=full_sequence_saplma \
    +probe.feature_extractor_setting=hs_middle +token_aggregation=average \
    +pre_compile_features=True +loadin4bit=False \
    +output_file=hbo_outputs/saplma_sciq_ID.jsonl
```

**Step 3 : SATMD** (one run per dataset + OOD setting; produces `.jsonl` + `_train.npy`):
```bash
HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python run_polygraph.py \
    use_density_based_ue=True batch_size=1 \
    subsample_train_dataset=1800 subsample_eval_dataset=2000 \
    subsample_background_train_dataset=1 \
    model.path=meta-llama/Meta-Llama-3.1-8B +model.attn_implementation=eager \
    +metric_thrs='[0.3]' +layers='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,-1]' \
    +clean_md_device=cuda \
    +method=satmd +token_aggregation=average +pre_compile_features=True +loadin4bit=False \
    +md_save_file=hbo_outputs/satmd_sciq_ID
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
HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python collect_llm_judge_inputs.py \
    +split=eval +output_file=judge_inputs/eval_sciq.json
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
HYDRA_CONFIG=configs/polygraph_eval_sciq.yaml python run_polygraph.py \
    +method=feature_supervision ... \
    +llm_judge_file=judge_scores/eval_sciq.json
```

`collect_llm_judge_inputs.py` must be run with the same `HYDRA_CONFIG` and overrides as the corresponding `run_polygraph.py` run, so the examples align.

---

## Contact

Questions about the paper or this repo: j.stacey@sheffield.ac.uk
