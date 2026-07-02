"""
Shared infrastructure for run_polygraph.py and collect_llm_judge_inputs.py.
"""

import os
from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from probe_drift.dataset import Dataset as ProbeDriftDataset

from lm_polygraph_lite.utils.model import WhiteboxModel
from lm_polygraph_lite.utils.generation_parameters import GenerationParameters
from lm_polygraph_lite.utils.common import load_external_module
from probe_drift import get_datasets as probe_drift_get_datasets


def get_cache_kwargs(args):
    token = os.getenv("HF_HUB_TOKEN")
    if os.environ.get('HF_DATASETS_OFFLINE', '').strip() == '1':
        return {'cache_dir': args.cache_path}
    return {'token': token}


def get_model_kwargs(args):
    model_kwargs = {}
    if getattr(args.model, 'attn_implementation', None):
        model_kwargs['attn_implementation'] = args.model.attn_implementation
    return model_kwargs


def get_abs_path(anchor: Path, path: str) -> Path:
    """
    Resolves `path` relative to anchor.parent, if `path` isn't already absolute.
    """
    path = Path(path)
    if not os.path.isabs(path):
        path = anchor.parent / path
    return path


def load_model(args, anchor: Path, cache_kwargs={}):
    if "path_to_load_script" not in args.model:
        model_kwargs = get_model_kwargs(args)
        extra = {'load_in_4bit': True} if args.loadin4bit else {}
        return WhiteboxModel.from_pretrained(
            args.model.path,
            getattr(args, "generation_params", {}),
            device_map=args.model.device_map,
            add_bos_token=getattr(args.model, "add_bos_token", True),
            **cache_kwargs,
            **model_kwargs,
            **extra,
        )

    path_to_load_script = get_abs_path(anchor, args.model.path_to_load_script)
    load_module = load_external_module(path_to_load_script)

    load_model_args = {'model_path': args.model.path}
    load_model_args.update(args.model.load_model_args)
    if args.loadin4bit:
        load_model_args['load_in_4bit'] = True
    base_model = load_module.load_model(**load_model_args)

    load_tok_args = {'model_path': args.model.path}
    load_tok_args.update(args.model.load_tokenizer_args)
    tokenizer = load_module.load_tokenizer(**load_tok_args)

    generation_params = GenerationParameters(**getattr(args, "generation_params", {}))
    return WhiteboxModel(base_model, tokenizer, args.model.path, args.model.type, generation_params)


def load_background_dataset(
    dataset_path, text_column, label_column, batch_size, data_files,
    size, max_new_tokens, cache_kwargs={},
):
    """Loads the C4 background dataset used by SATRMD-family MD methods."""
    dataset = hf_load_dataset(
        dataset_path, data_files=data_files, split="train",
        trust_remote_code=True, **cache_kwargs,
    )
    if size is not None and size < len(dataset):
        dataset = dataset.select(range(size))
    return ProbeDriftDataset(dataset[text_column], dataset[label_column], batch_size)


def load_datasets_via_probe_drift(args):
    """Loads (train_dataset, eval_dataset) via ProbeDrift."""
    ood_setting = getattr(args, "ood_setting", "ID")
    train_ds, eval_ds = probe_drift_get_datasets(
        eval_dataset=args.eval_dataset,
        ood_setting=ood_setting,
        instruct=getattr(args, "instruct", False),
        batch_size=args.batch_size,
    )
    return train_ds, eval_ds
