"""
Shared infrastructure for run_polygraph.py and collect_llm_judge_inputs.py.
"""

import codecs
import os
from pathlib import Path

from lm_polygraph_lite.utils.model import WhiteboxModel
from lm_polygraph_lite.utils.generation_parameters import GenerationParameters
from lm_polygraph_lite.utils.common import load_external_module
from utils.dataset import Dataset


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


def get_abs_path(hydra_config: Path, path: str) -> Path:
    path = Path(path)
    if not os.path.isabs(path):
        path = hydra_config.parent / path
    return path


def load_model(args, hydra_config: Path, cache_kwargs={}):
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

    path_to_load_script = get_abs_path(hydra_config, args.model.path_to_load_script)
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


def load_dataset(args, split: str, cache_kwargs={}):
    """
    Load either the eval split or the train split of the dataset.
    split must be 'eval' or 'train'.
    """
    assert split in ('eval', 'train')
    seed = 1
    common = dict(
        batch_size=args.batch_size,
        prompt=args.prompt,
        description=getattr(args, "description", ""),
        mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
        n_shot=getattr(args, "n_shot", 5),
        few_shot_split=getattr(args, "few_shot_split", "train"),
        instruct=getattr(args, "instruct", False),
        few_shot_prompt=getattr(args, "few_shot_prompt", None),
        load_from_disk=args.load_from_disk,
        max_new_tokens=getattr(args, "max_new_tokens", 100),
        **cache_kwargs,
    )

    if split == 'eval':
        dataset = Dataset.load(args.dataset, args.text_column, args.label_column, split=args.eval_split, **common)
        if args.subsample_eval_dataset != -1:
            dataset.subsample(args.subsample_eval_dataset, seed=seed)
    else:
        dataset_name = (
            args.train_dataset
            if (args.train_dataset is not None and args.train_dataset != args.dataset)
            else args.dataset
        )
        dataset = Dataset.load(dataset_name, args.text_column, args.label_column, split=args.train_split, size=10_000, **common)
        if args.subsample_train_dataset != -1:
            dataset.subsample(args.subsample_train_dataset, seed=seed)

    return dataset


def load_multi_train_dataset(args, cache_kwargs={}):
    """
    Loads and concatenates train_dataset_1, train_dataset_2, ... (LOO / DiffTask
    OOD settings). Each sub-dataset is independently subsampled to
    subsample_train_dataset (not divided across datasets) before concatenation.
    """
    seed = 1
    k_ds = 1
    dataset = None
    while getattr(args, f"train_dataset_{k_ds}", False):
        dataset_k = Dataset.load(
            getattr(args, f"train_dataset_{k_ds}"),
            getattr(args, f"train_text_column_{k_ds}"),
            getattr(args, f"train_label_column_{k_ds}"),
            batch_size=args.batch_size,
            prompt=codecs.decode(getattr(args, f"train_prompt_{k_ds}"), "unicode_escape"),
            description=codecs.decode(getattr(args, f"train_description_{k_ds}", ""), "unicode_escape"),
            mmlu_max_subject_size=getattr(args, "mmlu_max_subject_size", 100),
            n_shot=getattr(args, f"train_n_shot_{k_ds}", 5),
            few_shot_split=getattr(args, f"few_shot_split_{k_ds}", "train"),
            split=getattr(args, f"train_split_{k_ds}", "train"),
            max_new_tokens=getattr(args, f"max_new_tokens_{k_ds}", 100),
            size=10_000,
            instruct=getattr(args, "instruct", False),
            few_shot_prompt=getattr(args, f"few_shot_prompt_{k_ds}", None),
            load_from_disk=args.load_from_disk,
            **cache_kwargs,
        )
        k_ds += 1

        if args.subsample_train_dataset != -1:
            dataset_k.subsample(args.subsample_train_dataset, seed=seed)

        if dataset is None:
            dataset = dataset_k
        else:
            dataset.concat(dataset_k.x, dataset_k.y, dataset_k.max_new_tokens)

    return dataset
