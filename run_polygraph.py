#!/usr/bin/env python3

import hydra
import os
import re
import transformers
import logging
from pathlib import Path

log = logging.getLogger()

from omegaconf import OmegaConf
from luh import AutoUncertaintyHead

from lm_polygraph_lite.utils.manager import UEManager
from utils.alignscore import AlignScore as AlignScoreNew
from lm_polygraph_lite.utils.processor import Logger
from lm_polygraph_lite.generation_metrics.accuracy import AccuracyMetric
from lm_polygraph_lite.generation_metrics.rouge import RougeMetric
from lm_polygraph_lite.generation_metrics.alignscore import AlignScore
from lm_polygraph_lite.generation_metrics.llm_judge_metric import LLMJudgeMetric
from lm_polygraph_lite.generation_metrics.aggregated_metric import AggregatedMetric
from lm_polygraph_lite.estimators import MaximumSequenceProbability
from lm_polygraph_lite.ue_metrics import PredictionRejectionArea
from luh.feature_supervision import FeatureSupervision
from satmd_baseline.average_token_mahalanobis_distance import LinRegTokenMahalanobisDistance
from satmd_baseline.average_token_mahalanobis_distance_hybrid import LinRegTokenMahalanobisDistance_Hybrid
from satmd_baseline.huq_msp_lrtmd import HUQ_LRTMD
from utils.pipeline import get_cache_kwargs, load_model, load_dataset, load_multi_train_dataset
from utils.dataset import Dataset

import nltk
nltk.download('punkt_tab')

hydra_config = Path(os.environ["HYDRA_CONFIG"])


_FEATURE_EXTRACTORS = {
    'hs':            [{'name': 'luh.feature_extractors.basic_hidden_states', 'layer_nums': -1}],
    'hs_middle':     [{'name': 'luh.feature_extractors.basic_hidden_states', 'layer_nums': 'middle'}],
    'token_probs':   [{'name': 'luh.feature_extractors.token_probabilities', 'top_n': 4}],
    'uhead':         [{'name': 'luh.feature_extractors.token_probabilities', 'top_n': 4}, {'name': 'luh.feature_extractors.basic_attention', 'layer_nums': 'all', 'attn_history_sz': 2, 'pool': False, 'offset': 0}],
    'lookback_lens': [{'name': 'luh.feature_extractors.lookback_lens', 'pool': False, 'pool_type': 'mean', 'layer_nums': 'all'}],
    'token_probabilities': [{'name': 'luh.feature_extractors.token_probabilities', 'top_n': 4}],
}




@hydra.main(
    version_base=None,
    config_path=str(hydra_config.parent),
    config_name=str(hydra_config.name),
)
def main(args):
    save_path = os.getcwd()
    log.info(f"Main directory: {save_path}")
    os.chdir(hydra.utils.get_original_cwd())

    save_path = args.save_path if "save_path" in args else save_path

    seed = 1

    cache_kwargs = get_cache_kwargs(args)

    log.info("=" * 100)
    log.info(f"Loading model {args.model.path}...")
    transformers.set_seed(seed)

    model = load_model(args, hydra_config, cache_kwargs)

    log.info("Done with loading model.")
    log.info(f"Loading dataset {args.dataset}...")

    dataset = load_dataset(args, 'eval', cache_kwargs)

    estimator = get_estimator(args, model)

    train_dataset = None
    background_train_dataset = None

    if not getattr(estimator, "is_fitted", True):
        if args.train_test_split:
            X_train, _, y_train, _, _, _ = dataset.train_test_split(
                test_size=args.test_split_size, seed=seed, split=args.eval_split
            )
            train_dataset = Dataset(
                x=X_train, y=y_train, max_new_tokens=getattr(args, "max_new_tokens", 100), batch_size=args.batch_size
            )
        else:
            train_dataset = load_dataset(args, 'train', cache_kwargs)

        if getattr(args, "train_dataset_1", False):
            train_dataset = load_multi_train_dataset(args, cache_kwargs)

    needs_background_data = any(
        dep.startswith("background_train") for dep in getattr(estimator, "stats_dependencies", [])
    )

    if not getattr(estimator, "is_fitted", False) and needs_background_data:
        background_train_dataset = Dataset.load(
            args.background_train_dataset,
            args.background_train_dataset_text_column,
            args.background_train_dataset_label_column,
            batch_size=args.batch_size,
            data_files=args.background_train_dataset_data_files,
            split="train",
            size=100_000,
            instruct=getattr(args, "instruct", False),
            few_shot_prompt=getattr(args, "few_shot_prompt", None),
            load_from_disk=args.background_load_from_disk,
            **cache_kwargs
        )

        if args.subsample_background_train_dataset != -1:
            background_train_dataset.subsample(
                args.subsample_background_train_dataset, seed=seed
            )

    log.info("Done with loading data.")

    man = UEManager(
        dataset,
        model,
        estimator,
        get_generation_metrics(args),
        [PredictionRejectionArea()],
        [Logger()],
        train_data=train_dataset,
        ignore_exceptions=args.ignore_exceptions,
        background_train_data=background_train_dataset,
        max_new_tokens=args.max_new_tokens,
        output_file=getattr(args, 'output_file', None),
        md_save_file=getattr(args, 'md_save_file', None),
    )

    results_dict = man()
    log.info("\n\nResults from experiment: " + str(results_dict))


_MD_METHODS = ('satmd', 'satrmd', 'satmd_msp', 'satrmd_msp', 'huq_satmd', 'huq_satrmd')


def get_estimator(args, model):
    if args.use_seq_ue and args.method == 'msp':
        return MaximumSequenceProbability()
    if args.use_density_based_ue and args.method == 'feature_supervision':
        return _build_feature_supervision(args, model)
    if args.use_density_based_ue and args.method in _MD_METHODS:
        return _build_md_estimator(args)
    raise ValueError(
        f"No estimator configured: method={args.method!r}, "
        f"use_seq_ue={args.use_seq_ue}, use_density_based_ue={args.use_density_based_ue}"
    )


def _build_feature_supervision(args, model):
    if getattr(args, "llm_judge_file", None):
        metric = LLMJudgeMetric(args.llm_judge_file)
        metric_name = "LLMJudge"
    else:
        metric = AlignScoreNew(return_mean=True, batch_size=1)
        metric_name = "AlignScore"

    setting = args.probe.feature_extractor_setting
    if re.fullmatch(r'hs_\d+', setting):
        feature_extractors = [{'name': 'luh.feature_extractors.basic_hidden_states', 'layer_nums': int(setting.split('_')[1])}]
    elif re.fullmatch(r'att_\d+', setting):
        feature_extractors = [{'name': 'luh.feature_extractors.basic_attention', 'layer_nums': 'all', 'attn_history_sz': int(setting.split('_')[1]), 'pool': False, 'offset': 0}]
    elif setting in _FEATURE_EXTRACTORS:
        feature_extractors = _FEATURE_EXTRACTORS[setting]
    else:
        raise ValueError(f"Unknown feature_extractor_setting: {setting!r}")

    cfg = {
        'head_type': args.probe.head_type,
        'feature_extractor': feature_extractors,
    }

    if args.probe.head_type == 'full_sequence_uhead':
        cfg['uncertainty_head'] = {
            'head_dim': args.probe.head_dim,
            'n_layers': args.probe.n_layers,
            'n_heads': args.probe.n_heads,
            'dropout': args.probe.dropout,
            'weight_decay': args.probe.weight_decay,
            'learning_rate': args.probe.learning_rate,
            'warmup_ratio': args.probe.warmup_ratio,
            'num_train_epochs': args.probe.num_train_epochs,
            'train_batch_size': args.probe.train_batch_size,
        }
    elif args.probe.head_type in ('full_sequence_linear', 'full_sequence_saplma'):
        cfg['uncertainty_head'] = None
    elif args.probe.head_type == 'full_sequence_linear_regression':
        cfg['uncertainty_head'] = None

    cfg['token_aggregation'] = args.token_aggregation

    head = AutoUncertaintyHead.from_config(OmegaConf.create(cfg), base_model=model.model)
    base_model = model if args.probe.head_type == 'full_sequence_uhead' else None

    return FeatureSupervision(
        device='cuda',
        storage_device=getattr(args, "clean_md_device", "cpu"),
        metric_name=metric_name,
        metric=metric,
        aggregated=getattr(args, "multiref", False),
        config=cfg,
        head=head,
        pre_compile_features=args.pre_compile_features,
        base_model=base_model,
    )


def _get_md_metric(args):
    if getattr(args, "llm_judge_file", None):
        return LLMJudgeMetric(args.llm_judge_file), "LLMJudge"
    return AlignScoreNew(return_mean=True, batch_size=1), "AlignScore"


def _build_md_estimator(args):
    metric, metric_name = _get_md_metric(args)
    layers = list(getattr(args, "layers", [0, -1]))
    metric_thr = list(getattr(args, "metric_thrs", [0.3]))[0]
    storage_device = getattr(args, "clean_md_device", "cpu")
    ue_type = (
        "TokenMahalanobis"
        if args.method in ('satmd', 'satmd_msp', 'huq_satmd')
        else "RelativeTokenMahalanobis"
    )
    kwargs = dict(
        embeddings_type="decoder",
        metric=metric,
        metric_name=metric_name,
        metric_md=metric,
        metric_md_name=metric_name,
        aggregated=getattr(args, "multiref", False),
        hidden_layers=layers,
        metric_thr=metric_thr,
        ue=ue_type,
        positive=False,
        meta_model="LinReg",
        norm="orig",
        remove_corr=True,
        remove_alg=3,
        storage_device=storage_device,
    )
    if args.method in ('satmd', 'satrmd'):
        return LinRegTokenMahalanobisDistance(**kwargs)
    if args.method in ('satmd_msp', 'satrmd_msp'):
        return LinRegTokenMahalanobisDistance_Hybrid(**kwargs)
    if args.method in ('huq_satmd', 'huq_satrmd'):
        return HUQ_LRTMD(**kwargs)


def get_generation_metrics(args):
    if (args.task == "qa") and (args.dataset not in ["keivalya/MedQuad-MedicalQnADataset", "bigbio/pubmed_qa", ['truthful_qa', 'generation']]):
        alignscorer = AlignScore(batch_size=1)
    else:
        alignscorer = AlignScoreNew(return_mean=True, batch_size=1)
    result = [
        RougeMetric("rougeL"),
        AccuracyMetric(
            target_ignore_regex=getattr(args, "target_ignore_regex", None),
            output_ignore_regex=getattr(args, "output_ignore_regex", None),
            normalize=getattr(args, "normalize", False),
        ),
        alignscorer,
    ]
    if getattr(args, "multiref", False):
        result = [AggregatedMetric(base_metric=metric) for metric in result]
    return result


if __name__ == "__main__":
    main()
