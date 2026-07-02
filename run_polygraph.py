#!/usr/bin/env python3

import argparse
import re
import logging
from pathlib import Path

import transformers
from omegaconf import OmegaConf

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

from luh import AutoUncertaintyHead

from lm_polygraph_lite.utils.manager import UEManager
from utils.alignscore import AlignScore as AlignScoreNew
from lm_polygraph_lite.utils.processor import Logger
from lm_polygraph_lite.generation_metrics.accuracy import AccuracyMetric
from lm_polygraph_lite.generation_metrics.rouge import RougeMetric
from lm_polygraph_lite.generation_metrics.llm_judge_metric import LLMJudgeMetric
from lm_polygraph_lite.generation_metrics.aggregated_metric import AggregatedMetric
from lm_polygraph_lite.estimators import MaximumSequenceProbability
from lm_polygraph_lite.ue_metrics import PredictionRejectionArea
from luh.feature_supervision import FeatureSupervision
from satmd_baseline.average_token_mahalanobis_distance import LinRegTokenMahalanobisDistance
from satmd_baseline.average_token_mahalanobis_distance_hybrid import LinRegTokenMahalanobisDistance_Hybrid
from satmd_baseline.huq_msp_lrtmd import HUQ_LRTMD
from utils.pipeline import get_cache_kwargs, load_model, load_datasets_via_probe_drift, load_background_dataset

import nltk
nltk.download('punkt_tab')


_FEATURE_EXTRACTORS = {
    'hs':            [{'name': 'luh.feature_extractors.basic_hidden_states', 'layer_nums': -1}],
    'hs_middle':     [{'name': 'luh.feature_extractors.basic_hidden_states', 'layer_nums': 'middle'}],

    'uhead':         [{'name': 'luh.feature_extractors.token_probabilities', 'top_n': 4}, {'name': 'luh.feature_extractors.basic_attention', 'layer_nums': 'all', 'attn_history_sz': 2, 'pool': False, 'offset': 0}],
    'lookback_lens': [{'name': 'luh.feature_extractors.lookback_lens', 'pool': False, 'pool_type': 'mean', 'layer_nums': 'all'}],
    'token_probabilities': [{'name': 'luh.feature_extractors.token_probabilities', 'top_n': 4}],
}

_MD_METHODS = ('satmd', 'satrmd', 'satmd_msp', 'satrmd_msp', 'huq_satmd', 'huq_satrmd')
_SEQ_UE_METHODS = ('msp',)

_GENERATE_UNTIL = ["\n"]

# Background corpus for SATRMD-family Mahalanobis statistics.
_BACKGROUND_TRAIN_DATASET = "allenai/c4"
_BACKGROUND_TRAIN_DATASET_TEXT_COLUMN = "text"
_BACKGROUND_TRAIN_DATASET_LABEL_COLUMN = "url"
_BACKGROUND_TRAIN_DATASET_DATA_FILES = "en/c4-train.00000-of-01024.json.gz"

# trivia_qa is the only dataset with multiple reference answers per example.
_MULTIREF_DATASETS = {"trivia_qa"}

# AlignScore directionality: short-form QA scored unidirectionally, long-form
# scored bidirectionally.
_UNIDIRECTIONAL_ALIGNSCORE_DATASETS = {"sciq", "trivia_qa", "qa", "mmlu"}
_BIDIRECTIONAL_ALIGNSCORE_DATASETS = {
    "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "truthful_qa", "samsum",
}


def _alignscore_is_bidirectional(dataset_id):
    """Raises if dataset_id isn't in either directionality set."""
    base = dataset_id[: -len("_instruct")] if dataset_id.endswith("_instruct") else dataset_id
    if base in _UNIDIRECTIONAL_ALIGNSCORE_DATASETS:
        return False
    if base in _BIDIRECTIONAL_ALIGNSCORE_DATASETS:
        return True
    raise ValueError(
        f"Unknown dataset {dataset_id!r} has no AlignScore directionality classification. "
        "Add it to _UNIDIRECTIONAL_ALIGNSCORE_DATASETS or _BIDIRECTIONAL_ALIGNSCORE_DATASETS "
        "in run_polygraph.py before using it as a training or eval source."
    )


def _make_alignscore_router(aggregated):
    """Returns the correctly-directioned AlignScore metric per source dataset id.
    Each direction's AlignScorer is built lazily, on first use."""
    cache = {}

    def get_metric(is_bidirectional):
        if is_bidirectional not in cache:
            metric = AlignScoreNew(batch_size=1, return_mean=is_bidirectional)
            if aggregated:
                metric = AggregatedMetric(base_metric=metric)
            cache[is_bidirectional] = metric
        return cache[is_bidirectional]

    def router(source_id):
        return get_metric(_alignscore_is_bidirectional(source_id))

    return router


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_dataset", required=True,
                         choices=["sciq", "trivia_qa", "qa", "pubmed_qa", "xsum", "cnn_dailymail"],
                         help="ProbeDrift evaluation dataset.")
    parser.add_argument("--ood_setting", default="ID",
                         choices=["ID", "OOD_LEAVE_ONE_OUT", "OOD_ONE_DATASET_SAME_TASK",
                                  "OOD_DIFF_TASK", "OOD_ONE_DATASET_DIFF_TASK"],
                         help="ProbeDrift training composition. Ignored by methods that don't train (e.g. msp).")
    parser.add_argument("--instruct", action="store_true", default=False)
    parser.add_argument("--method", required=True,
                         choices=("msp", "feature_supervision") + _MD_METHODS)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--ignore_exceptions", action="store_true", default=False)
    parser.add_argument("--cache_path", default="./workdir/output")

    # --model_path for a standard HF causal LM; --model_config for instruct/4-bit/custom-load-script models.
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--loadin4bit", action="store_true", default=False)

    # Output/save destinations.
    parser.add_argument("--output_file", default=None)

    parser.add_argument("--md_save_file", default=None)
    parser.add_argument("--llm_judge_file", default=None)

    # feature_supervision (SAPLMA/UHEAD/etc) options.
    parser.add_argument("--probe_head_type", default=None,
                         choices=["full_sequence_saplma",
                                  "full_sequence_linear_regression",
                                  "full_sequence_uhead"])
    parser.add_argument("--probe_feature_extractor_setting", default=None)
    parser.add_argument("--probe_head_dim", type=int, default=None)
    parser.add_argument("--probe_n_layers", type=int, default=None)
    parser.add_argument("--probe_n_heads", type=int, default=None)
    parser.add_argument("--probe_dropout", type=float, default=None)
    parser.add_argument("--probe_weight_decay", type=float, default=None)
    parser.add_argument("--probe_learning_rate", type=float, default=None)
    parser.add_argument("--probe_warmup_ratio", type=float, default=None)
    parser.add_argument("--probe_num_train_epochs", type=int, default=None)
    parser.add_argument("--probe_train_batch_size", type=int, default=None)
    parser.add_argument("--token_aggregation", default="average")
    parser.add_argument("--clean_md_device", default="cpu")

    # MD methods (satmd/satrmd/...) options.
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layers for MD methods. Defaults to all layers of the loaded model.")
    parser.add_argument("--metric_thr", type=float, default=0.3)
    parser.add_argument("--subsample_background_train_dataset", type=int, default=2000)

    return parser.parse_args()


def build_model_config(args):
    if args.model_config:
        model_config_path = Path(args.model_config)
        default_path = model_config_path.parent / "default.yaml"
        model_cfg = OmegaConf.load(default_path) if default_path.exists() else OmegaConf.create({})
        model_cfg = OmegaConf.merge(model_cfg, OmegaConf.load(model_config_path))
        anchor = model_config_path.parent.parent / "_anchor.yaml"
    else:
        if not args.model_path:
            raise ValueError("Must pass --model_path or --model_config.")
        model_cfg = OmegaConf.create({"path": args.model_path, "device_map": "auto"})
        anchor = Path("configs/_anchor.yaml")

    if args.attn_implementation:
        model_cfg.attn_implementation = args.attn_implementation

    return model_cfg, anchor


def build_args(parsed):
    model_cfg, anchor = build_model_config(parsed)

    args = OmegaConf.create({
        k: v for k, v in vars(parsed).items()
        if not k.startswith("probe_") and k not in ("model_path", "model_config", "attn_implementation")
    })
    args.model = model_cfg
    args.generation_params = {"generate_until": _GENERATE_UNTIL}
    args.probe = OmegaConf.create({
        "head_type": parsed.probe_head_type,
        "feature_extractor_setting": parsed.probe_feature_extractor_setting,
        "head_dim": parsed.probe_head_dim,
        "n_layers": parsed.probe_n_layers,
        "n_heads": parsed.probe_n_heads,
        "dropout": parsed.probe_dropout,
        "weight_decay": parsed.probe_weight_decay,
        "learning_rate": parsed.probe_learning_rate,
        "warmup_ratio": parsed.probe_warmup_ratio,
        "num_train_epochs": parsed.probe_num_train_epochs,
        "train_batch_size": parsed.probe_train_batch_size,
    })
    args.multiref = parsed.eval_dataset in _MULTIREF_DATASETS

    return args, anchor


def main():
    parsed = get_args()
    args, anchor = build_args(parsed)

    seed = 1
    cache_kwargs = get_cache_kwargs(args)

    log.info("=" * 100)
    log.info(f"Loading model {args.model.path}...")
    transformers.set_seed(seed)

    model = load_model(args, anchor, cache_kwargs)

    if args.layers is None:
        cfg = model.model.config
        n = cfg.text_config.num_hidden_layers if hasattr(cfg, 'text_config') else cfg.num_hidden_layers
        args.layers = list(range(n - 1)) + [-1]

    log.info("Done with loading model.")

    estimator = get_estimator(args, model)

    log.info(f"Loading dataset {args.eval_dataset!r} (ood_setting={args.ood_setting!r}) via ProbeDrift...")
    train_dataset, dataset = load_datasets_via_probe_drift(args)
    if getattr(estimator, "is_fitted", True):
        train_dataset = None

    background_train_dataset = None
    needs_background_data = any(
        dep.startswith("background_train") for dep in getattr(estimator, "stats_dependencies", [])
    )

    if not getattr(estimator, "is_fitted", False) and needs_background_data:
        background_train_dataset = load_background_dataset(
            _BACKGROUND_TRAIN_DATASET,
            _BACKGROUND_TRAIN_DATASET_TEXT_COLUMN,
            _BACKGROUND_TRAIN_DATASET_LABEL_COLUMN,
            batch_size=args.batch_size,
            data_files=_BACKGROUND_TRAIN_DATASET_DATA_FILES,
            size=100_000,
            max_new_tokens=args.max_new_tokens,
            cache_kwargs=cache_kwargs,
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
        background_train_dataset_max_new_tokens=args.max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        output_file=args.output_file,
        md_save_file=args.md_save_file,
    )

    results_dict = man()
    log.info("\n\nResults from experiment: " + str(results_dict))


def get_estimator(args, model):
    if args.method in _SEQ_UE_METHODS:
        return MaximumSequenceProbability()
    if args.method == 'feature_supervision':
        return _build_feature_supervision(args, model)
    if args.method in _MD_METHODS:
        return _build_md_estimator(args)
    raise ValueError(f"No estimator configured for method={args.method!r}")


def _build_feature_supervision(args, model):
    if args.llm_judge_file:
        metric = LLMJudgeMetric(args.llm_judge_file)
        metric_name = "LLMJudge"
        metric_router = None
    else:
        metric = None
        metric_name = "AlignScore"
        metric_router = _make_alignscore_router(args.multiref)

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
    elif args.probe.head_type == 'full_sequence_saplma':
        cfg['uncertainty_head'] = None
    elif args.probe.head_type == 'full_sequence_linear_regression':
        cfg['uncertainty_head'] = None

    cfg['token_aggregation'] = args.token_aggregation

    head = AutoUncertaintyHead.from_config(OmegaConf.create(cfg), base_model=model.model)
    base_model = model if args.probe.head_type == 'full_sequence_uhead' else None

    # uhead trains on live model outputs; all other head types require pre-compiled features.
    pre_compile_features = args.probe.head_type != 'full_sequence_uhead'

    return FeatureSupervision(
        device='cuda',
        storage_device=args.clean_md_device,
        metric_name=metric_name,
        metric=metric,
        metric_router=metric_router,
        aggregated=args.multiref,
        config=cfg,
        head=head,
        pre_compile_features=pre_compile_features,
        base_model=base_model,
    )


def _get_md_metric(args):
    if args.llm_judge_file:
        return LLMJudgeMetric(args.llm_judge_file), "LLMJudge", None
    return None, "AlignScore", _make_alignscore_router(args.multiref)


def _build_md_estimator(args):
    metric, metric_name, metric_router = _get_md_metric(args)
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
        metric_router=metric_router,
        aggregated=args.multiref,
        hidden_layers=list(args.layers),
        metric_thr=args.metric_thr,
        ue=ue_type,
        positive=False,
        meta_model="LinReg",
        norm="orig",
        remove_corr=True,
        remove_alg=3,
        storage_device=args.clean_md_device,
    )
    if args.method in ('satmd', 'satrmd'):
        return LinRegTokenMahalanobisDistance(**kwargs)
    if args.method in ('satmd_msp', 'satrmd_msp'):
        return LinRegTokenMahalanobisDistance_Hybrid(**kwargs)
    if args.method in ('huq_satmd', 'huq_satrmd'):
        return HUQ_LRTMD(**kwargs)


def get_generation_metrics(args):
    return [
        RougeMetric("rougeL"),
        AccuracyMetric(normalize=True),  # normalisation: lowercase, strip punctuation before exact match
        AlignScoreNew(batch_size=1, return_mean=_alignscore_is_bidirectional(args.eval_dataset)),
    ]


if __name__ == "__main__":
    main()
