import torch
import sys
import logging

from dataclasses import asdict
from typing import List, Dict, Optional, Union
from abc import abstractmethod, ABC
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    BartForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedTokenizer,
)

from lm_polygraph_lite.utils.generation_parameters import GenerationParameters

log = logging.getLogger("lm_polygraph_lite")


class Model(ABC):
    """
    Abstract model class. Used as base class for white-box models.
    """

    def __init__(self, model_path: str, model_type: str):
        """
        Parameters:
            model_path (str): unique model path where it can be found.
            model_type (str): description of additional model properties (e.g. 'CausalLM', 'Seq2SeqLM').
        """
        self.model_path = model_path
        self.model_type = model_type

    @abstractmethod
    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Abstract method. Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def generate(self, **args):
        """
        Abstract method. Generates the model output with scores from batch formed by HF Tokenizer.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def __call__(self, **args):
        """
        Abstract method. Calls the model on the input batch. Returns the resulted scores.
        Not implemented for black-box models.
        """
        raise Exception("Not implemented")


def _validate_args(args):
    if "presence_penalty" in args.keys() and args["presence_penalty"] != 0.0:
        sys.stderr.write(
            "Skipping requested argument presence_penalty={}".format(
                args["presence_penalty"]
            )
        )

    # remove arguments that are not supported by the HF model.generate function
    keys_to_remove = [
        "presence_penalty",
        "generate_until",
        "allow_newlines",
    ]
    for key in keys_to_remove:
        args.pop(key, None)

    return args


class WhiteboxModel(Model):
    """
    White-box model class. Have access to model scores and logits. Currently implemented only for Huggingface models.

    Examples:

    ```python
    >>> from lm_polygraph_lite import WhiteboxModel
    >>> model = WhiteboxModel.from_pretrained(
    ...     "bigscience/bloomz-3b",
    ... )
    ```
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_path: str = None,
        model_type: str = "CausalLM",
        generation_parameters: GenerationParameters = GenerationParameters(),
    ):
        """
        Parameters:
            model (AutoModelForCausalLM): HuggingFace model.
            tokenizer (AutoTokenizer): HuggingFace tokenizer.
            model_path (Optional[str]): Unique model path in HuggingFace.
            model_type (str): Additional model specifications.
            parameters (GenerationParameters): parameters to use in model generation. Default: default parameters.
        """
        super().__init__(model_path, model_type)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_parameters = generation_parameters

    class _ScoresProcessor:
        # Stores original token scores instead of the ones modified with generation parameters
        def __init__(self):
            self.scores = []

        def __call__(self, input_ids=None, scores=None):
            self.scores.append(scores.log_softmax(-1))
            return scores

    class _MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence.
        Copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/utils.py#L208
        """

        def __init__(
            self,
            sequence: str,
            tokenizer: PreTrainedTokenizer,
            initial_decoder_input_length: int,
            batch_size: int,
            require_prior_content: bool = False,
        ) -> None:
            self.initial_decoder_input_length = initial_decoder_input_length
            self.done_tracker = [False] * batch_size
            self.sequence = sequence
            self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
            # Look back 2 extra tokens beyond the stop sequence's own token count:
            # a model might generate ['\n', '\n'] where the stop sequence is
            # tokenized as ['\n\n'], so decoding a slightly wider window catches
            # matches that a different tokenization would otherwise miss.
            self.sequence_id_len = len(self.sequence_ids) + 2
            self.tokenizer = tokenizer
            # If set, a match only ends generation once non-whitespace content
            # exists before it, so a leading stop-sequence match (e.g. '\n'
            # before any real answer) doesn't end generation with zero content.
            self.require_prior_content = require_prior_content

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
            generated_ids_batch = input_ids[:, self.initial_decoder_input_length :]

            lookback_ids_batch = generated_ids_batch[:, -self.sequence_id_len :]

            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

            if self.require_prior_content:
                full_tokens_batch = self.tokenizer.batch_decode(generated_ids_batch)

            for i, done in enumerate(self.done_tracker):
                if not done:
                    matched = self.sequence in lookback_tokens_batch[i]
                    if matched and self.require_prior_content:
                        # Check content before the matched occurrence, not the
                        # full text: a stale earlier match (e.g. leading '\n')
                        # could otherwise satisfy the check on its own.
                        full_text = full_tokens_batch[i]
                        last_idx = full_text.rfind(self.sequence)
                        content_before = (
                            full_text[:last_idx] if last_idx >= 0 else full_text
                        )
                        matched = bool(
                            content_before.replace(self.sequence, "").strip()
                        )
                    self.done_tracker[i] = matched
            return False not in self.done_tracker

    def get_stopping_criteria(self, input_ids: torch.Tensor):
        eos = self.tokenizer.decode(self.tokenizer.eos_token_id)
        generate_until = list(self.generation_parameters.generate_until)
        stop_sequences = generate_until + [eos]
        # Only the user-requested generate_until sequences require prior
        # content before they're allowed to stop generation; the appended
        # eos sequence stops immediately as before.
        require_prior_content_flags = [True] * len(generate_until) + [False]
        return StoppingCriteriaList(
            [
                self._MultiTokenEOSCriteria(
                    sequence,
                    self.tokenizer,
                    input_ids.shape[1],
                    input_ids.shape[0],
                    require_prior_content=flag,
                )
                for sequence, flag in zip(stop_sequences, require_prior_content_flags)
            ]
        )

    def generate(self, **args):
        """
        Generates the model output with scores from batch formed by HF Tokenizer.

        Parameters:
            **args: Any arguments that can be passed to model.generate function from HuggingFace.
        Returns:
            ModelOutput: HuggingFace generation output with scores overriden with original probabilities.
        """
        default_params = asdict(self.generation_parameters)

        if len(self.generation_parameters.generate_until) > 0:
            args["stopping_criteria"] = self.get_stopping_criteria(args["input_ids"])

        # add ScoresProcessor to collect original scores
        processor = self._ScoresProcessor()
        if "logits_processor" in args.keys():
            logits_processor = LogitsProcessorList(
                [processor, args["logits_processor"]]
            )
        else:
            logits_processor = LogitsProcessorList([processor])
        args["logits_processor"] = logits_processor

        # update default parameters with passed arguments
        default_params.update(args)
        args = default_params
        args = _validate_args(args)

        generation = self.model.generate(**args)

        # override generation.scores with original scores from model
        generation.generation_scores = generation.scores
        generation.scores = processor.scores

        return generation

    def generate_texts(self, input_texts: List[str], **args) -> List[str]:
        """
        Generates a list of model answers using input texts batch.

        Parameters:
            input_texts (List[str]): input texts batch.
        Return:
            List[str]: corresponding model generations. Have the same length as `input_texts`.
        """
        args = _validate_args(args)
        args["return_dict_in_generate"] = True
        batch: Dict[str, torch.Tensor] = self.tokenize(input_texts)
        batch = {k: v.to(self.device()) for k, v in batch.items()}
        sequences = self.generate(**batch, **args).sequences.cpu()
        input_len = batch["input_ids"].shape[1]
        texts = []

        decode_args = {}
        if self.tokenizer.chat_template is not None:
            decode_args["skip_special_tokens"] = True

        for seq in sequences:
            if self.model_type == "CausalLM":
                texts.append(self.tokenizer.decode(seq[input_len:], **decode_args))
            else:
                texts.append(self.tokenizer.decode(seq[1:], **decode_args))

        return texts

    def __call__(self, **args):
        """
        Calls the model on the input batch. Returns the resulted scores.
        """
        return self.model(**args)

    def device(self):
        """
        Returns the device the model is currently loaded on.

        Returns:
            str: device string.
        """
        return self.model.device

    @staticmethod
    def from_pretrained(
        model_path: str,
        generation_params: Optional[Dict] = {},
        add_bos_token: bool = True,
        **kwargs,
    ):
        """
        Initializes the model from HuggingFace. Automatically determines model type.

        Parameters:
            model_path (str): model path in HuggingFace.
            generation_params (Dict): generation arguments for
                lm_polygraph_lite.utils.generation_parametersGenerationParameters
            add_bos_token (bool): tokenizer argument. Default: True.
        """
        log.warning(
            "WhiteboxModel#from_pretrained is deprecated and will be removed in the next release. Please instantiate WhiteboxModel directly by passing an already loaded model, tokenizer and model path."
        )

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        generation_params = GenerationParameters(**generation_params)

        if any(["CausalLM" in architecture for architecture in config.architectures]):
            model_type = "CausalLM"

            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, **kwargs
            )
        elif any(
            [
                ("Seq2SeqLM" in architecture)
                or ("ConditionalGeneration" in architecture)
                for architecture in config.architectures
            ]
        ):
            model_type = "Seq2SeqLM"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
            if "falcon" in model_path:
                model.transformer.alibi = True
        elif any(
            ["JAISLMHeadModel" in architecture for architecture in config.architectures]
        ):
            model_type = "CausalLM"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs,
            )
        elif any(
            ["BartModel" in architecture for architecture in config.architectures]
        ):
            model_type = "Seq2SeqLM"
            model = BartForConditionalGeneration.from_pretrained(model_path, **kwargs)
        else:
            raise ValueError(
                f"Model {model_path} is not adapted for the sequence generation task"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            add_bos_token=add_bos_token,
            **kwargs,
        )

        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        instance = WhiteboxModel(
            model, tokenizer, model_path, model_type, generation_params
        )

        return instance

    def tokenize(
        self, texts: Union[List[str], List[List[Dict[str, str]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes input texts batch into a dictionary using the model tokenizer.

        Parameters:
            texts (List[str]): list of input texts batch.
        Returns:
            dict[str, torch.Tensor]: tensors dictionary obtained by tokenizing input texts batch.
        """
        # Apply chat template if tokenizer has it
        if (self.tokenizer.chat_template is not None):
            if ("Qwen2.5" in self.tokenizer.name_or_path) and ("Instruct" not in self.tokenizer.name_or_path):
                pass
            else:
                formatted_texts = []
                for chat in texts:
                    if isinstance(chat, str):
                        chat = [{"role": "user", "content": chat}]
                    formatted_chat = self.tokenizer.apply_chat_template(
                        chat, add_generation_prompt=True, tokenize=False
                    )
                    formatted_texts.append(formatted_chat)
                texts = formatted_texts
        batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        if batch["input_ids"].shape[-1] > 2048:
            return self.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            )

        return batch


