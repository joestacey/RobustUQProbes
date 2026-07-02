import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph_lite.utils.model import WhiteboxModel


def get_embeddings_from_output(
    output,
    batch,
    model_type,
    hidden_state: List[str] = ["encoder", "decoder"],
    ignore_padding: bool = True,
    use_averaging: bool = True,
    all_layers: bool = False,
    aggregation_method: str = "mean",
    level: str = "sequence",
    hidden_layer: int = -1,
    return_source_hidden_states: bool = False,
):
    batch_embeddings = None
    batch_embeddings_decoder = None
    batch_size = len(batch["input_ids"])

    if model_type == "CausalLM":
        input_tokens_hs = output.hidden_states[0][hidden_layer].cpu().detach()
        if return_source_hidden_states:
            batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()
        else:
            if not all_layers:
                if len(output.hidden_states) > 1:
                    generated_tokens_hs = torch.cat(
                        [
                            h[hidden_layer].cpu().detach()
                            for h in output.hidden_states[1:]
                        ],
                        dim=1,
                    )
            else:
                input_tokens_hs = output.hidden_states[0].mean(axis=0).cpu().detach()
                if len(output.hidden_states) > 1:
                    generated_tokens_hs = torch.cat(
                        [
                            h.mean(axis=0).cpu().detach()
                            for h in output.hidden_states[1:]
                        ],
                        dim=1,
                    )
            if len(output.hidden_states) > 1:
                if level == "sequence":
                    batch_embeddings_decoder = (
                        torch.cat([input_tokens_hs, generated_tokens_hs], dim=1)
                        .mean(axis=1)
                        .cpu()
                        .detach()
                    )
                elif level == "token":
                    batch_embeddings_decoder = (
                        torch.cat([input_tokens_hs[:, -1:], generated_tokens_hs], dim=1)
                        .cpu()
                        .detach()
                    )
            else:
                batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()
        batch_embeddings = None
    elif model_type == "Seq2SeqLM":
        if use_averaging:
            if "decoder" in hidden_state:
                try:
                    decoder_hidden_states = torch.stack(
                        [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                    )
                    if all_layers:
                        agg_decoder_hidden_states = decoder_hidden_states[
                            :, :, :, 0
                        ].mean(axis=1)
                    else:
                        agg_decoder_hidden_states = decoder_hidden_states[:, -1, :, 0]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=0
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(batch_size, -1, agg_decoder_hidden_states.shape[-1])[
                            :, 0
                        ]
                    )
                except TypeError:
                    if all_layers:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        ).mean(axis=0)
                    else:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        )[-1]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=1
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(-1, agg_decoder_hidden_states.shape[-1])
                    )

            if "encoder" in hidden_state:
                mask = batch["attention_mask"][:, :, None].cpu().detach()
                seq_lens = batch["attention_mask"].sum(-1)[:, None].cpu().detach()
                if all_layers:
                    encoder_embeddings = (
                        aggregate(
                            torch.stack(output.encoder_hidden_states), "mean", axis=0
                        )
                        .cpu()
                        .detach()
                        * mask
                    )
                else:
                    encoder_embeddings = (
                        output.encoder_hidden_states[-1].cpu().detach() * mask
                    )

                if ignore_padding:
                    if aggregation_method == "mean":
                        batch_embeddings = (encoder_embeddings).sum(
                            1
                        ).cpu().detach() / seq_lens
                    else:
                        batch_embeddings = (
                            aggregate(encoder_embeddings, aggregation_method, axis=1)
                            .cpu()
                            .detach()
                        )
                else:
                    batch_embeddings = (
                        aggregate(encoder_embeddings, aggregation_method, axis=1)
                        .cpu()
                        .detach()
                    )
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
        else:
            if "decoder" in hidden_state:
                decoder_hidden_states = torch.stack(
                    [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                )
                last_decoder_hidden_states = decoder_hidden_states[-1, -1, :, 0]
                batch_embeddings_decoder = (
                    last_decoder_hidden_states.reshape(
                        batch_size, -1, last_decoder_hidden_states.shape[-1]
                    )[:, 0]
                    .cpu()
                    .detach()
                )
            if "encoder" in hidden_state:
                batch_embeddings = output.encoder_hidden_states[-1][:, 0].cpu().detach()
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
    else:
        raise NotImplementedError

    return batch_embeddings, batch_embeddings_decoder


def aggregate(x, aggregation_method, axis):
    if aggregation_method == "max":
        return x.max(axis=axis).values
    elif aggregation_method == "mean":
        return x.mean(axis=axis)
    elif aggregation_method == "sum":
        return x.sum(axis=axis)


class OutputWrapper:
    hidden_states = None
    encoder_hidden_states = None
    decoder_hidden_states = None


class EmbeddingsCalculator(StatCalculator):
    def __init__(self, hidden_layers: List[int] = [-1], stage: str = "train"):
        self.hidden_layers = hidden_layers
        self.stage = stage
        if stage == "train":
            self.stage += "_"

        stats = []
        deps = [f"{self.stage}embeddings_all"]
        for layer in self.hidden_layers:
            if layer == -1:
                layer_name = ""
            else:
                layer_name = f"_{layer}"
            if stage == "train":
                stats += [
                    f"{self.stage}embeddings{layer_name}",
                    f"background_{self.stage}embeddings{layer_name}",
                    f"{self.stage}token_embeddings{layer_name}",
                    f"background_{self.stage}token_embeddings{layer_name}",
                ]
            else:
                stats += [
                    f"{self.stage}embeddings{layer_name}",
                    f"{self.stage}token_embeddings{layer_name}",
                ]
        if stage == "train":
            deps += [f"background_{self.stage}embeddings_all"]
        super().__init__(
            stats,
            deps,
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = OutputWrapper()
            if model.model_type == "CausalLM":
                out.hidden_states = dependencies["embeddings_all_decoder"]
            elif model.model_type == "Seq2SeqLM":
                out.decoder_hidden_states = dependencies["embeddings_all_decoder"]
                out.encoder_hidden_states = dependencies["embeddings_all_encoder"]

            results = {}
            for layer in self.hidden_layers:
                if layer == -1:
                    layer_name = ""
                else:
                    layer_name = f"_{layer}"
                embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                    out,
                    batch,
                    model.model_type,
                    level="sequence",
                    hidden_layer=layer,
                )
                token_embeddings_encoder, token_embeddings_decoder = (
                    get_embeddings_from_output(
                        out,
                        batch,
                        model.model_type,
                        level="token",
                        hidden_layer=layer,
                    )
                )
                if token_embeddings_decoder is None:
                    token_embeddings_decoder = torch.empty(
                        (0, embeddings_decoder.shape[-1]), dtype=torch.float32
                    )
                elif len(token_embeddings_decoder.shape) == 3:
                    token_embeddings_decoder = token_embeddings_decoder.reshape(
                        -1, token_embeddings_decoder.shape[-1]
                    )

                if model.model_type == "CausalLM":
                    results[f"embeddings_decoder{layer_name}"] = (
                        embeddings_decoder.cpu().detach().numpy()
                    )
                    results[f"token_embeddings_decoder{layer_name}"] = (
                        token_embeddings_decoder.cpu().detach().numpy()
                    )
                elif model.model_type == "Seq2SeqLM":
                    pass
                else:
                    raise NotImplementedError

        return results
