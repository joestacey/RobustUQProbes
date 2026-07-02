from collections.abc import Iterable
import math

def get_layer_nums(layer_nums, orig_base_model):
    if layer_nums == 'all':
        # Correct for attention (n layers, 0-indexed). For hidden_states, HF
        # adds an embedding at index 0, so 'all' would include the embedding
        # and miss the final transformer layer, so only pass 'all' for attention.
        return list(range(orig_base_model.config.num_hidden_layers))
    elif layer_nums == 'middle':
        middle_layer = math.ceil(orig_base_model.config.num_hidden_layers/2) - 1
        return [middle_layer]

    elif isinstance(layer_nums, Iterable):
        return list(layer_nums)
    return (layer_nums,)


def get_head_nums(head_nums, layer_nums, orig_base_model):
    if head_nums == 'all':
        all_heads = list(range(orig_base_model.config.num_attention_heads))
        return {l: all_heads for l in layer_nums}
    elif isinstance(head_nums, dict):
        heads: dict[int, list[int]] = {}  # list of heads for each layer
        for key, val in head_nums.items():
            for l in get_layer_nums(key, orig_base_model):
                heads.update(get_head_nums(val, [l], orig_base_model))
        assert all(l in heads.keys() for l in layer_nums)
        return heads
    elif isinstance(head_nums, Iterable):
        return {l: list(head_nums) for l in layer_nums}
    return {l: (head_nums,) for l in layer_nums}
