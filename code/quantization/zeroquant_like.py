from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers, set_layer_weight
from .uniform import quantize_tensor


def apply_zeroquant_like(model, bits: int = 8, group_size: int = 64):
    """Group-wise weight quantization as a lightweight ZeroQuant-style PTQ step."""
    for _name, module in iter_quantizable_layers(model):
        weight = get_layer_weight(module)
        result = torch.empty_like(weight)
        for start in range(0, weight.shape[1], group_size):
            end = min(start + group_size, weight.shape[1])
            result[:, start:end], _ = quantize_tensor(weight[:, start:end], bits, per_channel=True)
        set_layer_weight(module, result)
    return model
