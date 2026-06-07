from __future__ import annotations

import torch

from .uniform import quantize_tensor


def apply_zeroquant_like(model, bits: int = 8, group_size: int = 64):
    """Group-wise weight quantization as a lightweight ZeroQuant-style PTQ step."""
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight.data
        result = torch.empty_like(weight)
        for start in range(0, weight.shape[1], group_size):
            end = min(start + group_size, weight.shape[1])
            result[:, start:end], _ = quantize_tensor(weight[:, start:end], bits, per_channel=True)
        module.weight.data.copy_(result)
    return model

