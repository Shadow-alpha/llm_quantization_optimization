from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers, set_layer_weight
from .uniform import quantize_tensor


def apply_llm_int8_like(model, activation_cache: dict[str, torch.Tensor], threshold: float = 6.0):
    """Quantize normal columns to INT8 and keep activation-outlier columns in full precision."""
    outlier_stats = {}
    for name, module in iter_quantizable_layers(model):
        if name not in activation_cache:
            continue
        weight = get_layer_weight(module)
        act = activation_cache[name].abs()
        outlier_mask = act.amax(dim=0).to(weight.device) > threshold
        q_weight, _ = quantize_tensor(weight.float(), 8)
        mixed = q_weight.to(weight.dtype)
        if weight.shape[1] == outlier_mask.numel():
            mixed[:, outlier_mask] = weight[:, outlier_mask]
        elif weight.shape[0] == outlier_mask.numel():
            mixed[outlier_mask, :] = weight[outlier_mask, :]
        else:
            raise ValueError(f"Cannot align outlier mask {tuple(outlier_mask.shape)} to weight {tuple(weight.shape)}")
        set_layer_weight(module, mixed)
        outlier_stats[name] = float(outlier_mask.float().mean().cpu())
    return model, outlier_stats
