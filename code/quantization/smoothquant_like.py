from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers, set_layer_weight
from .uniform import quantize_tensor


def broadcast_input_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.shape[1] == scale.numel():
        return scale.unsqueeze(0)
    if weight.shape[0] == scale.numel():
        return scale.unsqueeze(1)
    raise ValueError(f"Cannot broadcast activation scale {tuple(scale.shape)} to weight {tuple(weight.shape)}")


def apply_smoothquant_like(model, activation_cache: dict[str, torch.Tensor], bits: int = 8, alpha: float = 0.5):
    """Weight-side SmoothQuant approximation for report experiments."""
    for name, module in iter_quantizable_layers(model):
        if name not in activation_cache:
            continue
        weight = get_layer_weight(module)
        act_max = activation_cache[name].abs().amax(dim=0).to(weight.device).clamp_min(1e-8)
        if weight.shape[1] == act_max.numel():
            weight_max = weight.float().abs().amax(dim=0).clamp_min(1e-8)
        elif weight.shape[0] == act_max.numel():
            weight_max = weight.float().abs().amax(dim=1).clamp_min(1e-8)
        else:
            raise ValueError(f"Cannot align activation scale {tuple(act_max.shape)} to weight {tuple(weight.shape)}")
        scale = act_max.pow(alpha) / weight_max.pow(1 - alpha)
        smoothed_weight = weight.float() * broadcast_input_scale(weight, scale)
        q_weight, _ = quantize_tensor(smoothed_weight, bits)
        set_layer_weight(module, q_weight)
    return model
