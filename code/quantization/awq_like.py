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


def apply_awq_like(model, activation_cache: dict[str, torch.Tensor], bits: int = 4, alpha: float = 0.5):
    """Activation-aware scaling before weight quantization."""
    for name, module in iter_quantizable_layers(model):
        if name not in activation_cache:
            continue
        weight = get_layer_weight(module)
        act = activation_cache[name].abs().mean(dim=0).to(weight.device).clamp_min(1e-8)
        scale = act.pow(alpha)
        scale_view = broadcast_input_scale(weight, scale)
        scaled_weight = weight.float() * scale_view
        q_weight, _ = quantize_tensor(scaled_weight, bits)
        restored = q_weight / scale_view
        set_layer_weight(module, restored)
    return model
