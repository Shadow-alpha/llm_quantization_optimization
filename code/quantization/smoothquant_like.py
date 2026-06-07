from __future__ import annotations

import torch

from .uniform import quantize_tensor


def apply_smoothquant_like(model, activation_cache: dict[str, torch.Tensor], bits: int = 8, alpha: float = 0.5):
    """Weight-side SmoothQuant approximation for report experiments."""
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name not in activation_cache:
            continue
        act_max = activation_cache[name].abs().amax(dim=0).to(module.weight.device).clamp_min(1e-8)
        weight_max = module.weight.data.float().abs().amax(dim=0).clamp_min(1e-8)
        scale = act_max.pow(alpha) / weight_max.pow(1 - alpha)
        smoothed_weight = module.weight.data.float() * scale.unsqueeze(0)
        q_weight, _ = quantize_tensor(smoothed_weight, bits)
        module.weight.data.copy_(q_weight.to(module.weight.dtype))
    return model

