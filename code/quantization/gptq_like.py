from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers, set_layer_weight
from .uniform import quantize_tensor


def apply_gptq_like(model, activation_cache: dict[str, torch.Tensor], bits: int = 4, damping: float = 1e-4):
    """Project-course GPTQ variant using diagonal Hessian compensation."""
    for name, module in iter_quantizable_layers(model):
        if name not in activation_cache:
            continue
        weight = get_layer_weight(module)
        x = activation_cache[name].to(weight.device, dtype=torch.float32)
        diag_h = (x * x).mean(dim=0).clamp_min(damping)
        if weight.shape[1] == diag_h.numel():
            diag_view = diag_h.unsqueeze(0)
        elif weight.shape[0] == diag_h.numel():
            diag_view = diag_h.unsqueeze(1)
        else:
            raise ValueError(f"Cannot align Hessian diagonal {tuple(diag_h.shape)} to weight {tuple(weight.shape)}")
        q_weight, _ = quantize_tensor(weight.float(), bits)
        error = weight.float() - q_weight.float()
        compensated = q_weight.float() + error / diag_view * damping
        set_layer_weight(module, compensated)
    return model
