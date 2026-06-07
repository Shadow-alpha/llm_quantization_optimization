from __future__ import annotations

import torch

from .uniform import quantize_tensor


def apply_awq_like(model, activation_cache: dict[str, torch.Tensor], bits: int = 4, alpha: float = 0.5):
    """Activation-aware scaling before weight quantization."""
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name not in activation_cache:
            continue
        act = activation_cache[name].abs().mean(dim=0).to(module.weight.device).clamp_min(1e-8)
        scale = act.pow(alpha)
        scaled_weight = module.weight.data.float() * scale.unsqueeze(0)
        q_weight, _ = quantize_tensor(scaled_weight, bits)
        restored = q_weight / scale.unsqueeze(0)
        module.weight.data.copy_(restored.to(module.weight.dtype))
    return model

