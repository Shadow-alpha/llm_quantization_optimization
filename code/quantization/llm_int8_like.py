from __future__ import annotations

import torch

from .uniform import quantize_tensor


def apply_llm_int8_like(model, activation_cache: dict[str, torch.Tensor], threshold: float = 6.0):
    """Quantize normal columns to INT8 and keep activation-outlier columns in full precision."""
    outlier_stats = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name not in activation_cache:
            continue
        act = activation_cache[name].abs()
        outlier_mask = act.amax(dim=0).to(module.weight.device) > threshold
        q_weight, _ = quantize_tensor(module.weight.data.float(), 8)
        mixed = q_weight.to(module.weight.dtype)
        mixed[:, outlier_mask] = module.weight.data[:, outlier_mask]
        module.weight.data.copy_(mixed)
        outlier_stats[name] = float(outlier_mask.float().mean().cpu())
    return model, outlier_stats

