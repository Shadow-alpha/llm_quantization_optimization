from __future__ import annotations

import torch

from .uniform import quantize_tensor


def apply_gptq_like(model, activation_cache: dict[str, torch.Tensor], bits: int = 4, damping: float = 1e-4):
    """Project-course GPTQ variant using diagonal Hessian compensation."""
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or name not in activation_cache:
            continue
        x = activation_cache[name].to(module.weight.device, dtype=torch.float32)
        diag_h = (x * x).mean(dim=0).clamp_min(damping)
        q_weight, _ = quantize_tensor(module.weight.data.float(), bits)
        error = module.weight.data.float() - q_weight.float()
        compensated = q_weight.float() + error / diag_h.unsqueeze(0) * damping
        module.weight.data.copy_(compensated.to(module.weight.dtype))
    return model

