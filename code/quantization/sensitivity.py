from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers
from .uniform import quantize_tensor


def weight_mse_by_layer(model, bits: int) -> dict[str, float]:
    scores = {}
    for name, module in iter_quantizable_layers(model):
        weight = get_layer_weight(module)
        q_weight, _ = quantize_tensor(weight, bits)
        scores[name] = float(torch.mean((weight - q_weight) ** 2).cpu())
    return scores


def hessian_trace_proxy(activation_cache: dict[str, torch.Tensor]) -> dict[str, float]:
    scores = {}
    for name, x in activation_cache.items():
        scores[name] = float((x.float() ** 2).mean().cpu())
    return scores


def layer_quantization_costs(model, candidate_bits: tuple[int, ...], sensitivity: dict[str, float]):
    costs = {}
    for name, module in iter_quantizable_layers(model):
        weight = get_layer_weight(module)
        layer_costs = {}
        for bits in candidate_bits:
            q_weight, _ = quantize_tensor(weight, bits)
            mse = float(torch.mean((weight - q_weight) ** 2).cpu())
            layer_costs[bits] = sensitivity.get(name, 1.0) * mse
        costs[name] = layer_costs
    return costs
