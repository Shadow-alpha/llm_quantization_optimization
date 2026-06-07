from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers
from optimization.knapsack import solve_layer_bit_knapsack
from .sensitivity import layer_quantization_costs
from .uniform import quantize_linear_layer


def apply_mixed_precision(model, candidate_bits: tuple[int, ...], sensitivity: dict[str, float], budget_mb: float):
    costs = layer_quantization_costs(model, candidate_bits, sensitivity)
    sizes = {}
    for name, module in iter_quantizable_layers(model):
        sizes[name] = get_layer_weight(module).numel()

    assignment = solve_layer_bit_knapsack(costs, sizes, candidate_bits, budget_mb)
    for name, module in iter_quantizable_layers(model):
        if name in assignment:
            quantize_linear_layer(module, assignment[name])
    return model, assignment
