from __future__ import annotations

import torch

from models import get_layer_weight, iter_quantizable_layers, set_layer_weight


def quantize_tensor(weight: torch.Tensor, bits: int, per_channel: bool = True):
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    w = weight.detach()

    if per_channel and w.ndim >= 2:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    else:
        scale = w.abs().max().clamp_min(1e-8) / qmax

    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    dequant = q * scale
    return dequant.to(weight.dtype), {"scale": scale, "bits": bits}


def quantize_linear_layer(layer, bits: int, per_channel: bool = True) -> None:
    dequant, _ = quantize_tensor(get_layer_weight(layer), bits, per_channel)
    set_layer_weight(layer, dequant)


def apply_uniform_quantization(model, bits: int, per_channel: bool = True):
    for _name, module in iter_quantizable_layers(model):
        quantize_linear_layer(module, bits, per_channel)
    return model
