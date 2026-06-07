from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name_or_path: str, device: str = "cpu"):
    local_files_only = Path(model_name_or_path).exists()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def is_quantizable_layer(module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.nn.Parameter) and weight.ndim == 2


def get_layer_weight(module) -> torch.Tensor:
    return module.weight.data


def set_layer_weight(module, weight: torch.Tensor) -> None:
    module.weight.data.copy_(weight.to(module.weight.dtype))


def iter_quantizable_layers(model):
    for name, module in model.named_modules():
        if is_quantizable_layer(module):
            yield name, module


def iter_linear_layers(model):
    yield from iter_quantizable_layers(model)


def estimate_parameter_memory_mb(model, bit_overrides: dict[str, float] | None = None) -> float:
    bit_overrides = bit_overrides or {}
    total_bits = 0
    for name, param in model.named_parameters():
        layer_name = name.rsplit(".", 1)[0]
        bits = bit_overrides.get(name, bit_overrides.get(layer_name, param.element_size() * 8))
        total_bits += param.numel() * bits
    return total_bits / 8 / 1024 / 1024


def linear_weight_bit_overrides(model, bits: float) -> dict[str, float]:
    return {f"{name}.weight": bits for name, _module in iter_quantizable_layers(model)}


def mixed_precision_bit_overrides(assignment: dict[str, int]) -> dict[str, float]:
    return {f"{name}.weight": bits for name, bits in assignment.items()}
