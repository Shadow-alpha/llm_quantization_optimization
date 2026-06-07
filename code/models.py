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


def iter_linear_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def estimate_parameter_memory_mb(model, bit_overrides: dict[str, int] | None = None) -> float:
    bit_overrides = bit_overrides or {}
    total_bits = 0
    for name, param in model.named_parameters():
        layer_name = name.rsplit(".", 1)[0]
        bits = bit_overrides.get(layer_name, param.element_size() * 8)
        total_bits += param.numel() * bits
    return total_bits / 8 / 1024 / 1024
