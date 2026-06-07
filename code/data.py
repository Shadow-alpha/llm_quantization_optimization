from __future__ import annotations

import torch
from datasets import load_dataset, load_from_disk


def load_text_splits(dataset_name: str, dataset_config: str, text_field: str, dataset_path: str | None = None):
    dataset = load_from_disk(dataset_path) if dataset_path else load_dataset(dataset_name, dataset_config)
    train = [x[text_field] for x in dataset["train"] if x[text_field].strip()]
    test = [x[text_field] for x in dataset["test"] if x[text_field].strip()]
    return train, test


def tokenize_texts(tokenizer, texts, max_length: int, limit: int, device: str):
    encoded = []
    for text in texts[:limit]:
        item = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded.append({k: v.to(device) for k, v in item.items()})
    return encoded


def collect_linear_inputs(model, batches, max_batches: int = 16):
    caches: dict[str, list[torch.Tensor]] = {}
    hooks = []

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float().cpu()
            caches.setdefault(name, []).append(x.reshape(-1, x.shape[-1]))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in batches[:max_batches]:
            model(**batch)

    for hook in hooks:
        hook.remove()

    return {name: torch.cat(values, dim=0) for name, values in caches.items()}
