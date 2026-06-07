from __future__ import annotations

from pathlib import Path

import torch


def load_text_splits(dataset_name: str, dataset_config: str, text_field: str, dataset_path: str | None = None):
    if dataset_path:
        return load_text_splits_from_parquet(Path(dataset_path), text_field)

    raise RuntimeError(
        "dataset_path is required. Download parquet files first with code/download_dataset.py, "
        "then pass --dataset-path to the experiment command."
    )


def load_text_splits_from_parquet(dataset_dir: Path, text_field: str):
    train = read_parquet_split(dataset_dir, "train", text_field)
    test = read_parquet_split(dataset_dir, "test", text_field)
    if not test:
        test = read_parquet_split(dataset_dir, "validation", text_field)
    if not train or not test:
        raise RuntimeError(
            f"Could not find non-empty train/test text splits under {dataset_dir}. "
            "Expected parquet filenames or parent directories to contain train/test/validation."
        )
    return train, test


def read_parquet_split(dataset_dir: Path, split: str, text_field: str):
    import pandas as pd

    files = find_split_parquet_files(dataset_dir, split)
    texts = []
    for file in files:
        frame = pd.read_parquet(file, columns=[text_field])
        values = frame[text_field].dropna().astype(str)
        texts.extend(value for value in values if value.strip())
    return texts


def find_split_parquet_files(dataset_dir: Path, split: str):
    split = split.lower()
    files = sorted(dataset_dir.rglob("*.parquet"))
    return [
        file
        for file in files
        if split in file.stem.lower() or any(part.lower() == split for part in file.parts)
    ]


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
