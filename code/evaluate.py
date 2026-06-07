from __future__ import annotations

import math
import time

import torch


@torch.no_grad()
def evaluate_perplexity(model, batches) -> dict[str, float]:
    losses = []
    for batch in batches:
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
        output = model(**batch, labels=labels)
        losses.append(float(output.loss.detach().cpu()))
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}


@torch.no_grad()
def measure_latency_ms(model, batches, warmup: int = 2, repeat: int = 5) -> float:
    if not batches:
        raise ValueError("At least one batch is required to measure latency.")
    sample = batches[0]
    for _ in range(warmup):
        model(**sample)

    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        model(**sample)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat
