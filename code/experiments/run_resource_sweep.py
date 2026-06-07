from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from config import ExperimentConfig, FIGURES_DIR, TABLES_DIR, ensure_result_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sensitivity-aware mixed-precision quantization under several memory budgets."
    )
    parser.add_argument(
        "--budget-ratios",
        default="0.25,0.35,0.50,0.75,1.00",
        help="Comma-separated ratios of the 16-bit Linear-layer memory budget.",
    )
    parser.add_argument("--model-path", default=None, help="Local model directory. Overrides config.model_name.")
    parser.add_argument("--dataset-path", default=None, help="Local dataset directory saved by code/download_dataset.py.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_budget_ratios(raw: str) -> list[float]:
    ratios = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not ratios:
        raise ValueError("At least one budget ratio is required.")
    if any(r <= 0 for r in ratios):
        raise ValueError("Budget ratios must be positive.")
    return ratios


def linear_weight_memory_mb(model, bits: int = 16) -> float:
    import torch

    total_bits = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_bits += module.weight.numel() * bits
    return total_bits / 8 / 1024 / 1024


def assignment_memory_mb(model, assignment: dict[str, int]) -> float:
    import torch

    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_bits += module.weight.numel() * assignment.get(name, 16)
    return total_bits / 8 / 1024 / 1024


def summarize_assignment(assignment: dict[str, int]) -> str:
    counts: dict[int, int] = {}
    for bits in assignment.values():
        counts[bits] = counts.get(bits, 0) + 1
    return ", ".join(f"{bits}bit:{counts[bits]}" for bits in sorted(counts))


def main():
    args = parse_args()
    config = ExperimentConfig()
    if args.model_path:
        config.model_path = args.model_path
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    budget_ratios = parse_budget_ratios(args.budget_ratios)
    ensure_result_dirs()

    if args.dry_run:
        print("Resource sweep dry run passed.")
        print(f"Budget ratios: {budget_ratios}")
        print(config)
        return

    import pandas as pd

    from data import collect_linear_inputs, load_text_splits, tokenize_texts
    from evaluate import evaluate_perplexity, measure_latency_ms
    from models import load_causal_lm
    from optimization.pareto import pareto_frontier
    from quantization.mixed_precision import apply_mixed_precision
    from quantization.sensitivity import hessian_trace_proxy
    from utils.plotting import plot_pareto

    model_source = config.model_path or config.model_name
    model, tokenizer = load_causal_lm(model_source, config.device)
    train_texts, test_texts = load_text_splits(
        config.dataset_name,
        config.dataset_config,
        config.text_field,
        config.dataset_path,
    )
    calibration = tokenize_texts(tokenizer, train_texts, config.max_length, config.calibration_samples, config.device)
    evaluation = tokenize_texts(tokenizer, test_texts, config.max_length, config.evaluation_samples, config.device)

    activation_cache = collect_linear_inputs(model, calibration)
    sensitivity = hessian_trace_proxy(activation_cache)
    base_linear_memory_mb = linear_weight_memory_mb(model, bits=16)

    rows = []
    for ratio in budget_ratios:
        budget_mb = base_linear_memory_mb * ratio
        quantized = copy.deepcopy(model)
        quantized, assignment = apply_mixed_precision(
            quantized,
            config.candidate_bits,
            sensitivity,
            budget_mb,
        )
        metrics = evaluate_perplexity(quantized, evaluation)
        row = {
            "method": "mixed_precision",
            "budget_ratio": ratio,
            "budget_mb": budget_mb,
            "linear_memory_mb": assignment_memory_mb(model, assignment),
            "latency_ms": measure_latency_ms(quantized, evaluation),
            "bit_summary": summarize_assignment(assignment),
            "assignment": str(assignment),
            **metrics,
        }
        rows.append(row)
        print(row)

    output = TABLES_DIR / "resource_sweep.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    points = [
        {
            "method": f"mp-{row['budget_ratio']:.2f}",
            "memory_mb": row["linear_memory_mb"],
            "perplexity": row["perplexity"],
        }
        for row in rows
    ]
    frontier = pareto_frontier(points)
    plot_pareto(frontier, FIGURES_DIR / "resource_sweep_pareto.png")

    print(df)
    print(f"Saved table to {output}")
    print(f"Saved Pareto figure to {FIGURES_DIR / 'resource_sweep_pareto.png'}")


if __name__ == "__main__":
    main()
