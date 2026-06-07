from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExperimentConfig, TABLES_DIR, ensure_result_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="uniform")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--model-path", default=None, help="Local model directory. Overrides config.model_name.")
    parser.add_argument("--dataset-path", default=None, help="Local dataset directory saved by code/download_dataset.py.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_method(method, model, activation_cache, config, bits):
    from models import estimate_parameter_memory_mb
    from quantization.awq_like import apply_awq_like
    from quantization.gptq_like import apply_gptq_like
    from quantization.llm_int8_like import apply_llm_int8_like
    from quantization.mixed_precision import apply_mixed_precision
    from quantization.sensitivity import hessian_trace_proxy
    from quantization.smoothquant_like import apply_smoothquant_like
    from quantization.uniform import apply_uniform_quantization
    from quantization.zeroquant_like import apply_zeroquant_like

    if method == "uniform":
        return apply_uniform_quantization(model, bits), {}
    if method == "gptq_like":
        return apply_gptq_like(model, activation_cache, bits), {}
    if method == "awq_like":
        return apply_awq_like(model, activation_cache, bits), {}
    if method == "smoothquant_like":
        return apply_smoothquant_like(model, activation_cache, bits), {}
    if method == "llm_int8_like":
        return apply_llm_int8_like(model, activation_cache)
    if method == "mixed_precision":
        budget = config.memory_budget_mb or estimate_parameter_memory_mb(model) * 0.35
        sensitivity = hessian_trace_proxy(activation_cache)
        return apply_mixed_precision(model, config.candidate_bits, sensitivity, budget)
    if method == "zeroquant_like":
        return apply_zeroquant_like(model, bits, config.group_size), {}
    raise ValueError(f"Unknown method: {method}")


def main():
    args = parse_args()
    config = ExperimentConfig()
    if args.model_path:
        config.model_path = args.model_path
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    ensure_result_dirs()

    if args.dry_run:
        print("Dry run passed. Configuration:")
        print(config)
        return

    from data import collect_linear_inputs, load_text_splits, tokenize_texts
    from evaluate import evaluate_perplexity, measure_latency_ms
    from models import estimate_parameter_memory_mb, load_causal_lm

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

    baseline = {
        "method": "fp",
        "bits": 16,
        "memory_mb": estimate_parameter_memory_mb(model),
        "latency_ms": measure_latency_ms(model, evaluation),
        **evaluate_perplexity(model, evaluation),
    }

    activation_cache = collect_linear_inputs(model, calibration)
    quantized = copy.deepcopy(model)
    quantized, metadata = apply_method(args.method, quantized, activation_cache, config, args.bits)
    result = {
        "method": args.method,
        "bits": args.bits,
        "memory_mb": estimate_parameter_memory_mb(quantized),
        "latency_ms": measure_latency_ms(quantized, evaluation),
        **evaluate_perplexity(quantized, evaluation),
        "metadata": str(metadata),
    }

    import pandas as pd

    df = pd.DataFrame([baseline, result])
    output = TABLES_DIR / f"{args.method}_{args.bits}.csv"
    df.to_csv(output, index=False)
    print(df)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
