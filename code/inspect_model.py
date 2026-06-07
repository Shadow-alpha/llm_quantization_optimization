from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Print model module structure and quantizable layer coverage.")
    parser.add_argument("--model-path", default=None, help="Local model directory. Overrides config.model_name.")
    parser.add_argument("--all-layers", action="store_true", help="Print every named module, not only quantizable ones.")
    return parser.parse_args()


def format_shape(shape) -> str:
    return "x".join(str(item) for item in shape)


def main():
    args = parse_args()
    config = ExperimentConfig()
    if args.model_path:
        config.model_path = args.model_path

    from models import get_layer_weight, is_quantizable_layer, iter_quantizable_layers, load_causal_lm

    model_source = config.model_path or config.model_name
    model, _tokenizer = load_causal_lm(model_source, config.device)

    if args.all_layers:
        print("All named modules:")
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            weight_shape = format_shape(tuple(weight.shape)) if weight is not None else "-"
            mark = "quantizable" if is_quantizable_layer(module) else "-"
            print(f"{name or '<root>'}\t{module.__class__.__name__}\tweight={weight_shape}\t{mark}")
        print()

    print("Quantizable layers:")
    total_params = 0
    count = 0
    for name, module in iter_quantizable_layers(model):
        weight = get_layer_weight(module)
        params = weight.numel()
        total_params += params
        count += 1
        print(f"{count:03d}\t{name}\t{module.__class__.__name__}\tweight={format_shape(tuple(weight.shape))}\tparams={params}")

    memory_mb = total_params * 16 / 8 / 1024 / 1024
    print()
    print(f"quantizable_layer_count={count}")
    print(f"quantizable_weight_params={total_params}")
    print(f"quantizable_weight_memory_16bit_mb={memory_mb:.4f}")


if __name__ == "__main__":
    main()
