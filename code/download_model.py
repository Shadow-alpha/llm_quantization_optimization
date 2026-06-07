from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import MODELS_DIR, ensure_model_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download a HuggingFace causal LM for offline experiments.")
    parser.add_argument("--model", default="distilgpt2", help="HuggingFace model id, e.g. distilgpt2.")
    parser.add_argument(
        "--output",
        default=None,
        help="Local output directory. Defaults to models/<model-name-with-slashes-replaced>.",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        help="Optional HuggingFace endpoint mirror, e.g. https://hf-mirror.com.",
    )
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    return parser.parse_args()


def default_output_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id.replace("/", "__")


def main():
    args = parse_args()
    ensure_model_dir()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror

    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output) if args.output else default_output_dir(args.model)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)

    tokenizer.save_pretrained(output)
    model.save_pretrained(output)

    print(f"Saved model and tokenizer to: {output.resolve()}")
    print("Use it with:")
    print(f"python code/main.py --model-path {output} --local-files-only --method uniform --bits 8")


if __name__ == "__main__":
    main()
