from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import DATA_DIR, ensure_data_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset for offline experiments.")
    parser.add_argument("--dataset", default="wikitext", help="HuggingFace dataset name.")
    parser.add_argument("--config", default="wikitext-2-raw-v1", help="Dataset config name.")
    parser.add_argument(
        "--output",
        default=None,
        help="Local output directory. Defaults to data/<dataset>__<config>.",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        help="Optional HuggingFace endpoint mirror, e.g. https://hf-mirror.com.",
    )
    return parser.parse_args()


def default_output_dir(dataset_name: str, dataset_config: str) -> Path:
    return DATA_DIR / f"{dataset_name}__{dataset_config}".replace("/", "__")


def main():
    args = parse_args()
    ensure_data_dir()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror

    from datasets import load_dataset

    output = Path(args.output) if args.output else default_output_dir(args.dataset, args.config)
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, args.config)
    dataset.save_to_disk(output)

    print(f"Saved dataset to: {output.resolve()}")
    print("Use it with:")
    print(f"python code/main.py --dataset-path {output} --method uniform --bits 8")


if __name__ == "__main__":
    main()
