from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, ensure_data_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Download parquet files from a HuggingFace dataset repo.")
    parser.add_argument("--dataset", default="Salesforce/wikitext", help="HuggingFace dataset repo id.")
    parser.add_argument("--config", default="wikitext-2-raw-v1", help="Dataset config/subset name.")
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


def select_parquet_files(files: list[str], config: str) -> list[str]:
    parquet_files = [file for file in files if file.endswith(".parquet")]
    if not config:
        return parquet_files

    config_key = config.lower()
    matched = [file for file in parquet_files if config_key in file.lower()]
    return matched or parquet_files


def main():
    args = parse_args()
    ensure_data_dir()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror

    from huggingface_hub import HfApi, hf_hub_download

    output = Path(args.output) if args.output else default_output_dir(args.dataset, args.config)
    output.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(args.dataset, repo_type="dataset")
    parquet_files = select_parquet_files(repo_files, args.config)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in dataset repo: {args.dataset}")

    print(f"Downloading {len(parquet_files)} parquet files from {args.dataset}...")
    for filename in parquet_files:
        path = hf_hub_download(
            repo_id=args.dataset,
            filename=filename,
            repo_type="dataset",
            local_dir=output,
        )
        print(f"  {filename} -> {path}")

    print(f"Saved parquet dataset files to: {output.resolve()}")
    print("Use it with:")
    print(f"python code/main.py --dataset-path {output} --method uniform --bits 8")


if __name__ == "__main__":
    main()
