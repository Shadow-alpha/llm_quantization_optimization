import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    command = [sys.executable, str(ROOT / "code" / "main.py"), "--method", "uniform", "--bits", "8"]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if args.dataset_path:
        command.extend(["--dataset-path", args.dataset_path])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
