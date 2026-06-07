import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
METHODS = [
    ("uniform", "8"),
    ("uniform", "4"),
    ("gptq_like", "4"),
    ("awq_like", "4"),
    ("smoothquant_like", "8"),
    ("llm_int8_like", "8"),
    ("mixed_precision", "4"),
    ("zeroquant_like", "8"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for method, bits in METHODS:
        command = [sys.executable, str(ROOT / "code" / "main.py"), "--method", method, "--bits", bits]
        if args.model_path:
            command.extend(["--model-path", args.model_path])
        if args.local_files_only:
            command.append("--local-files-only")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
