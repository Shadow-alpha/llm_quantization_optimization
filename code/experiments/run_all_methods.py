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


def main():
    for method, bits in METHODS:
        subprocess.run([sys.executable, str(ROOT / "code" / "main.py"), "--method", method, "--bits", bits], check=True)


if __name__ == "__main__":
    main()

