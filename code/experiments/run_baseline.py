from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    subprocess.run([sys.executable, str(ROOT / "code" / "main.py"), "--method", "uniform", "--bits", "8"], check=True)


if __name__ == "__main__":
    main()

