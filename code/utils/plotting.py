from pathlib import Path

import matplotlib.pyplot as plt


def plot_pareto(points, output: Path):
    if not points:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    xs = [p["memory_mb"] for p in points]
    ys = [p["perplexity"] for p in points]
    labels = [p["method"] for p in points]
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), fontsize=8)
    plt.xlabel("Memory (MB)")
    plt.ylabel("Perplexity")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

