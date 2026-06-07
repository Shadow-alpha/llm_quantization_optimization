from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass
class ExperimentConfig:
    model_name: str = "distilgpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    text_field: str = "text"
    calibration_samples: int = 64
    evaluation_samples: int = 128
    max_length: int = 128
    device: str = "cpu"
    seed: int = 42
    methods: list[str] = field(
        default_factory=lambda: [
            "fp16",
            "uniform",
            "gptq_like",
            "awq_like",
            "smoothquant_like",
            "llm_int8_like",
            "mixed_precision",
            "zeroquant_like",
        ]
    )
    candidate_bits: tuple[int, ...] = (2, 3, 4, 8, 16)
    memory_budget_mb: float | None = None
    group_size: int = 64


def ensure_result_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

