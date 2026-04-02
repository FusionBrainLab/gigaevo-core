from __future__ import annotations

import os
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _BASE_DIR.parents[4]
_PROBLEM_DATA_ROOT = _BASE_DIR / "dataset"
_LOCAL_DATA_ROOT = _WORKSPACE_ROOT / "data" / "datasets" / "sudoku"
_LEGACY_DILIGENT_DATA_ROOT = (
    _WORKSPACE_ROOT / "diligent-learner" / "data" / "datasets" / "sudoku"
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USER_PROMPT_TEMPLATE = "Steps so far:\n{ctx}\n\nNext step:"

PROMPT_CONFIG = {
    "layout": "rows_sep",
    "empty_symbol": "_",
}

MODEL_CONFIG = {
    "model_name": os.environ.get("GIGAEVO_SUDOKU_MODEL", "Qwen/Qwen3-4B"),
    "generation": {
        "max_new_tokens": _env_int("GIGAEVO_SUDOKU_MAX_NEW_TOKENS", 128),
        "temperature": _env_float("GIGAEVO_SUDOKU_TEMPERATURE", 0.0),
        "top_p": _env_float("GIGAEVO_SUDOKU_TOP_P", 1.0),
        "repetition_penalty": _env_float("GIGAEVO_SUDOKU_REPETITION_PENALTY", 1.0),
    },
    "gpu_memory_utilization": _env_float("GIGAEVO_SUDOKU_GPU_MEMORY_UTILIZATION", 0.7),
    "max_model_len": _env_int("GIGAEVO_SUDOKU_MAX_MODEL_LEN", 2048),
    "bf16": _env_bool("GIGAEVO_SUDOKU_BF16", True),
}

DATASET_CONFIG = {
    "train_path": os.environ.get(
        "GIGAEVO_SUDOKU_TRAIN_DATASET_PATH",
        str(_PROBLEM_DATA_ROOT / "easy12_4"),
    ),
    "test_path": os.environ.get(
        "GIGAEVO_SUDOKU_TEST_DATASET_PATH",
        str(_PROBLEM_DATA_ROOT / "easy12_4"),
    ),
    "fallback_path": os.environ.get(
        "GIGAEVO_SUDOKU_DATASET_PATH",
        str(_PROBLEM_DATA_ROOT / "easy12_4"),
    ),
    "legacy_train_path": str(_LEGACY_DILIGENT_DATA_ROOT / "medium4"),
    "legacy_test_path": str(_LEGACY_DILIGENT_DATA_ROOT / "medium4_val"),
}

VALIDATION_CONFIG = {
    "train_examples": _env_int("GIGAEVO_SUDOKU_TRAIN_EXAMPLES", 16),
    "test_examples": _env_int("GIGAEVO_SUDOKU_TEST_EXAMPLES", 32),
    "max_steps": _env_int("GIGAEVO_SUDOKU_MAX_STEPS", 25),
    "log_level": os.environ.get("GIGAEVO_SUDOKU_LOG_LEVEL", "WARNING"),
}


def _dataset_candidates(split: str) -> list[Path]:
    train_path = Path(DATASET_CONFIG["train_path"]).expanduser().resolve()
    test_path = Path(DATASET_CONFIG["test_path"]).expanduser().resolve()
    fallback_path = Path(DATASET_CONFIG["fallback_path"]).expanduser().resolve()
    legacy_train_path = Path(DATASET_CONFIG["legacy_train_path"]).expanduser().resolve()
    legacy_test_path = Path(DATASET_CONFIG["legacy_test_path"]).expanduser().resolve()

    if split == "train":
        return [
            train_path,
            fallback_path,
            test_path,
            legacy_train_path,
            legacy_test_path,
        ]
    if split == "test":
        return [
            test_path,
            fallback_path,
            train_path,
            legacy_test_path,
            legacy_train_path,
        ]
    raise ValueError(f"Unknown split: {split}")


def _is_valid_dataset_dir(path: Path) -> bool:
    required = ("index.yaml", "puzzles.jsonl", "chains.jsonl")
    return path.is_dir() and all((path / name).exists() for name in required)


def resolve_dataset_path(split: str) -> Path:
    for candidate in _dataset_candidates(split):
        if _is_valid_dataset_dir(candidate):
            return candidate

    checked = "\n".join(f"- {candidate}" for candidate in _dataset_candidates(split))
    raise FileNotFoundError(
        "Could not find a usable Sudoku dataset directory for split "
        f"{split!r}. Checked:\n{checked}\n"
        "Each directory must contain index.yaml, puzzles.jsonl, and chains.jsonl. "
        "Set GIGAEVO_SUDOKU_TRAIN_DATASET_PATH / "
        "GIGAEVO_SUDOKU_TEST_DATASET_PATH / GIGAEVO_SUDOKU_DATASET_PATH "
        "to point to your generated local Sudoku dataset."
    )


def load_baseline() -> str:
    """Load baseline prompt template from initial_programs/baseline.py."""
    baseline_path = _BASE_DIR / "initial_programs" / "baseline.py"
    baseline_globals: dict[str, object] = {}
    exec(baseline_path.read_text(encoding="utf-8"), baseline_globals)
    return baseline_globals["entrypoint"]()
