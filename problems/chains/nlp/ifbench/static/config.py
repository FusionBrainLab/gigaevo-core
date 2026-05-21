"""Configuration for nlp/ifbench static chain evolution."""

from pathlib import Path

STATIC_CHAIN_TOPOLOGY = {
    "num_steps": 2,
    "steps": [
        {"number": 1, "step_type": "llm", "dependencies": [], "frozen": False},
        {"number": 2, "step_type": "llm", "dependencies": [1], "frozen": False},
    ],
}

_EXPERIMENT_DIR = Path(__file__).parent


def load_baseline() -> dict:
    """Load baseline chain specification from initial_programs/baseline.py."""
    baseline_path = _EXPERIMENT_DIR / "initial_programs" / "baseline.py"
    baseline_globals: dict = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
