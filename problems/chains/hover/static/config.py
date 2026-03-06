"""Configuration for HoVer static chain evolution."""

from pathlib import Path

# --- HoVer Chain Topology (static mode, 7 steps) ---

STATIC_CHAIN_TOPOLOGY = {
    "num_steps": 7,
    "steps": [
        {"number": 1, "step_type": "tool", "dependencies": [], "frozen": True},
        {"number": 2, "step_type": "llm", "dependencies": [1], "frozen": False},
        {"number": 3, "step_type": "llm", "dependencies": [2], "frozen": False},
        {"number": 4, "step_type": "tool", "dependencies": [3], "frozen": True},
        {"number": 5, "step_type": "llm", "dependencies": [2, 4], "frozen": False},
        {"number": 6, "step_type": "llm", "dependencies": [5], "frozen": False},
        {"number": 7, "step_type": "tool", "dependencies": [6], "frozen": True},
    ],
}

_EXPERIMENT_DIR = Path(__file__).parent


def load_baseline() -> dict:
    """Load baseline chain specification from initial_programs/baseline.py.

    Returns:
        Dict with "system_prompt" and "steps" keys.
    """
    baseline_path = _EXPERIMENT_DIR / "initial_programs" / "baseline.py"
    baseline_globals = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
