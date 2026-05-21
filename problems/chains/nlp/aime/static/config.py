"""Configuration for nlp/aime chain evolution (full_chain mode, up to 3 LLM steps)."""

from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).parent


def load_baseline() -> dict:
    """Load baseline chain specification from initial_programs/baseline.py."""
    baseline_path = _EXPERIMENT_DIR / "initial_programs" / "baseline.py"
    baseline_globals: dict = {}
    exec(baseline_path.read_text(), baseline_globals)
    return baseline_globals["entrypoint"]()
