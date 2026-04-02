"""Configuration for HoVer full_chain chain evolution (standard retrieval only)."""

from pathlib import Path

# --- Full Chain Validation Config ---
# No retrieve_deep — only standard BM25 retrieval (k=7).
# Programs referencing retrieve_deep will be rejected by validate_chain_spec.

FULL_CHAIN_CONFIG = {
    "max_steps": 10,
    "allowed_step_types": ["llm", "tool"],
    "available_tools": ["retrieve"],
    "require_final_llm": False,
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
