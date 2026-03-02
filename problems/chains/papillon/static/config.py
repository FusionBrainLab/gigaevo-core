"""Configuration for PAPILLON static chain evolution.

3-step topology: LLM (redact) → TOOL (external_llm) → LLM (aggregate).
Step 2 is frozen (tool step) to ensure the external model receives ONLY
the redacted query. Steps 1 and 3 are evolvable LLM steps.
"""

from pathlib import Path

# --- PAPILLON Chain Topology (static mode, 3 steps) ---

STATIC_CHAIN_TOPOLOGY = {
    "num_steps": 3,
    "steps": [
        {"number": 1, "step_type": "llm", "dependencies": [], "frozen": False},
        {"number": 2, "step_type": "tool", "dependencies": [1], "frozen": True},
        {"number": 3, "step_type": "llm", "dependencies": [1, 2], "frozen": False},
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
