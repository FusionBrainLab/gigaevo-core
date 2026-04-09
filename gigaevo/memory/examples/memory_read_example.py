"""Example: search the memory backend using the same query format as MemorySelectorAgent.

Run with:
    PYTHONPATH=. $GIGAEVO_PYTHON gigaevo/memory/examples/memory_read_example.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import (
    ApiConfig,
    GamConfig,
    MemoryConfig,
)
from gigaevo.memory.write_pipeline_config import load_config

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

_cfg = load_config()

MEMORY_DIR = _cfg.memory_dir
MEMORY_API_URL = str(_cfg.memory_api_url)
NAMESPACE = _cfg.namespace
USE_API = _cfg.use_api
CHANNEL = _cfg.channel
ENABLE_BM25 = _cfg.enable_bm25
ALLOWED_GAM_TOOLS = list(_cfg.allowed_gam_tools)
GAM_PIPELINE_MODE = _cfg.gam_pipeline_mode or "default"
GAM_TOP_K_BY_TOOL = dict(_cfg.gam_top_k_by_tool)


def _build_memory_selector_style_request() -> str:
    """Example request payload aligned with gigaevo/prompts/memory_selector/user.txt."""
    return """MUTATION INPUTS

TASK DESCRIPTION:
Place 11 distinct points inside a unit-area equilateral triangle to maximize the minimum triangle area (Heilbronn 11-point target >= 0.0365).

AVAILABLE METRICS:
- fitness: min_triangle_area (higher is better)
- valid_points_rate
- boundary_violation_count
- near_collinearity_count

MUTATION MODE:
guided_innovation

PARENTS (same parent code + mutation context given to mutation agent):
=== Parent 1 ===
```python
def entrypoint():
    # baseline: optimize with smooth min objective
    ...
```
Parent context:
- Last fitness: 0.0349
- Weakness: frequent near-collinear triplets
- Prior change: switched to L-BFGS-B with moderate smoothing

=== Parent 2 ===
```python
def entrypoint():
    # baseline variant: basin hopping + SLSQP constraints
    ...
```
Parent context:
- Last fitness: 0.0352
- Weakness: unstable local convergence at final stage
- Prior change: higher exploration noise

"""


def main() -> None:
    api_config = (
        ApiConfig(
            base_url=MEMORY_API_URL or "http://localhost:8000",
            namespace=NAMESPACE or "default",
            channel=CHANNEL or "latest",
        )
        if USE_API
        else None
    )
    mem_config = MemoryConfig(
        checkpoint_path=MEMORY_DIR,
        api=api_config,
        gam=GamConfig(
            enable_bm25=ENABLE_BM25,
            allowed_tools=ALLOWED_GAM_TOOLS or [],
            top_k_by_tool=GAM_TOP_K_BY_TOOL or {},
            pipeline_mode=GAM_PIPELINE_MODE,
        ),
    )
    memory = AmemGamMemory(config=mem_config)

    print("\n==============================")
    print("API Memory Demo: Search")
    print("==============================\n")
    print(f"Config file: {_cfg.settings_path}")
    print(f"Using API: {USE_API}")

    query = _build_memory_selector_style_request()

    print(">>> MEMORY SELECTOR STYLE INPUT:\n")
    print(query)
    print()
    try:
        result = memory.search(query)
    except RuntimeError as exc:
        print(f"Search failed: {exc}")
        return
    print("Result:\n")
    print(result)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
