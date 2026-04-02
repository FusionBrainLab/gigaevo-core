from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .runtime_config import (
        deep_get,
        load_settings,
        resolve_local_path,
        resolve_settings_path,
        to_bool,
        to_int,
        to_list,
        to_str,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from runtime_config import (
        deep_get,
        load_settings,
        resolve_local_path,
        resolve_settings_path,
        to_bool,
        to_int,
        to_list,
        to_str,
    )

try:
    from .shared_memory.memory import AmemGamMemory
except ImportError:  # pragma: no cover - direct script execution fallback
    from shared_memory.memory import AmemGamMemory


THIS_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

SETTINGS_PATH = resolve_settings_path()
SETTINGS = load_settings(SETTINGS_PATH)

MEMORY_DIR = resolve_local_path(
    THIS_DIR,
    deep_get(SETTINGS, "paths.checkpoint_dir"),
    default_relative="memory_usage_store/api_exp4",
)
MEMORY_API_URL = os.getenv(
    "MEMORY_API_URL",
    to_str(deep_get(SETTINGS, "api.base_url"), default="http://localhost:8000"),
)
NAMESPACE = os.getenv(
    "MEMORY_NAMESPACE",
    to_str(deep_get(SETTINGS, "api.namespace"), default="exp6"),
)
USE_API = to_bool(
    os.getenv("MEMORY_USE_API"),
    default=to_bool(deep_get(SETTINGS, "api.use_api"), default=True),
)
CHANNEL = to_str(deep_get(SETTINGS, "api.channel"), default="latest")
ENABLE_BM25 = to_bool(deep_get(SETTINGS, "gam.enable_bm25"), default=False)
ALLOWED_GAM_TOOLS = [
    str(tool).strip() for tool in to_list(deep_get(SETTINGS, "gam.allowed_tools"))
]
GAM_PIPELINE_MODE = to_str(
    os.getenv("MEMORY_GAM_PIPELINE_MODE"),
    default=to_str(deep_get(SETTINGS, "gam.pipeline_mode"), default="default"),
)
RAW_GAM_TOP_K_BY_TOOL = deep_get(SETTINGS, "gam.top_k_by_tool", default={})
if isinstance(RAW_GAM_TOP_K_BY_TOOL, dict):
    GAM_TOP_K_BY_TOOL = {
        str(tool).strip(): max(1, to_int(value, default=5))
        for tool, value in RAW_GAM_TOP_K_BY_TOOL.items()
        if str(tool).strip()
    }
else:
    GAM_TOP_K_BY_TOOL = {}


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
    memory = AmemGamMemory(
        checkpoint_path=str(MEMORY_DIR),
        base_url=MEMORY_API_URL,
        use_api=USE_API,
        namespace=NAMESPACE,
        channel=CHANNEL,
        enable_bm25=ENABLE_BM25,
        allowed_gam_tools=ALLOWED_GAM_TOOLS,
        gam_top_k_by_tool=GAM_TOP_K_BY_TOOL,
        gam_pipeline_mode=GAM_PIPELINE_MODE,
    )

    print("\n==============================")
    print("API Memory Demo: Search")
    print("==============================\n")
    print(f"Config file: {SETTINGS_PATH}")
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
