"""Typed module-level constants consumed by the preset builders.

Each scalar is a ``Final``-typed module-level binding grouped by domain
(pipeline, evolution, islands, runner, llm). The preset builders in
``gigaevo.config.*_presets`` import these constants and thread them
into the schema constructors so the default values for shipped
experiments come from a single source.

The naming convention is ``DEFAULT_<DOMAIN>_<KNOB>``; the
``test_defaults.TestImmutabilitySurface`` checks both that every
public symbol follows the prefix and that no foreign module leaks
into the namespace.
"""

from __future__ import annotations

from typing import Final

from gigaevo.programs.metrics.context import VALIDITY_KEY as _VALIDITY_KEY

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

DEFAULT_STAGE_TIMEOUT_S: Final[int] = 2400
DEFAULT_DAG_TIMEOUT_S: Final[int] = 7200


# ---------------------------------------------------------------------------
# Evolution engine
# ---------------------------------------------------------------------------

DEFAULT_LOOP_INTERVAL_S: Final[float] = 1.0
DEFAULT_MAX_ELITES_PER_GENERATION: Final[int] = 5
DEFAULT_MAX_MUTATIONS_PER_GENERATION: Final[int] = 8
DEFAULT_NUM_PARENTS: Final[int] = 2
DEFAULT_MAX_GENERATIONS: Final[int | None] = None


# ---------------------------------------------------------------------------
# MAP-Elites islands
# ---------------------------------------------------------------------------

DEFAULT_MIGRATION_INTERVAL: Final[int] = 25
DEFAULT_MAX_MIGRANTS_PER_ISLAND: Final[int] = 5
DEFAULT_ENABLE_MIGRATION: Final[bool] = True
DEFAULT_ISLAND_ID: Final[str] = "fitness_island"
DEFAULT_ISLAND_MAX_SIZE: Final[int] = 75
DEFAULT_PRIMARY_RESOLUTION: Final[int] = 150
DEFAULT_VALIDITY_RESOLUTION: Final[int] = 2
DEFAULT_BINNING_TYPE: Final[str] = "linear"
# The validity-metric key lives next to the rest of the metrics-context
# vocabulary; re-export it under the canonical constant name so the
# preset layer can import a single value instead of crossing into the
# metrics package directly.
DEFAULT_VALIDITY_KEY: Final[str] = _VALIDITY_KEY


# ---------------------------------------------------------------------------
# DAG runner
# ---------------------------------------------------------------------------

DEFAULT_RUNNER_POLL_INTERVAL_S: Final[float] = 5.0
DEFAULT_MAX_CONCURRENT_DAGS: Final[int] = 10


# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_TEMPERATURE: Final[float] = 0.6
# 80 * 1024 — accommodates the long-context output ceilings on the
# Gemini 2.5/3 family without truncating multi-step reasoning traces.
DEFAULT_LLM_MAX_TOKENS: Final[int] = 81_920
DEFAULT_LLM_REQUEST_TIMEOUT_S: Final[int] = 600
