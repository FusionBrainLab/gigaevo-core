"""Typed module-level constants consumed by the schema layer.

Each scalar is a ``Final``-typed module-level binding grouped by
domain (pipeline, redis, evolution, islands, runner, llm, logging,
endpoints). Experiment files and preset builders import these
constants and pass them to schema constructors so the default values
are shared across the typed surface from a single source.

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
DEFAULT_OPTIMIZATION_TIME_BUDGET_S: Final[float | None] = None
DEFAULT_DAG_CONCURRENCY: Final[int] = 16
DEFAULT_MAX_CODE_LENGTH: Final[int] = 30_000
DEFAULT_MAX_INSIGHTS: Final[int] = 8


# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

DEFAULT_REDIS_MAX_CONNECTIONS: Final[int] = 150
DEFAULT_REDIS_CONNECTION_POOL_TIMEOUT_S: Final[float] = 45.0
DEFAULT_REDIS_HEALTH_CHECK_INTERVAL_S: Final[int] = 120
DEFAULT_REDIS_MAX_RETRIES: Final[int] = 5
DEFAULT_REDIS_RETRY_DELAY_S: Final[float] = 0.5


# ---------------------------------------------------------------------------
# Evolution engine
# ---------------------------------------------------------------------------

DEFAULT_LOOP_INTERVAL_S: Final[float] = 1.0
DEFAULT_MAX_ELITES_PER_GENERATION: Final[int] = 5
DEFAULT_MAX_MUTATIONS_PER_GENERATION: Final[int] = 8
DEFAULT_NUM_PARENTS: Final[int] = 2
DEFAULT_MUTATION_MODE: Final[str] = "rewrite"
DEFAULT_MAX_GENERATIONS: Final[int | None] = None
DEFAULT_STRIP_COMMENTS_AND_DOCSTRINGS: Final[bool] = False


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
# schema layer can import a single value instead of crossing into the
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
DEFAULT_LLM_MAX_TOKENS: Final[int] = 81_920
DEFAULT_LLM_TOP_P: Final[float] = 0.95
DEFAULT_LLM_TOP_K: Final[int] = 20
DEFAULT_LLM_REQUEST_TIMEOUT_S: Final[int] = 600


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DEFAULT_LOG_ROTATION: Final[str] = "50 MB"
DEFAULT_LOG_RETENTION: Final[str] = "30 days"
DEFAULT_LOG_TAG: Final[str] = "experiment"
# ``log_dir`` has no module-level scalar -- the resolved output
# directory comes from ``ExperimentConfig.output_dir / experiment_id``
# at run time and is threaded into ``LoggingConfig.build()``.


