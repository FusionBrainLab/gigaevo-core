"""Regression tests for gigaevo/config/defaults.py.

For each scalar in config/constants/*.yaml the test asserts the
corresponding DEFAULT_* constant in gigaevo.config.defaults carries
the byte-equal value. Any drift between the two surfaces during
Phase 2 migration trips this gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gigaevo.config import defaults as D

CONSTANTS_DIR = Path(__file__).resolve().parents[2] / "config" / "constants"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONSTANTS_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict), f"{path} did not parse as a mapping"
    raw.pop("defaults", None)
    return raw


class TestPipelineConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("pipeline")
        assert D.DEFAULT_STAGE_TIMEOUT_S == y["stage_timeout"]
        assert D.DEFAULT_DAG_TIMEOUT_S == y["dag_timeout"]
        assert D.DEFAULT_OPTIMIZATION_TIME_BUDGET_S == y["optimization_time_budget"]
        assert D.DEFAULT_DAG_CONCURRENCY == y["dag_concurrency"]
        assert D.DEFAULT_MAX_CODE_LENGTH == y["max_code_length"]
        assert D.DEFAULT_MAX_INSIGHTS == y["max_insights"]


class TestRedisConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("redis")
        assert D.DEFAULT_REDIS_MAX_CONNECTIONS == y["redis_max_connections"]
        assert (
            D.DEFAULT_REDIS_CONNECTION_POOL_TIMEOUT_S
            == y["redis_connection_pool_timeout"]
        )
        assert (
            D.DEFAULT_REDIS_HEALTH_CHECK_INTERVAL_S
            == y["redis_health_check_interval"]
        )
        assert D.DEFAULT_REDIS_MAX_RETRIES == y["redis_max_retries"]
        assert D.DEFAULT_REDIS_RETRY_DELAY_S == y["redis_retry_delay"]


class TestEvolutionConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("evolution")
        assert D.DEFAULT_LOOP_INTERVAL_S == y["loop_interval"]
        assert D.DEFAULT_MAX_ELITES_PER_GENERATION == y["max_elites_per_generation"]
        assert (
            D.DEFAULT_MAX_MUTATIONS_PER_GENERATION
            == y["max_mutations_per_generation"]
        )
        assert D.DEFAULT_NUM_PARENTS == y["num_parents"]
        assert D.DEFAULT_MUTATION_MODE == y["mutation_mode"]
        assert D.DEFAULT_MAX_GENERATIONS == y["max_generations"]
        assert D.DEFAULT_STRIP_COMMENTS_AND_DOCSTRINGS == y[
            "strip_comments_and_docstrings"
        ]


class TestIslandsConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("islands")
        assert D.DEFAULT_MIGRATION_INTERVAL == y["migration_interval"]
        assert D.DEFAULT_MAX_MIGRANTS_PER_ISLAND == y["max_migrants_per_island"]
        assert D.DEFAULT_ENABLE_MIGRATION == y["enable_migration"]
        assert D.DEFAULT_ISLAND_ID == y["island_id"]
        assert D.DEFAULT_ISLAND_MAX_SIZE == y["island_max_size"]
        assert D.DEFAULT_PRIMARY_RESOLUTION == y["primary_resolution"]
        assert D.DEFAULT_VALIDITY_RESOLUTION == y["validity_resolution"]
        assert D.DEFAULT_BINNING_TYPE == y["binning_type"]

    def test_validity_key_matches_runtime(self) -> None:
        """The YAML uses ``${get_object:...VALIDITY_KEY}`` which
        resolves to the runtime constant — defaults.py imports the
        same constant directly. The check confirms the two paths
        agree on the spelling."""
        from gigaevo.programs.metrics.context import VALIDITY_KEY

        assert D.DEFAULT_VALIDITY_KEY == VALIDITY_KEY


class TestRunnerConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("runner")
        assert D.DEFAULT_RUNNER_POLL_INTERVAL_S == y["runner_poll_interval"]
        assert D.DEFAULT_MAX_CONCURRENT_DAGS == y["max_concurrent_dags"]


class TestLLMConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("llm")
        assert D.DEFAULT_LLM_TEMPERATURE == y["temperature"]
        assert D.DEFAULT_LLM_MAX_TOKENS == y["max_tokens"]
        assert D.DEFAULT_LLM_TOP_P == y["top_p"]
        assert D.DEFAULT_LLM_TOP_K == y["top_k"]
        assert D.DEFAULT_LLM_REQUEST_TIMEOUT_S == y["request_timeout"]


class TestLoggingConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("logging")
        assert D.DEFAULT_LOG_ROTATION == y["log_rotation"]
        assert D.DEFAULT_LOG_RETENTION == y["log_retention"]
        assert D.DEFAULT_LOG_TAG == y["tag"]
        # log_dir is intentionally not module-level — it's a
        # runtime decision derived from ExperimentConfig.output_dir.
        assert "log_dir" in y
        assert "${hydra:runtime.output_dir}" in y["log_dir"]


class TestEndpointsConstants:
    def test_byte_equal_with_yaml(self) -> None:
        y = _load_yaml("endpoints")
        assert D.DEFAULT_LLM_BASE_URL == y["llm_base_url"]
        assert D.DEFAULT_MODEL_NAME == y["model_name"]


class TestImmutabilitySurface:
    """The constants are ``Final``-typed but Python does not enforce
    Final at runtime. The convention is enforced by mypy + ruff. The
    test below confirms the constants are accessible as module
    attributes and that the canonical naming pattern (DEFAULT_*) is
    followed without drift."""

    def test_every_constant_uses_default_prefix(self) -> None:
        constants = [
            name
            for name in dir(D)
            if not name.startswith("_") and name.isupper()
        ]
        # Strip the re-export
        constants = [c for c in constants if c != "VALIDITY_KEY"]
        # ``Final`` and ``yaml`` aren't constants
        non_default = [c for c in constants if not c.startswith("DEFAULT_")]
        assert non_default == [], f"non-DEFAULT_ constants: {non_default}"

    def test_no_unexpected_modules_in_namespace(self) -> None:
        # Typing helpers and standard-library imports that legitimately
        # appear in the module's namespace but are not constants.
        _ALLOWED_NON_CONSTANTS = {"Final", "annotations"}
        public_names = {n for n in dir(D) if not n.startswith("_")}
        non_constants = {
            n for n in public_names if not n.startswith("DEFAULT_")
        } - _ALLOWED_NON_CONSTANTS
        assert non_constants == set(), (
            f"unexpected public symbols leaked into defaults: "
            f"{sorted(non_constants)}"
        )
