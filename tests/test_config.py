"""Validates that all YAML configs in config/ compose correctly with Hydra.

These tests catch two classes of bug:
  1. Composition errors — a YAML references a non-existent group or has bad syntax.
  2. @package bugs — a config group uses the wrong @package directive so its keys
     land at the wrong level in the composed config (e.g. @package _global_ in a
     prompts/ file causes prompts.dir to be missing).
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest

CONFIG_DIR = Path(__file__).parent.parent / "config"

# problem.name is required (???); supply a dummy value so composition succeeds.
_BASE_OVERRIDES = ["problem.name=_test_"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exists(cfg, path: str) -> bool:
    """Return True if *path* exists in cfg without resolving interpolations.

    Uses _get_node() traversal so that keys whose values are unresolvable
    interpolations (e.g. ${hydra:runtime.cwd}) still count as present.
    """
    node = cfg
    for key in path.split("."):
        if not OmegaConf.is_dict(node) or key not in node:
            return False
        node = node._get_node(key)
    return True


def _compose(*overrides: str):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(
            config_name="config", overrides=_BASE_OVERRIDES + list(overrides)
        )


def _group_choices(group: str) -> list[str]:
    """Return non-private YAML stems in a config group directory."""
    return [
        f.stem
        for f in sorted((CONFIG_DIR / group).glob("*.yaml"))
        if not f.name.startswith("_")
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# Default config — structural assertions
#
# Each path here documents an invariant that must hold.  Add a new assertion
# whenever a new @package directive or config group is introduced.
# ---------------------------------------------------------------------------

# Paths that must exist in every composed config.
# Key invariant: if a prompts/*.yaml uses @package _global_ by mistake,
# cfg.prompts.dir will be absent (the key lands at the root instead).
_REQUIRED_PATHS = [
    "prompts.dir",  # catches @package _global_ in prompts/ files
    "evolution_strategy",  # set by algorithm/ group
    "dag_blueprint",  # set by pipeline/ group
    "dag_runner",  # set by runner/ group
]


def test_default_config_composes():
    cfg = _compose()
    for path in _REQUIRED_PATHS:
        assert _exists(cfg, path), (
            f"Path '{path}' missing from default composed config — "
            f"possible wrong @package directive or missing defaults entry"
        )


def test_prompts_dir_not_at_root():
    """prompts.dir must not appear as a top-level 'dir' key (classic @package bug)."""
    cfg = _compose()
    # If prompts/default.yaml used @package _global_, 'dir' would leak to the root.
    assert not _exists(cfg, "dir"), (
        "Spurious top-level 'dir' key found — prompts/*.yaml likely uses "
        "@package _global_ instead of @package prompts"
    )


# ---------------------------------------------------------------------------
# Parametrized group variant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", _group_choices("experiment"))
def test_experiment_variant_composes(variant: str):
    cfg = _compose(f"experiment={variant}")
    assert _exists(cfg, "prompts.dir"), f"prompts.dir missing with experiment={variant}"
    assert _exists(cfg, "dag_blueprint"), (
        f"dag_blueprint missing with experiment={variant}"
    )


@pytest.mark.parametrize("variant", _group_choices("algorithm"))
def test_algorithm_variant_composes(variant: str):
    cfg = _compose(f"algorithm={variant}")
    assert _exists(cfg, "evolution_strategy._target_"), (
        f"evolution_strategy._target_ missing with algorithm={variant}"
    )


@pytest.mark.parametrize("variant", _group_choices("pipeline"))
def test_pipeline_variant_composes(variant: str):
    cfg = _compose(f"pipeline={variant}")
    assert _exists(cfg, "dag_blueprint._target_"), (
        f"dag_blueprint._target_ missing with pipeline={variant}"
    )
    # Every pipeline config must be able to reach prompts.dir
    assert _exists(cfg, "prompts.dir"), (
        f"prompts.dir missing with pipeline={variant} — check @package directive"
    )


@pytest.mark.parametrize("variant", _group_choices("prompts"))
def test_prompts_variant_composes(variant: str):
    cfg = _compose(f"prompts={variant}")
    assert _exists(cfg, "prompts.dir"), (
        f"prompts.dir missing with prompts={variant} — "
        f"check @package directive (must be '# @package prompts')"
    )


@pytest.mark.parametrize("variant", _group_choices("llm"))
def test_llm_variant_composes(variant: str):
    _compose(f"llm={variant}")


@pytest.mark.parametrize("variant", _group_choices("metrics"))
def test_metrics_variant_composes(variant: str):
    _compose(f"metrics={variant}")


@pytest.mark.parametrize("variant", _group_choices("loader"))
def test_loader_variant_composes(variant: str):
    _compose(f"loader={variant}")


@pytest.mark.parametrize("variant", _group_choices("logging"))
def test_logging_variant_composes(variant: str):
    _compose(f"logging={variant}")


# ---------------------------------------------------------------------------
# Constants invariants — catch accidental value changes
#
# These tests document the canonical values for key constants.  If you
# intentionally change a constant, update both the YAML and this test.
# ---------------------------------------------------------------------------


def test_evolution_constants_default_values():
    """Evolution engine constants must match their documented defaults.

    These are the values that determine experiment throughput and reproducibility.
    An accidental change here (e.g. max_mutations drifting to 16) would silently
    alter experimental conditions without a config-review gate catching it.
    """
    cfg = _compose()
    assert cfg.max_elites_per_generation == 5, (
        "max_elites_per_generation changed from 5 — update CONTEXT.md for active experiments"
    )
    assert cfg.max_mutations_per_generation == 8, (
        "max_mutations_per_generation changed from 8 — verify throughput calculations"
    )
    assert cfg.num_parents == 2, (
        "num_parents changed from 2 — crossover experiments use this default"
    )
    assert cfg.loop_interval == pytest.approx(1.0), (
        "loop_interval changed from 1.0 — affects engine polling frequency"
    )
    assert cfg.max_generations is None, (
        "max_generations should default to None (run until stopped)"
    )


def test_pipeline_constants_default_values():
    """Pipeline execution timeout constants must match their documented values."""
    cfg = _compose()
    assert cfg.stage_timeout == 2400, (
        "stage_timeout changed from 2400s — validate.py runs need at least 40 min"
    )
    assert cfg.dag_timeout == 7200, (
        "dag_timeout changed from 7200s — full DAG needs 2 hours for slow eval runs"
    )
    assert cfg.dag_concurrency == 16, "dag_concurrency changed from 16"
    assert cfg.max_code_length == 30000, "max_code_length changed from 30000 chars"


@pytest.mark.parametrize("variant", _group_choices("pipeline"))
def test_pipeline_evolution_context_prompts_dir_when_defined(variant: str):
    """Any pipeline that defines evolution_context must include prompts_dir in it.

    This is a regression guard for the config bug fixed in commit 920c975:
    pipelines that define their own evolution_context block were missing
    prompts_dir, causing custom prompts to be silently ignored.

    The rule: if a pipeline composes an evolution_context key at all, that
    evolution_context must contain prompts_dir (so it can be interpolated as
    ${prompts.dir} at runtime).
    """
    cfg = _compose(f"pipeline={variant}")
    if not _exists(cfg, "evolution_context"):
        return  # pipeline doesn't define evolution_context — nothing to check
    assert _exists(cfg, "evolution_context.prompts_dir"), (
        f"pipeline={variant} defines evolution_context but is missing "
        f"evolution_context.prompts_dir — custom prompts will be silently ignored "
        f"(regression of bug 920c975)"
    )
