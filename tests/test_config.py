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
    An accidental change here would silently alter experimental conditions without
    a config-review gate catching it.

    v2.0.0 baseline: one-parent archetype mutation (num_parents=1) is the canonical
    operator; max_mutants=250 sizes a one-workday-at-parallelism-5 sweep.
    """
    cfg = _compose()
    assert cfg.max_elites_per_generation == 5, (
        "max_elites_per_generation changed from 5 — update CONTEXT.md for active experiments"
    )
    assert cfg.num_parents == 1, (
        "num_parents changed from 1 — one-parent archetype mutation is the v2 canonical "
        "operator. If you intentionally bump to 2 (or more), update this assertion and "
        "the canonical-benchmark contract together."
    )
    assert cfg.loop_interval == pytest.approx(1.0), (
        "loop_interval changed from 1.0 — affects engine polling frequency"
    )
    assert cfg.max_mutants == 250, (
        "max_mutants changed from 250 — the default stopper "
        "(config/stopper/max_mutants.yaml) resolves this top-level value. "
        "Bumping this voids comparability with prior BENCHMARK_HISTORY.md rows."
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


def test_sync_min_delta_is_independent_constant():
    """sync_min_delta must be an independent config constant.

    The ProgressBasedSyncHook's min_delta/drift_cap controls how many programs
    the opponent must process before this population unblocks. It is its own
    top-level constant, decoupled from any generation-sized batching knob.
    """
    # 1. sync_min_delta exists as a top-level constant with default 8
    cfg = _compose()
    assert cfg.sync_min_delta == 8, "sync_min_delta should default to 8"

    # 2. Both adversarial pipelines wire the drift/sync value from sync_min_delta.
    # ProgressBasedSyncHook accepts either `drift_cap` (preferred) or `min_delta`
    # (legacy alias). Pipelines differ in which key they use:
    #   adversarial_asymmetric: min_delta: ${sync_min_delta}
    #   adversarial_coevo_ss:   drift_cap: ${sync_min_delta}
    _ADVERSARIAL_OVERRIDES = [
        "opponent_redis_db=0",
        "opponent_redis_prefix=test",
    ]

    def _hook_sync_value(hook_cfg) -> int:
        if "min_delta" in hook_cfg:
            return hook_cfg.min_delta
        return hook_cfg.drift_cap

    for pipeline in ("adversarial_asymmetric", "adversarial_coevo_ss"):
        cfg = _compose(f"pipeline={pipeline}", *_ADVERSARIAL_OVERRIDES)
        assert _hook_sync_value(cfg.pre_step_hook) == 8, (
            f"pipeline={pipeline}: pre_step_hook sync value should be 8 "
            f"(from sync_min_delta), got {_hook_sync_value(cfg.pre_step_hook)}"
        )

    # 3. sync_min_delta can be overridden independently.
    for pipeline in ("adversarial_asymmetric", "adversarial_coevo_ss"):
        cfg = _compose(
            f"pipeline={pipeline}",
            "sync_min_delta=1",
            *_ADVERSARIAL_OVERRIDES,
        )
        assert _hook_sync_value(cfg.pre_step_hook) == 1, (
            f"pipeline={pipeline}: overriding sync_min_delta=1 should set hook sync=1"
        )


def test_redis_prefix_resolves_to_problem_name():
    """${redis.prefix} must resolve everywhere it's referenced (I-12).

    Pipelines such as adversarial_coevo_ss.yaml reference ${redis.prefix} for
    ProgressBasedSyncHook.own_prefix. Before the I-12 fix, `redis.prefix` was
    never defined, so Hydra raised InterpolationKeyError at run time. The fix
    defines `redis.prefix: ${problem.name}` in config/redis/default.yaml so
    the intuitive "redis.prefix = my namespace" mental model actually works.
    """
    cfg = _compose()
    assert cfg.redis.prefix == "_test_", (
        "redis.prefix should resolve to ${problem.name} — did you delete the "
        "`prefix:` line in config/redis/default.yaml? (regression of I-12)"
    )

    # Any pipeline that references ${redis.prefix} must now resolve cleanly.
    cfg = _compose(
        "pipeline=adversarial_coevo_ss",
        "opponent_redis_db=2",
        "opponent_redis_prefix=_test_opponent_",
    )
    assert cfg.pre_step_hook.own_prefix == "_test_", (
        "adversarial_coevo_ss.pre_step_hook.own_prefix must resolve to "
        "${problem.name} via ${redis.prefix} (I-12)"
    )


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
