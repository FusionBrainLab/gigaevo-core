"""Memory crediting Hydra group: point default, paired opt-in + compose guard.

``memory/crediting`` must be a real treatment switch: the writer node consumes
the group by reference, and ``memory/crediting=paired`` under a pipeline that
never routes per-sample scores is the inert-treatment trap the compose guard
exists to reject.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from gigaevo.config.resolvers import register_resolvers
from gigaevo.config.validation import validate_crediting_pipeline_compat
from gigaevo.memory.write.crediting import PairedEffectEstimator, PointEffectEstimator

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_BASE = ["problem.name=_test_"]

POINT = "gigaevo.memory.write.crediting.PointEffectEstimator"
PAIRED = "gigaevo.memory.write.crediting.PairedEffectEstimator"


def _compose(*overrides: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=[*_BASE, *overrides])


@pytest.mark.parametrize("memory", ["full", "writer"])
def test_default_is_point_and_writer_consumes_the_group(memory: str) -> None:
    cfg = _compose(f"memory={memory}")

    assert cfg.memory.crediting._target_ == POINT
    raw_writer = OmegaConf.to_container(cfg.memory.writer, resolve=False)
    assert raw_writer["effect_estimator"] == "${ref:memory.crediting}"


def test_paired_override_reaches_the_group_node() -> None:
    cfg = _compose("memory=full", "memory/crediting=paired")

    assert cfg.memory.crediting._target_ == PAIRED
    assert cfg.memory.crediting.comparison.n_resamples == 2000
    assert cfg.memory.crediting.comparison.seed == 0


def test_paired_without_metadata_pipeline_is_rejected() -> None:
    cfg = _compose("memory=full", "memory/crediting=paired")

    with pytest.raises(ValueError, match="routes_program_metadata"):
        validate_crediting_pipeline_compat(cfg)


@pytest.mark.parametrize("pipeline", ["memory_guided_noise", "guided_noise"])
def test_paired_with_metadata_routing_pipeline_passes(pipeline: str) -> None:
    cfg = _compose("memory=full", "memory/crediting=paired", f"pipeline={pipeline}")

    validate_crediting_pipeline_compat(cfg)


def test_point_passes_under_any_pipeline() -> None:
    validate_crediting_pipeline_compat(_compose("memory=full"))
    validate_crediting_pipeline_compat(_compose("memory=writer"))


def test_no_memory_group_passes() -> None:
    validate_crediting_pipeline_compat(_compose())


@pytest.mark.parametrize(
    ("group", "expected"),
    [("point", PointEffectEstimator), ("paired", PairedEffectEstimator)],
)
def test_ref_writeback_survives_runtime_resolution(group: str, expected: type) -> None:
    """run.py resolves ``${ref:memory.crediting}`` inside instantiate(); the
    ref write-back must store the estimator opaquely — dataclass estimators
    get structured-wrapped by OmegaConf, crashing (paired) or replacing the
    object with a DictConfig (point)."""
    register_resolvers()
    full = _compose("memory=full", f"memory/crediting={group}")
    tree = OmegaConf.create(
        {
            "memory": {
                "crediting": OmegaConf.to_container(
                    full.memory.crediting, resolve=False
                ),
                "writer": {"effect_estimator": "${ref:memory.crediting}"},
            }
        }
    )

    instantiated = instantiate(tree, _recursive_=True)

    assert isinstance(instantiated.memory.writer.effect_estimator, expected)
