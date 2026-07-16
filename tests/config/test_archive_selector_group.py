"""Archive-selector Hydra group: point default, paired_bootstrap opt-in.

Every single-selector algorithm must route through the ``archive_selector``
group so ``archive_selector=paired_bootstrap`` is a real treatment switch —
a config group that silently no-ops under some algorithms is the inert-
treatment trap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest

from gigaevo.config.validation import validate_paired_selector_pipeline_compat

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_BASE = ["problem.name=_test_"]

POINT = "gigaevo.evolution.strategies.map_elites.SumArchiveSelector"
PAIRED = "gigaevo.evolution.strategies.paired_selectors.PairedBootstrapArchiveSelector"

SINGLE_SELECTOR_ALGORITHMS = [
    "single_island",
    "single_island_no_distant_parents",
    "single_island_2d",
    "topology_3d",
    "topology_3d_ret",
    "chains_bd3d",
    "tabular/2d_local_ood",
]


def _compose(*overrides: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=[*_BASE, *overrides])


def test_default_is_point_comparison() -> None:
    cfg = _compose()

    assert cfg.archive_selector._target_ == POINT
    assert cfg.islands[0].archive_selector._target_ == POINT


def test_paired_bootstrap_override_reaches_island() -> None:
    cfg = _compose("archive_selector=paired_bootstrap")

    assert cfg.islands[0].archive_selector._target_ == PAIRED
    assert cfg.islands[0].archive_selector.p_accept == 0.75


def test_p_accept_is_overridable() -> None:
    cfg = _compose(
        "archive_selector=paired_bootstrap", "archive_selector.p_accept=0.65"
    )

    assert cfg.islands[0].archive_selector.p_accept == 0.65


@pytest.mark.parametrize("algorithm", SINGLE_SELECTOR_ALGORITHMS)
def test_group_binds_under_every_single_selector_algorithm(algorithm: str) -> None:
    cfg = _compose(f"algorithm={algorithm}", "archive_selector=paired_bootstrap")

    for island in cfg.islands:
        assert island.archive_selector._target_ == PAIRED


def test_multi_island_fitness_island_binds_simplicity_stays_inline() -> None:
    cfg = _compose("algorithm=multi_island", "archive_selector=paired_bootstrap")

    assert cfg.islands[0].archive_selector._target_ == PAIRED
    # Simplicity island is multi-key; PairedBootstrapArchiveSelector is
    # single-key by contract, so it keeps its inline point selector.
    assert cfg.islands[1].archive_selector._target_ == POINT
    assert len(cfg.islands[1].archive_selector.fitness_keys) == 2


def test_paired_selector_without_metadata_pipeline_is_rejected() -> None:
    cfg = _compose("archive_selector=paired_bootstrap", "pipeline=guided")

    with pytest.raises(ValueError, match="routes_program_metadata"):
        validate_paired_selector_pipeline_compat(cfg)


def test_paired_selector_with_noise_pipeline_passes() -> None:
    cfg = _compose("archive_selector=paired_bootstrap", "pipeline=memory_guided_noise")

    validate_paired_selector_pipeline_compat(cfg)


def test_paired_selector_with_guided_noise_pipeline_passes() -> None:
    cfg = _compose("archive_selector=paired_bootstrap", "pipeline=guided_noise")

    validate_paired_selector_pipeline_compat(cfg)


def test_point_selector_passes_under_any_pipeline() -> None:
    validate_paired_selector_pipeline_compat(_compose())
    validate_paired_selector_pipeline_compat(_compose("pipeline=memory_guided_noise"))


def test_inline_paired_island_is_caught() -> None:
    cfg = OmegaConf.create(
        {
            "islands": [{"archive_selector": {"_target_": PAIRED}}],
            "pipeline": {"id": "guided"},
        }
    )

    with pytest.raises(ValueError, match="routes_program_metadata"):
        validate_paired_selector_pipeline_compat(cfg)
