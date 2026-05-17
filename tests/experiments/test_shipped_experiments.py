"""Parametrised tests for every shipped experiment file.

Each experiment under ``experiments/`` must load, validate, and
produce a fully-constructed :class:`ExperimentConfig`. The tests
walk the directory at collection time so a new experiment file
added to the tree is automatically covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.config.experiment_loader import build_experiment
from gigaevo.config.schemas import ExperimentConfig

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
EXPERIMENT_FILES = sorted(
    p
    for p in EXPERIMENTS_DIR.glob("*.py")
    if not p.name.startswith("_")
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.parametrize(
    "experiment_path",
    EXPERIMENT_FILES,
    ids=[p.stem for p in EXPERIMENT_FILES],
)
def test_experiment_builds(experiment_path: Path) -> None:
    cfg = build_experiment(experiment_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name


@pytest.mark.parametrize(
    "experiment_path",
    EXPERIMENT_FILES,
    ids=[p.stem for p in EXPERIMENT_FILES],
)
def test_experiment_key_prefix_follows_convention(
    experiment_path: Path,
) -> None:
    cfg = build_experiment(experiment_path)
    assert cfg.dataplane.key_prefix == f"gigaevo:{cfg.name}"


@pytest.mark.parametrize(
    "experiment_path",
    EXPERIMENT_FILES,
    ids=[p.stem for p in EXPERIMENT_FILES],
)
def test_experiment_id_is_stable(experiment_path: Path) -> None:
    first = build_experiment(experiment_path).experiment_id
    second = build_experiment(experiment_path).experiment_id
    assert first == second


@pytest.mark.parametrize(
    "experiment_path",
    EXPERIMENT_FILES,
    ids=[p.stem for p in EXPERIMENT_FILES],
)
def test_experiment_serializes_to_json(experiment_path: Path) -> None:
    cfg = build_experiment(experiment_path)
    payload = cfg.model_dump_json()
    parsed = ExperimentConfig.model_validate_json(payload)
    assert parsed.name == cfg.name


def test_prompt_coevolution_carries_fetcher_through_to_evolution_context() -> None:
    """prompt_coevolution.py constructs a GigaEvoArchivePromptFetcherConfig
    and threads it through the ExperimentConfig root. The
    build_object_graph adapter consumes that field; without it the
    EvolutionContext would be wired with prompt_fetcher=None and
    the coevolved pattern would silently degrade to fixed-dir
    fetching."""
    from gigaevo.config.schemas import GigaEvoArchivePromptFetcherConfig

    cfg = build_experiment(EXPERIMENTS_DIR / "prompt_coevolution.py")
    assert isinstance(cfg.prompt_fetcher, GigaEvoArchivePromptFetcherConfig)
    assert cfg.prompt_fetcher.prompt_redis_db == 6


def test_other_experiments_have_no_prompt_fetcher() -> None:
    """The seven non-coevolved experiments must leave prompt_fetcher
    as None — the YAML's /prompt_fetcher: fixed default falls back to
    the package prompts directory inside the runtime fetcher, no
    schema-level wiring required."""
    for stem in (
        "base",
        "steady_state",
        "full_featured",
        "multi_island_complexity",
        "multi_llm_exploration",
        "migration_bus",
        "steady_state_bus",
    ):
        cfg = build_experiment(EXPERIMENTS_DIR / f"{stem}.py")
        assert cfg.prompt_fetcher is None, (
            f"experiment {stem} unexpectedly carries a prompt_fetcher"
        )


