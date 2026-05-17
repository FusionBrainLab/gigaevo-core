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


def test_eight_experiment_yamls_have_python_counterparts() -> None:
    """The plan §5.14 requires one experiments/X.py for every
    config/experiment/X.yaml. Eight YAMLs shipped today + the
    reference experiment from hydra-1.11 + the new task migrations."""
    yaml_dir = EXPERIMENTS_DIR.parent / "config" / "experiment"
    yaml_stems = {p.stem for p in yaml_dir.glob("*.yaml")}
    python_stems = {p.stem for p in EXPERIMENT_FILES}
    missing = yaml_stems - python_stems
    assert not missing, (
        f"YAMLs without Python counterparts: {sorted(missing)}"
    )
