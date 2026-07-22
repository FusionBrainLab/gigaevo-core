from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
COMMON = ROOT / "problems/tabular/_common"


def _tabular_problem(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(COMMON))
    return importlib.import_module("tabular_problem")


def test_cv_std_is_reported_with_replicate_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tabular_problem(monkeypatch)
    monkeypatch.delenv(module.FITNESS_ENV, raising=False)

    artifact = module._evaluation_measurement_artifact(0.12, 4)

    assert artifact == {
        "_evaluation_measurements": {
            "fitness": {
                "sample_sd": 0.12,
                "n": 4,
                "method": "cross_validation",
            }
        }
    }


def test_single_fold_has_unknown_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _tabular_problem(monkeypatch)
    monkeypatch.delenv(module.FITNESS_ENV, raising=False)

    assert module._evaluation_measurement_artifact(0.0, 1) is None


def test_lcb_fitness_does_not_claim_mean_score_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _tabular_problem(monkeypatch)
    monkeypatch.setenv(module.FITNESS_ENV, "lcb")

    assert module._evaluation_measurement_artifact(0.12, 4) is None
