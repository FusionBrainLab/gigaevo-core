"""Tests for problems.toy_example.validate — output schema and error formatting."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_validate():
    """Load validate.py directly — problems/toy_example is not an importable package."""
    path = Path(__file__).resolve().parents[2] / "problems" / "toy_example" / "validate.py"
    spec = importlib.util.spec_from_file_location("toy_example_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_layout() -> tuple[np.ndarray, np.ndarray]:
    """Construct a trivially valid 9-circle packing inside the unit square."""
    centers = np.array(
        [
            [0.2 + (i % 3) * 0.3, 0.2 + (i // 3) * 0.3]
            for i in range(9)
        ],
        dtype=float,
    )
    radii = np.full(9, 0.1, dtype=float)
    return centers, radii


def test_validate_returns_python_floats_for_json_safety() -> None:
    """Metrics dict values must be plain Python floats so downstream JSON / Redis
    serialization paths do not encounter numpy scalars."""
    validate = _load_validate().validate
    centers, radii = _valid_layout()

    metrics = validate((centers, radii))

    assert set(metrics) == {"is_valid", "fitness"}
    assert type(metrics["is_valid"]) is float
    assert type(metrics["fitness"]) is float
    assert metrics["is_valid"] == 1.0
    assert metrics["fitness"] == pytest.approx(0.9)


def test_validate_rejects_wrong_shape() -> None:
    """A valid 8-circle packing must still be rejected — the task fixes n=9."""
    validate = _load_validate().validate
    # Compact 8-circle layout that satisfies packing constraints but the wrong size.
    centers = np.array(
        [
            [0.2 + (i % 3) * 0.3, 0.2 + (i // 3) * 0.3]
            for i in range(8)
        ],
        dtype=float,
    )
    radii = np.full(8, 0.1, dtype=float)

    with pytest.raises(ValueError, match=r"Shape is invalid"):
        validate((centers, radii))


def test_validate_rejects_nan_radii() -> None:
    """Non-finite radii must raise."""
    validate = _load_validate().validate
    centers, radii = _valid_layout()
    radii = radii.copy()
    radii[0] = np.nan

    with pytest.raises(ValueError, match=r"NaN or infinite"):
        validate((centers, radii))
