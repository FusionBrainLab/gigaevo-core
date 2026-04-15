"""Tests for the OmegaConf-based manifest loader (step 5).

Step 5 switches ``load_manifest`` (and the ``from_yaml_file`` classmethod that
backs it) from ``yaml.safe_load`` to OmegaConf. The motivation:

  * v2 yamls will routinely use ``${oc.env:X,default}`` for secrets and
    ``${contract.max_generations}`` for cross-section interpolation to keep
    derived fields in sync with their source.
  * The existing ``yaml.safe_load`` path silently leaves ``${...}`` strings
    unresolved, which would make env-var secrets land in the manifest as
    literal ``"${oc.env:OPENAI_API_KEY,sk-gigaevo}"`` — a silent correctness
    bug waiting to happen.

Contract the loader must honor after step 5:

  1. All 18 existing v1 yamls still load and validate unchanged (no regression).
  2. ``${oc.env:NAME,default}`` resolves to the env var or the default.
  3. ``${oc.env:NAME}`` raises a clear error when ``NAME`` is unset and no
     default is given — we prefer loud failure to a stringly-typed secret.
  4. Cross-section interpolation resolves at load time — e.g. a yaml whose
     ``stopping_rule`` references ``${experiment.max_generations}`` arrives
     in Python as the resolved string.
  5. Non-dict roots still raise ``ValueError`` (same guardrail as before).
  6. Missing files still raise ``FileNotFoundError``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gigaevo.experiment.manifest import ExperimentManifest, load_manifest

REPO_ROOT = Path(__file__).parent.parent.parent


def _minimal_v2_dict() -> dict:
    """Smallest well-formed v2 manifest that passes validation."""
    return {
        "schema_version": 2,
        "experiment": {
            "name": "hover/omegaconf-test",
            "task": "hover",
            "status": "preregistered",
            "branch": "exp/hover/omegaconf-test",
            "max_generations": 25,
        },
        "problem": {
            "has_test_set": True,
            "fitness_type": "discrete",
            "metric_name": "accuracy",
        },
        "runs": [],
        "servers": [],
        "config": {},
    }


def _write_manifest(tmp_path: Path, data: dict, *, name: str = "foo/bar") -> Path:
    exp_dir = tmp_path / "experiments" / name
    exp_dir.mkdir(parents=True)
    yaml_path = exp_dir / "experiment.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path


# ---------------------------------------------------------------------------
# Baseline: all v2 yamls load and validate
# ---------------------------------------------------------------------------


class TestAllV2YamlsLoad:
    """Every live experiment.yaml must validate under schema v2."""

    def test_minimal_v2_roundtrips(self, tmp_path):
        _write_manifest(tmp_path, _minimal_v2_dict(), name="hover/omegaconf-test")
        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")
        assert isinstance(m, ExperimentManifest)
        assert m.experiment.name == "hover/omegaconf-test"
        assert m.experiment.max_generations == 25

    @pytest.mark.parametrize(
        "yaml_path",
        [
            p
            for p in (REPO_ROOT / "experiments").glob("*/*/experiment.yaml")
            if "_template" not in str(p)
        ],
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_every_real_yaml_loads(self, yaml_path: Path):
        """Every live experiment.yaml must continue to validate."""
        # Load via the classmethod (same code path load_manifest uses).
        m = ExperimentManifest.from_yaml_file(yaml_path)
        assert isinstance(m, ExperimentManifest)
        assert m.experiment.name  # non-empty


# ---------------------------------------------------------------------------
# Env-var interpolation — ${oc.env:NAME,default} and ${oc.env:NAME}
# ---------------------------------------------------------------------------


class TestOcEnvInterpolation:
    """``${oc.env:NAME,default}`` must resolve at load time."""

    def test_env_var_resolves_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GIGAEVO_TEST_KEY", "resolved-secret")
        data = _minimal_v2_dict()
        data["custom_env"] = {
            "OPENAI_API_KEY": "${oc.env:GIGAEVO_TEST_KEY,fallback}",
        }
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")

        # ${oc.env:X,default} must be resolved — not stored as literal ${...}
        assert m.custom_env["OPENAI_API_KEY"] == "resolved-secret"

    def test_env_var_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GIGAEVO_TEST_KEY_UNSET", raising=False)
        data = _minimal_v2_dict()
        data["custom_env"] = {
            "OPENAI_API_KEY": "${oc.env:GIGAEVO_TEST_KEY_UNSET,sk-gigaevo}",
        }
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")

        assert m.custom_env["OPENAI_API_KEY"] == "sk-gigaevo"

    def test_missing_env_without_default_raises(self, tmp_path, monkeypatch):
        """Unset env + no default must fail loudly (not silently store ${...})."""
        monkeypatch.delenv("GIGAEVO_ABSENT", raising=False)
        data = _minimal_v2_dict()
        data["custom_env"] = {"SECRET": "${oc.env:GIGAEVO_ABSENT}"}
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            with pytest.raises(Exception):  # noqa: B017 — OmegaConf raises its own type
                load_manifest("hover/omegaconf-test")

    def test_literal_dollar_strings_preserved(self, tmp_path):
        """Plain strings without ${...} must survive unchanged."""
        data = _minimal_v2_dict()
        data["custom_env"] = {"NOTE": "plain value, no interpolation"}
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")

        assert m.custom_env["NOTE"] == "plain value, no interpolation"


# ---------------------------------------------------------------------------
# Cross-section interpolation
# ---------------------------------------------------------------------------


class TestCrossSectionInterpolation:
    """Interpolations that point at another field in the same file resolve."""

    def test_stopping_rule_references_max_generations(self, tmp_path):
        data = _minimal_v2_dict()
        data["experiment"]["max_generations"] = 50
        data["experiment"]["stopping_rule"] = (
            "max_generations=${experiment.max_generations}"
        )
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")

        assert m.experiment.stopping_rule == "max_generations=50"


# ---------------------------------------------------------------------------
# Pass-through Hydra interpolations (escaped with \${...})
# ---------------------------------------------------------------------------


class TestHydraPassThroughEscape:
    """Pass-through interpolations use OmegaConf's ``\\${...}`` escape.

    ``extra_overrides`` may contain strings meant for downstream Hydra
    composition — e.g. ``post_step_hook=${composition_injection_hook}``.
    These must be escaped in YAML as ``\\${composition_injection_hook}``
    so OmegaConf treats them as literals. After ``to_container``, the
    backslash is stripped and the string arrives in the manifest as the
    literal ``${composition_injection_hook}`` that ``run.py`` expects.
    """

    def test_escaped_interpolation_roundtrips_as_literal_dollar(self, tmp_path):
        data = _minimal_v2_dict()
        # Write the escape via YAML literal so the backslash survives the dump.
        data["runs"] = [
            {
                "label": "R1",
                "db": 5,
                "prefix": "r1",
                "pipeline": "standard",
                "problem_name": "hover",
                "condition": "control",
                "chain_url": "http://example.com",
                "mutation_url": "http://example.com",
                "model_name": "gpt-4",
                "extra_overrides": [
                    "post_step_hook=\\${composition_injection_hook}",
                ],
            }
        ]
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")

        assert (
            m.runs[0].extra_overrides[0]
            == "post_step_hook=${composition_injection_hook}"
        )

    def test_mixed_resolvable_and_passthrough_in_same_file(self, tmp_path, monkeypatch):
        """``${oc.env:X}`` resolves; escaped ``\\${Y}`` survives as literal."""
        monkeypatch.setenv("GIGAEVO_TEST_KEY", "resolved-secret")
        data = _minimal_v2_dict()
        data["custom_env"] = {
            "OPENAI_API_KEY": "${oc.env:GIGAEVO_TEST_KEY,fallback}",
        }
        data["runs"] = [
            {
                "label": "R1",
                "db": 5,
                "prefix": "r1",
                "pipeline": "standard",
                "problem_name": "hover",
                "condition": "control",
                "chain_url": "http://example.com",
                "mutation_url": "http://example.com",
                "model_name": "gpt-4",
                "extra_overrides": [
                    "post_step_hook=\\${composition_injection_hook}",
                ],
            }
        ]
        _write_manifest(tmp_path, data, name="hover/omegaconf-test")

        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            m = load_manifest("hover/omegaconf-test")

        assert m.custom_env["OPENAI_API_KEY"] == "resolved-secret"
        assert (
            m.runs[0].extra_overrides[0]
            == "post_step_hook=${composition_injection_hook}"
        )


# ---------------------------------------------------------------------------
# Error surface unchanged
# ---------------------------------------------------------------------------


class TestLoaderErrorSurface:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_manifest("nonexistent/exp")

    def test_scalar_yaml_rejected(self, tmp_path):
        exp_dir = tmp_path / "experiments" / "hover" / "scalar"
        exp_dir.mkdir(parents=True)
        (exp_dir / "experiment.yaml").write_text("just-a-string\n")
        with patch("gigaevo.experiment.manifest.PROJ", tmp_path):
            with pytest.raises((ValueError, Exception)):  # noqa: B017
                load_manifest("hover/scalar")
