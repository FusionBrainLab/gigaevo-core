"""Tests for RunResolver: bridges CLI flags to monitoring RunConfig."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest

from gigaevo.cli.run_resolver import RunResolver
from gigaevo.monitoring.run_spec import RunSpec


class TestResolveFromRunFlags:
    def test_single_run(self):
        configs = RunResolver.resolve(
            experiment=None,
            runs=["prefix@4:O"],
            redis_host="localhost",
            redis_port=6379,
        )
        assert len(configs) == 1
        assert configs[0].run_spec == RunSpec(prefix="prefix", db=4, label="O")
        assert configs[0].metric_names == ["fitness"]

    def test_multiple_runs(self):
        configs = RunResolver.resolve(
            experiment=None,
            runs=["p@1:A", "p@2:B"],
            redis_host="localhost",
            redis_port=6379,
        )
        assert len(configs) == 2
        assert configs[0].run_spec.label == "A"
        assert configs[1].run_spec.label == "B"


class TestResolveFromExperiment:
    def test_from_experiment_flag(self, tmp_path):
        mock_manifest = MagicMock()
        mock_run_a = MagicMock()
        mock_run_a.prefix = "chains/hover/static"
        mock_run_a.db = 4
        mock_run_a.label = "A"
        mock_run_a.problem_name = "chains/hover/static"
        mock_run_a.pid = 12345

        mock_run_b = MagicMock()
        mock_run_b.prefix = "chains/hover/static"
        mock_run_b.db = 5
        mock_run_b.label = "B"
        mock_run_b.problem_name = "chains/hover/static"
        mock_run_b.pid = 12346

        mock_manifest.contract.runs = [mock_run_a, mock_run_b]

        with (
            patch(
                "gigaevo.cli.run_resolver._load_manifest", return_value=mock_manifest
            ),
            patch(
                "gigaevo.cli.run_resolver._load_metric_names",
                return_value=["fitness", "prompt_length"],
            ),
        ):
            configs = RunResolver.resolve(
                experiment="hover/test",
                runs=[],
                redis_host="localhost",
                redis_port=6379,
            )

        assert len(configs) == 2
        assert configs[0].run_spec == RunSpec(
            prefix="chains/hover/static", db=4, label="A"
        )
        assert configs[0].metric_names == ["fitness", "prompt_length"]
        assert configs[0].pid == 12345
        assert configs[1].run_spec.label == "B"
        assert configs[1].pid == 12346


class TestResolveErrors:
    def test_raises_if_neither(self):
        with pytest.raises(click.UsageError, match="Provide --experiment or"):
            RunResolver.resolve(
                experiment=None,
                runs=[],
                redis_host="localhost",
                redis_port=6379,
            )

    def test_raises_if_both(self):
        with pytest.raises(click.UsageError, match="not both"):
            RunResolver.resolve(
                experiment="hover/test",
                runs=["p@1:A"],
                redis_host="localhost",
                redis_port=6379,
            )
