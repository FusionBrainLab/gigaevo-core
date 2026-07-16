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

    def test_duplicate_labels_are_rejected(self):
        with pytest.raises(click.UsageError, match="labels must be unique.*same"):
            RunResolver.resolve(
                experiment=None,
                runs=["p@1:same", "q@2:same"],
                redis_host="localhost",
                redis_port=6379,
            )


class TestResolveDiskRuns:
    def _make_storage_dir(self, tmp_path, prefix: str = "toy"):
        base = tmp_path / "storage" / prefix
        (base / "programs").mkdir(parents=True)
        return tmp_path / "storage"

    def test_root_with_single_prefix_autodiscovers(self, tmp_path):
        root = self._make_storage_dir(tmp_path, "toy")
        configs = RunResolver.resolve(
            experiment=None,
            runs=[str(root)],
            redis_host="localhost",
            redis_port=6379,
        )
        spec = configs[0].run_spec
        assert spec.is_disk
        assert spec.prefix == "toy"
        assert spec.path == str(root)
        assert spec.label == "toy"

    def test_direct_prefix_dir(self, tmp_path):
        root = self._make_storage_dir(tmp_path, "toy")
        configs = RunResolver.resolve(
            experiment=None,
            runs=[str(root / "toy")],
            redis_host="localhost",
            redis_port=6379,
        )
        spec = configs[0].run_spec
        assert spec.is_disk
        assert spec.prefix == "toy"
        assert spec.path == str(root)

    def test_explicit_label_preserved(self, tmp_path):
        root = self._make_storage_dir(tmp_path, "toy")
        configs = RunResolver.resolve(
            experiment=None,
            runs=[f"{root}:mylabel"],
            redis_host="localhost",
            redis_port=6379,
        )
        assert configs[0].run_spec.label == "mylabel"

    def test_relative_path_without_dot_prefix(self, tmp_path, monkeypatch):
        self._make_storage_dir(tmp_path, "toy")
        monkeypatch.chdir(tmp_path)
        configs = RunResolver.resolve(
            experiment=None,
            runs=["storage" + "/toy"],
            redis_host="localhost",
            redis_port=6379,
        )
        assert configs[0].run_spec.is_disk
        assert configs[0].run_spec.prefix == "toy"

    def test_duplicate_disk_auto_labels_are_rejected(self, tmp_path):
        roots = []
        for name in ("run-a", "run-b"):
            root = tmp_path / name / "storage"
            (root / "toy" / "programs").mkdir(parents=True)
            roots.append(root)
        with pytest.raises(click.UsageError, match="labels must be unique.*toy"):
            RunResolver.resolve(
                experiment=None,
                runs=[str(root) for root in roots],
                redis_host="localhost",
                redis_port=6379,
            )

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(click.UsageError, match="not a directory"):
            RunResolver.resolve(
                experiment=None,
                runs=[str(tmp_path / "nope")],
                redis_host="localhost",
                redis_port=6379,
            )

    def test_no_storage_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(click.UsageError, match="No program storage"):
            RunResolver.resolve(
                experiment=None,
                runs=[str(empty)],
                redis_host="localhost",
                redis_port=6379,
            )

    def test_multiple_prefixes_raises(self, tmp_path):
        self._make_storage_dir(tmp_path, "alpha")
        root = self._make_storage_dir(tmp_path, "beta")
        with pytest.raises(click.UsageError, match="alpha.*beta"):
            RunResolver.resolve(
                experiment=None,
                runs=[str(root)],
                redis_host="localhost",
                redis_port=6379,
            )


class TestRejectDiskSpecs:
    def test_rejects_disk_specs_with_command_name(self, tmp_path):
        from gigaevo.cli.run_resolver import reject_disk_specs
        from gigaevo.monitoring.experiment_monitor import RunConfig

        spec = RunSpec(prefix="toy", db=-1, label="toy", path=str(tmp_path))
        with pytest.raises(click.UsageError, match="status.*Redis"):
            reject_disk_specs([RunConfig(run_spec=spec)], "status")

    def test_passes_redis_specs(self):
        from gigaevo.cli.run_resolver import reject_disk_specs
        from gigaevo.monitoring.experiment_monitor import RunConfig

        spec = RunSpec(prefix="p", db=4, label="A")
        reject_disk_specs([RunConfig(run_spec=spec)], "status")


class TestVerifyPrefixesSkipsDisk:
    def test_verify_prefixes_skips_disk_specs(self, tmp_path):
        """export's Redis existence probe must ignore disk-backed specs."""
        from unittest.mock import MagicMock

        from gigaevo.cli.export import _verify_prefixes_exist
        from gigaevo.monitoring.experiment_monitor import RunConfig

        ctx = MagicMock()
        ctx.obj = {}
        spec = RunSpec(prefix="toy", db=-1, label="toy", path=str(tmp_path))
        _verify_prefixes_exist(ctx, [RunConfig(run_spec=spec)], "localhost", 6379)


class TestVerifyRedisPrefixes:
    def test_probe_uses_scan_iter_and_accepts_program_data(self):
        from gigaevo.cli.export import _verify_prefixes_exist
        from gigaevo.monitoring.experiment_monitor import RunConfig

        ctx = MagicMock()
        ctx.obj = {}
        client = MagicMock()
        client.scan_iter.return_value = iter(["p:program:abc"])
        spec = RunSpec(prefix="p", db=4, label="p")
        with (
            patch(
                "gigaevo.cli.inspect_cmd.discover_prefixes",
                return_value=[],
            ),
            patch("redis.Redis", return_value=client),
        ):
            _verify_prefixes_exist(
                ctx,
                [RunConfig(run_spec=spec)],
                "localhost",
                6379,
            )
        client.scan_iter.assert_called_once_with(match="p:*", count=1000)
        client.scan.assert_not_called()
