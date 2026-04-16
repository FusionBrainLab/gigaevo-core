"""Tests for the watchdog CLI subcommand."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gigaevo.cli import main


def _make_fake_manifest():
    """Build a fake manifest object for testing."""
    mock = MagicMock()

    # contract section
    mock.contract.identity.name = "test/exp"
    mock.contract.servers = ["10.0.0.1", "10.0.0.2"]
    mock.contract.max_generations = 50

    run1 = MagicMock()
    run1.prefix = "test/prefix"
    run1.db = 4
    run1.label = "A"
    run1.pid = 12345
    mock.contract.runs = [run1]

    # control_plane section
    watchdog = MagicMock()
    watchdog.no_proxy_hosts = ["custom.host.com"]
    watchdog.poll_interval_s = 3600
    watchdog.plot_retries = 3
    watchdog.plot_retry_delay_s = 30
    watchdog.checkpoint_milestones = [0.1, 0.2, 0.5, 1.0]
    mock.control_plane.watchdog = watchdog

    return mock


class TestWatchdogRequiresExperiment:
    def test_no_experiment_shows_error(self):
        """Watchdog without --experiment shows usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["watchdog"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "experiment" in result.output.lower()


class TestWatchdogStartsEngine:
    def test_constructs_engine_with_correct_args(self):
        """Watchdog creates WatchdogEngine with experiment name and plugin."""
        manifest = _make_fake_manifest()

        with (
            patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_plugin_cls = MagicMock()
            mock_plugin_cls.__name__ = "MockPlugin"
            mock_resolve.return_value = mock_plugin_cls
            mock_engine = mock_engine_cls.return_value
            mock_engine.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_engine_cls.assert_called_once()
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["experiment_name"] == "test/exp"
            mock_engine.run.assert_called_once()


class TestWatchdogPollInterval:
    def test_custom_poll_interval(self):
        """--poll-interval sets config.poll_interval_s."""
        manifest = _make_fake_manifest()

        with (
            patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            plugin_mock = MagicMock()
            plugin_mock.__name__ = "MockPlugin"
            mock_resolve.return_value = plugin_mock
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog", "--poll-interval", "1800"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["config"].poll_interval_s == 1800


class TestWatchdogPluginOverride:
    def test_plugin_override_uses_registry(self):
        """--plugin forces a specific plugin from registry."""
        manifest = _make_fake_manifest()
        mock_plugin_cls = MagicMock()
        mock_plugin_cls.__name__ = "MockPlugin"

        with (
            patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest),
            patch(
                "gigaevo.monitoring.watchdog_plugin.get_registry",
                return_value={"solo": mock_plugin_cls},
            ),
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog", "--plugin", "solo"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["plugin"] == mock_plugin_cls()

    def test_invalid_plugin_shows_error(self):
        """--plugin with unknown name shows error."""
        manifest = _make_fake_manifest()

        with (
            patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest),
            patch(
                "gigaevo.monitoring.watchdog_plugin.get_registry",
                return_value={"solo": MagicMock()},
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog", "--plugin", "nonexistent"],
                catch_exceptions=False,
            )
            assert result.exit_code != 0
            assert "nonexistent" in result.output.lower()


class TestWatchdogNoProxy:
    def test_no_proxy_set_from_manifest_servers(self):
        """NO_PROXY includes manifest servers and api.github.com."""
        import os

        manifest = _make_fake_manifest()

        with (
            patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
            patch.dict(os.environ, {"NO_PROXY": ""}, clear=False),
        ):
            plugin_mock = MagicMock()
            plugin_mock.__name__ = "MockPlugin"
            mock_resolve.return_value = plugin_mock
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "NO_PROXY" in result.output
            assert "10.0.0.1" in result.output
            assert "10.0.0.2" in result.output
            assert "api.github.com" in result.output
            assert "custom.host.com" in result.output
