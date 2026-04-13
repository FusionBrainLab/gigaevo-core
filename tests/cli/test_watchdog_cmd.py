"""Tests for the watchdog CLI subcommand."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gigaevo.cli import main


def _make_fake_manifest():
    """Build a fake manifest object for testing."""
    from gigaevo.monitoring.manifest_schema import WatchdogSection

    mock = MagicMock()
    mock.name = "test/exp"
    mock.task = "hover"
    mock.max_generations = 50
    mock.watchdog_plugin = None
    mock.servers = ["10.0.0.1"]
    mock.watchdog = WatchdogSection()

    run1 = MagicMock()
    run1.prefix = "test/prefix"
    run1.db = 4
    run1.label = "A"
    run1.pid = 12345

    mock.runs = [run1]
    mock.experiment = MagicMock()
    mock.experiment.task = "hover"

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
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_plugin_cls = MagicMock()
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
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_resolve.return_value = MagicMock()
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

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
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
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
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
        """NO_PROXY env var is auto-configured from manifest.servers."""
        import os

        manifest = _make_fake_manifest()
        manifest.servers = ["10.0.0.1", "10.0.0.2"]

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            old_no_proxy = os.environ.get("NO_PROXY", "")
            try:
                os.environ["NO_PROXY"] = ""
                runner = CliRunner()
                result = runner.invoke(
                    main,
                    ["-e", "test/exp", "watchdog"],
                    catch_exceptions=False,
                )
                assert result.exit_code == 0, result.output
                no_proxy = os.environ.get("NO_PROXY", "")
                assert "10.0.0.1" in no_proxy
                assert "10.0.0.2" in no_proxy
                assert "api.github.com" in no_proxy
            finally:
                os.environ["NO_PROXY"] = old_no_proxy

    def test_no_proxy_includes_extra_hosts(self):
        """no_proxy_hosts from watchdog section are included."""
        import os

        from gigaevo.monitoring.manifest_schema import WatchdogSection

        manifest = _make_fake_manifest()
        manifest.servers = ["10.0.0.1"]
        manifest.watchdog = WatchdogSection(no_proxy_hosts=["custom.host.com"])

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            old_no_proxy = os.environ.get("NO_PROXY", "")
            try:
                os.environ["NO_PROXY"] = ""
                runner = CliRunner()
                result = runner.invoke(
                    main,
                    ["-e", "test/exp", "watchdog"],
                    catch_exceptions=False,
                )
                assert result.exit_code == 0, result.output
                no_proxy = os.environ.get("NO_PROXY", "")
                assert "custom.host.com" in no_proxy
            finally:
                os.environ["NO_PROXY"] = old_no_proxy


class TestWatchdogManifestConfig:
    def test_config_from_manifest_watchdog_section(self):
        """WatchdogConfig is built from manifest.watchdog section."""
        from gigaevo.monitoring.manifest_schema import WatchdogSection

        manifest = _make_fake_manifest()
        manifest.watchdog = WatchdogSection(
            poll_interval_s=1800,
            plot_retries=5,
            plot_retry_delay_s=60,
            checkpoint_milestones=[0.25, 0.5, 1.0],
        )

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            config = call_kwargs["config"]
            assert config.poll_interval_s == 1800
            assert config.plot_retries == 5
            assert config.checkpoint_milestones == (0.25, 0.5, 1.0)

    def test_cli_flag_overrides_manifest(self):
        """CLI --poll-interval takes precedence over manifest.watchdog.poll_interval_s."""
        from gigaevo.monitoring.manifest_schema import WatchdogSection

        manifest = _make_fake_manifest()
        manifest.watchdog = WatchdogSection(poll_interval_s=1800)

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog", "--poll-interval", "900"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["config"].poll_interval_s == 900


class TestWatchdogBaselineFromManifest:
    def test_baseline_passed_to_engine(self):
        """baseline from manifest.baseline.mean is passed to WatchdogEngine."""
        manifest = _make_fake_manifest()
        manifest.baseline = MagicMock()
        manifest.baseline.mean = 0.034

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
            patch("gigaevo.cli.watchdog_cmd._get_github_token", return_value=None),
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["baseline"] == 0.034

    def test_baseline_none_when_not_set(self):
        """baseline is None when manifest.baseline.mean is None."""
        manifest = _make_fake_manifest()
        manifest.baseline = MagicMock()
        manifest.baseline.mean = None

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
            patch("gigaevo.cli.watchdog_cmd._get_github_token", return_value=None),
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["baseline"] is None


class TestWatchdogDispatcher:
    def test_dispatcher_passed_to_engine(self):
        """A NotificationDispatcher is passed to WatchdogEngine."""
        manifest = _make_fake_manifest()
        manifest.baseline = MagicMock()
        manifest.baseline.mean = None

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
            patch("gigaevo.cli.watchdog_cmd._get_github_token", return_value=None),
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            call_kwargs = mock_engine_cls.call_args[1]
            assert call_kwargs["dispatcher"] is not None

    def test_github_channel_created_with_token_and_pr(self):
        """GitHubPRChannel is created when token and pr_number are available."""
        manifest = _make_fake_manifest()
        manifest.experiment.pr_number = 99
        manifest.experiment.branch = "exp/test"
        manifest.baseline = MagicMock()
        manifest.baseline.mean = None

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
            patch(
                "gigaevo.cli.watchdog_cmd._get_github_token",
                return_value="ghp_test123",
            ),
            patch(
                "gigaevo.monitoring.github_pr_channel.GitHubPRChannel"
            ) as mock_gh_cls,
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None
            mock_gh_cls.return_value = MagicMock()

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_gh_cls.assert_called_once()
            gh_kwargs = mock_gh_cls.call_args[1]
            assert gh_kwargs["experiment_name"] == "test/exp"
            assert gh_kwargs["pr_number"] == 99
            assert gh_kwargs["branch"] == "exp/test"
            assert gh_kwargs["rolling_comment_threshold_hours"] == 24


class TestWatchdogMetricNamesPropagation:
    def test_run_configs_contain_metric_names_from_metrics_yaml(self):
        """RunConfigs built by watchdog contain metric_names loaded from metrics.yaml."""
        manifest = _make_fake_manifest()
        # Add problem_name to the mock run so _load_metric_names can use it
        manifest.runs[0].problem_name = "chains/hover/test"

        expected_metrics = ["fitness", "actual_fitness", "quality"]

        with (
            patch("gigaevo.monitoring.manifest.load_manifest", return_value=manifest),
            patch("gigaevo.monitoring.watchdog_plugin.resolve_plugin") as mock_resolve,
            patch(
                "gigaevo.cli.run_resolver._load_metric_names",
                return_value=expected_metrics,
            ) as mock_load_metrics,
            patch(
                "gigaevo.monitoring.watchdog_engine.WatchdogEngine"
            ) as mock_engine_cls,
        ):
            mock_resolve.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "watchdog"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output

            mock_load_metrics.assert_called_once_with("chains/hover/test")

            call_kwargs = mock_engine_cls.call_args[1]
            run_configs = call_kwargs["run_configs"]
            assert len(run_configs) == 1
            assert run_configs[0].metric_names == expected_metrics
