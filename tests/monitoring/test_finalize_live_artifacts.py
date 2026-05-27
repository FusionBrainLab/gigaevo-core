"""Tests for run._finalize_live_artifacts — guarantees plot + dashboard
are rendered once at exit, even when the engine crashed mid-run."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

# run.py lives at repo root, not under a package — make it importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run  # noqa: E402


@pytest.fixture
def tmp_log_and_out(tmp_path: Path) -> tuple[str, Path]:
    # setup_logger() returns str via os.path.join — pass a str here so the
    # finalize helper is exercised the way production calls it. A regression
    # to "forgot to wrap in Path()" surfaces immediately.
    log = tmp_path / "run.log"
    log.write_text("")
    out = tmp_path / "out"
    out.mkdir()
    return str(log), out


class TestProfilerRender:
    def test_render_called_with_path(self, tmp_log_and_out):
        log, out = tmp_log_and_out
        with patch.object(run, "_render_once", return_value=(3, 7)) as p:
            run._finalize_live_artifacts(
                log_file_path=log, output_dir=out, last_n=None, frontier_ctx=None
            )
        # _render_once expects a Path (it uses .exists()/.open()), even when
        # setup_logger handed back a str
        p.assert_called_once_with(
            Path(log), out / "profile_live.html", "final", last_n=None
        )

    def test_render_exception_swallowed(self, tmp_log_and_out):
        log, out = tmp_log_and_out
        with patch.object(run, "_render_once", side_effect=RuntimeError("boom")):
            # must not propagate
            run._finalize_live_artifacts(
                log_file_path=log, output_dir=out, last_n=None, frontier_ctx=None
            )


class TestFrontierRender:
    def test_frontier_ctx_none_skips_redis(self, tmp_log_and_out):
        log, out = tmp_log_and_out
        with (
            patch.object(run, "_render_once", return_value=(0, 0)),
            patch.object(run, "_fetch_histories") as fh,
            patch.object(run, "_render_frontier_plot") as fp,
        ):
            run._finalize_live_artifacts(
                log_file_path=log, output_dir=out, last_n=None, frontier_ctx=None
            )
            fh.assert_not_called()
            fp.assert_not_called()

    def test_frontier_renders_each_metric(self, tmp_log_and_out):
        log, out = tmp_log_and_out
        ctx = {
            "redis_url": "redis://localhost:6379/0",
            "key_prefix": "p:metrics",
            "metrics": ["fitness", "size"],
            "higher_is_better": {"fitness": True, "size": False},
        }
        fake_frontier = {"fitness": [(1, 0.9)], "size": [(1, 5.0)]}
        fake_iter = {"fitness": [(1, 0.5)], "size": [(1, 7.0)]}

        with (
            patch.object(run, "_render_once", return_value=(0, 0)),
            patch.object(
                run, "_fetch_histories", return_value=(fake_frontier, fake_iter, {})
            ),
            patch.object(
                run, "_render_frontier_plot", return_value=out / "frontier.png"
            ) as fp,
            patch("redis.Redis.from_url", return_value=MagicMock()),
        ):
            run._finalize_live_artifacts(
                log_file_path=log, output_dir=out, last_n=None, frontier_ctx=ctx
            )
            # one call per metric
            assert fp.call_count == 2
            kwargs = [c.kwargs for c in fp.call_args_list]
            assert {k["metric"] for k in kwargs} == {"fitness", "size"}
            # higher_is_better forwarded per-metric
            by_metric = {k["metric"]: k["higher_is_better"] for k in kwargs}
            assert by_metric == {"fitness": True, "size": False}

    def test_redis_unreachable_skips_plot_without_raising(self, tmp_log_and_out):
        log, out = tmp_log_and_out
        ctx = {
            "redis_url": "redis://localhost:6379/0",
            "key_prefix": "p:metrics",
            "metrics": ["fitness"],
            "higher_is_better": {"fitness": True},
        }
        with (
            patch.object(run, "_render_once", return_value=(0, 0)),
            patch.object(run, "_fetch_histories", side_effect=ConnectionError("nope")),
            patch.object(run, "_render_frontier_plot") as fp,
        ):
            run._finalize_live_artifacts(
                log_file_path=log, output_dir=out, last_n=None, frontier_ctx=ctx
            )
            fp.assert_not_called()

    def test_per_metric_failure_is_isolated(self, tmp_log_and_out):
        log, out = tmp_log_and_out
        ctx = {
            "redis_url": "redis://localhost:6379/0",
            "key_prefix": "p:metrics",
            "metrics": ["fitness", "size"],
            "higher_is_better": {"fitness": True, "size": True},
        }

        def side_effect(*, metric, **kw):
            if metric == "fitness":
                raise RuntimeError("plot blew up")
            return out / f"frontier_{metric}.png"

        with (
            patch.object(run, "_render_once", return_value=(0, 0)),
            patch.object(run, "_fetch_histories", return_value=({}, {}, {})),
            patch.object(run, "_render_frontier_plot", side_effect=side_effect) as fp,
            patch("redis.Redis.from_url", return_value=MagicMock()),
        ):
            run._finalize_live_artifacts(
                log_file_path=log, output_dir=out, last_n=None, frontier_ctx=ctx
            )
            # both metrics attempted despite the first failing
            assert fp.call_count == 2
