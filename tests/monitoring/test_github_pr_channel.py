"""Tests for gigaevo.monitoring.github_pr_channel — GitHub PR notification channel."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from gigaevo.monitoring.alerts import Alert, AlertSeverity, AlertType
from gigaevo.monitoring.github_pr_channel import GitHubPRChannel
from gigaevo.monitoring.notifications import PlotAttachment, StatusUpdate
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot

# ── Test infrastructure ─────────────────────────────────────────────────────


class RequestRecorder:
    """Records HTTP requests for assertion in tests.

    Supports two response modes:
    - Simple dict: pattern -> response (matches on "METHOD /path" substring)
    - Ordered list: list of (pattern, response) tuples, consumed in order for
      the same pattern (allows different responses for repeated requests)
    """

    def __init__(
        self,
        responses: dict[str, httpx.Response] | None = None,
        ordered_responses: list[tuple[str, httpx.Response]] | None = None,
    ):
        self.requests: list[httpx.Request] = []
        self._responses = responses or {}
        self._ordered = list(ordered_responses) if ordered_responses else []
        self._default_response = httpx.Response(200, json={"id": 1})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"

        # Try ordered responses first (consumed in order)
        for i, (pattern, resp) in enumerate(self._ordered):
            if pattern in key:
                self._ordered.pop(i)
                return resp

        # Fall back to simple dict
        for pattern, resp in self._responses.items():
            if pattern in key:
                return resp
        return self._default_response


def _make_snapshot(
    label: str = "A",
    db: int = 1,
    generation: int | None = 5,
    fitness: float | None = 0.762,
    invalid_rate_inputs: tuple[int, int] | None = (100, 80),
    val_mean: float | None = 639.0,
    val_max: float | None = 980.0,
    keys: int | None = 157,
    pid: int | None = 49341,
    pid_alive: bool | None = True,
) -> RunSnapshot:
    """Factory for test RunSnapshots."""
    total, valid = invalid_rate_inputs if invalid_rate_inputs else (None, None)
    return RunSnapshot(
        run_spec=RunSpec(prefix="chains/test/static", db=db, label=label),
        generation=generation,
        metrics={"fitness": fitness},
        total_programs=total,
        valid_programs=valid,
        validator_mean_s=val_mean,
        validator_max_s=val_max,
        total_keys=keys,
        pid=pid,
        pid_alive=pid_alive,
    )


def _make_alert(
    alert_type: AlertType = AlertType.STALL,
    severity: AlertSeverity = AlertSeverity.WARN,
    run_label: str = "A",
    message: str = "Run A stalled at gen 5",
) -> Alert:
    return Alert(
        alert_type=alert_type,
        severity=severity,
        run_label=run_label,
        message=message,
    )


def _make_channel(
    recorder: RequestRecorder | None = None,
    repo: str = "owner/repo",
    pr_number: int = 42,
    token: str = "ghp_test",
) -> GitHubPRChannel:
    """Create a GitHubPRChannel with a mock transport."""
    rec = recorder or RequestRecorder()
    transport = httpx.MockTransport(rec.handler)
    return GitHubPRChannel(
        repo=repo,
        pr_number=pr_number,
        token=token,
        base_url="https://api.github.com",
        transport=transport,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Construction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_creates_instance(self) -> None:
        ch = _make_channel()
        assert isinstance(ch, GitHubPRChannel)

    def test_telegram_down_defaults_to_false(self) -> None:
        ch = _make_channel()
        assert ch.telegram_down is False

    def test_comment_id_starts_as_none(self) -> None:
        ch = _make_channel()
        assert ch._comment_id is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. check_health tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_health_success(self) -> None:
        recorder = RequestRecorder(
            responses={
                "GET /repos/owner/repo": httpx.Response(200, json={"id": 123}),
            }
        )
        ch = _make_channel(recorder=recorder)
        result = await ch.check_health()
        assert result is True
        assert len(recorder.requests) == 1
        assert recorder.requests[0].method == "GET"

    @pytest.mark.asyncio
    async def test_health_bad_token(self) -> None:
        recorder = RequestRecorder(
            responses={
                "GET /repos/owner/repo": httpx.Response(
                    401, json={"message": "Bad credentials"}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        result = await ch.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_repo_not_found(self) -> None:
        recorder = RequestRecorder(
            responses={
                "GET /repos/owner/repo": httpx.Response(
                    404, json={"message": "Not Found"}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        result = await ch.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_network_error(self) -> None:
        def error_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(error_handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
        )
        result = await ch.check_health()
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. send_status -- rolling comment tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_update(
    experiment_name: str = "hover/test-exp",
    snapshots: list[RunSnapshot] | None = None,
    alerts: list[Alert] | None = None,
    plots: list[PlotAttachment] | None = None,
    max_generations: int | None = 50,
    timestamp: datetime | None = None,
) -> StatusUpdate:
    return StatusUpdate(
        experiment_name=experiment_name,
        snapshots=snapshots or [_make_snapshot(label="A"), _make_snapshot(label="B")],
        alerts=alerts or [],
        plots=plots or [],
        max_generations=max_generations,
        timestamp=timestamp or datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC),
    )


class TestSendStatusRollingComment:
    @pytest.mark.asyncio
    async def test_first_call_creates_new_comment(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 99}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        update = _make_update()

        result = await ch.send_status(update)

        assert result is True
        assert len(recorder.requests) == 1
        assert recorder.requests[0].method == "POST"
        assert "/issues/42/comments" in str(recorder.requests[0].url)
        # Body should contain markdown table
        import json

        body = json.loads(recorder.requests[0].content)["body"]
        assert "|" in body
        # Rolling comment ID stored
        assert ch._comment_id == 99

    @pytest.mark.asyncio
    async def test_edits_existing_comment_after_threshold(self) -> None:
        recorder = RequestRecorder(
            responses={
                "PATCH /repos/owner/repo/issues/comments/99": httpx.Response(
                    200, json={"id": 99}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        ch._comment_id = 99
        ch._status_count = 25  # Past default threshold of 24
        update = _make_update()

        result = await ch.send_status(update)

        assert result is True
        assert len(recorder.requests) == 1
        assert recorder.requests[0].method == "PATCH"
        assert "/issues/comments/99" in str(recorder.requests[0].url)

    @pytest.mark.asyncio
    async def test_edit_failure_falls_back_to_new_comment(self) -> None:
        recorder = RequestRecorder(
            ordered_responses=[
                # PATCH fails (comment was deleted)
                (
                    "PATCH /repos/owner/repo/issues/comments/99",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                # POST succeeds with new ID
                (
                    "POST /repos/owner/repo/issues/42/comments",
                    httpx.Response(201, json={"id": 100}),
                ),
            ]
        )
        ch = _make_channel(recorder=recorder)
        ch._comment_id = 99
        ch._status_count = 25  # Past threshold to trigger edit attempt
        update = _make_update()

        result = await ch.send_status(update)

        assert result is True
        assert ch._comment_id == 100
        assert len(recorder.requests) == 2
        assert recorder.requests[0].method == "PATCH"
        assert recorder.requests[1].method == "POST"

    @pytest.mark.asyncio
    async def test_body_contains_experiment_name_and_table(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 1}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        update = _make_update(experiment_name="hover/my-exp")

        await ch.send_status(update)

        import json

        body = json.loads(recorder.requests[0].content)["body"]
        assert "hover/my-exp" in body
        assert "|" in body  # markdown table
        assert "A" in body  # run label A
        assert "B" in body  # run label B

    @pytest.mark.asyncio
    async def test_body_contains_alerts_section(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 1}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        alert = _make_alert(
            alert_type=AlertType.STALL,
            severity=AlertSeverity.WARN,
            message="Run A stalled at gen 5",
        )
        update = _make_update(alerts=[alert])

        await ch.send_status(update)

        import json

        body = json.loads(recorder.requests[0].content)["body"]
        assert "Alert" in body
        assert "stall" in body.lower() or "Run A stalled" in body

    @pytest.mark.asyncio
    async def test_telegram_down_header_present(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 1}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        ch.telegram_down = True
        update = _make_update()

        await ch.send_status(update)

        import json

        body = json.loads(recorder.requests[0].content)["body"]
        assert "TELEGRAM DOWN" in body
        assert body.startswith("> ")

    @pytest.mark.asyncio
    async def test_telegram_down_false_no_header(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 1}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        assert ch.telegram_down is False
        update = _make_update()

        await ch.send_status(update)

        import json

        body = json.loads(recorder.requests[0].content)["body"]
        assert "TELEGRAM DOWN" not in body

    @pytest.mark.asyncio
    async def test_api_failure_returns_false(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    500, json={"message": "Internal Server Error"}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        update = _make_update()

        result = await ch.send_status(update)

        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. send_alert tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSendAlert:
    @pytest.mark.asyncio
    async def test_always_creates_new_comment(self) -> None:
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 200}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        ch._comment_id = 99  # Rolling comment exists
        alert = _make_alert(
            alert_type=AlertType.CRASH,
            severity=AlertSeverity.ERROR,
            message="Run B process (PID 12345) is not alive",
        )

        result = await ch.send_alert(alert)

        assert result is True
        assert len(recorder.requests) == 1
        assert recorder.requests[0].method == "POST"
        assert "/issues/42/comments" in str(recorder.requests[0].url)
        # Rolling comment ID should NOT be changed by alert
        assert ch._comment_id == 99


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plot upload + cache-busting tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUploadPlot:
    @pytest.mark.asyncio
    async def test_upload_plot_success(self, tmp_path: Path) -> None:
        png_file = tmp_path / "fitness.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        plot = PlotAttachment(path=png_file, caption="Fitness curves")

        recorder = RequestRecorder(
            ordered_responses=[
                # GET to check if file exists (404 = new file)
                (
                    "GET /repos/owner/repo/contents/",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                # PUT to upload the file
                (
                    "PUT /repos/owner/repo/contents/",
                    httpx.Response(
                        201,
                        json={
                            "content": {
                                "download_url": "https://raw.githubusercontent.com/owner/repo/main/plots/fitness.png"
                            }
                        },
                    ),
                ),
            ]
        )
        ch = _make_channel(recorder=recorder)
        url = await ch._upload_plot(plot, branch="exp/my-branch")

        assert url is not None
        assert "fitness.png" in url

    @pytest.mark.asyncio
    async def test_upload_plot_failure_returns_none(self, tmp_path: Path) -> None:
        png_file = tmp_path / "broken.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        plot = PlotAttachment(path=png_file, caption="Broken plot")

        recorder = RequestRecorder(
            ordered_responses=[
                # GET returns 404
                (
                    "GET /repos/owner/repo/contents/",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                # PUT fails
                (
                    "PUT /repos/owner/repo/contents/",
                    httpx.Response(500, json={"message": "Server Error"}),
                ),
            ]
        )
        ch = _make_channel(recorder=recorder)
        url = await ch._upload_plot(plot, branch="exp/my-branch")

        assert url is None

    @pytest.mark.asyncio
    async def test_send_status_with_plots_embeds_urls(self, tmp_path: Path) -> None:
        """send_status with plots: uploaded plots are embedded as markdown images."""
        png_file = tmp_path / "fitness.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        plot = PlotAttachment(path=png_file, caption="Fitness curves")
        ts = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
        update = _make_update(plots=[plot], timestamp=ts)

        recorder = RequestRecorder(
            ordered_responses=[
                # GET for plot (new file)
                (
                    "GET /repos/owner/repo/contents/",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                # PUT to upload
                (
                    "PUT /repos/owner/repo/contents/",
                    httpx.Response(
                        201,
                        json={
                            "content": {
                                "download_url": "https://raw.githubusercontent.com/owner/repo/main/plots/fitness.png"
                            }
                        },
                    ),
                ),
                # POST the PR comment
                (
                    "POST /repos/owner/repo/issues/42/comments",
                    httpx.Response(201, json={"id": 1}),
                ),
            ]
        )
        transport = httpx.MockTransport(recorder.handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
            branch="exp/my-branch",
        )

        result = await ch.send_status(update)

        assert result is True
        # Find the POST comment request
        post_reqs = [r for r in recorder.requests if r.method == "POST"]
        assert len(post_reqs) == 1
        import json

        body = json.loads(post_reqs[0].content)["body"]
        # Should contain cache-busted image URL
        assert "![Fitness curves](" in body
        assert "?v=" in body

    @pytest.mark.asyncio
    async def test_send_status_with_plots_upload_failure_still_posts(
        self, tmp_path: Path
    ) -> None:
        """When plot upload fails, comment is still posted with text reference."""
        png_file = tmp_path / "broken.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        plot = PlotAttachment(path=png_file, caption="Broken plot")
        update = _make_update(plots=[plot])

        recorder = RequestRecorder(
            ordered_responses=[
                # GET returns 404
                (
                    "GET /repos/owner/repo/contents/",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                # PUT fails
                (
                    "PUT /repos/owner/repo/contents/",
                    httpx.Response(500, json={"message": "Server Error"}),
                ),
                # POST comment still succeeds
                (
                    "POST /repos/owner/repo/issues/42/comments",
                    httpx.Response(201, json={"id": 1}),
                ),
            ]
        )
        transport = httpx.MockTransport(recorder.handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
            branch="exp/my-branch",
        )

        result = await ch.send_status(update)

        assert result is True
        post_reqs = [r for r in recorder.requests if r.method == "POST"]
        assert len(post_reqs) == 1
        import json

        body = json.loads(post_reqs[0].content)["body"]
        # Should contain caption as text reference, not image embed
        assert "Broken plot" in body
        assert "broken.png" in body


class TestCacheBustUrl:
    def test_appends_query_param(self) -> None:
        result = GitHubPRChannel._cache_bust_url(
            "https://example.com/plot.png", timestamp=1234567890
        )
        assert result == "https://example.com/plot.png?v=1234567890"

    def test_preserves_existing_query_params(self) -> None:
        result = GitHubPRChannel._cache_bust_url(
            "https://example.com/plot.png?ref=main", timestamp=1234567890
        )
        assert result == "https://example.com/plot.png?ref=main&v=1234567890"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Experiment-name upload path tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestExperimentNameUploadPath:
    @pytest.mark.asyncio
    async def test_upload_uses_experiment_path(self, tmp_path: Path) -> None:
        """With experiment_name set, uploads to experiments/{name}/plots/."""
        png_file = tmp_path / "fitness.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        plot = PlotAttachment(path=png_file, caption="Fitness")

        recorder = RequestRecorder(
            ordered_responses=[
                (
                    "GET /repos/owner/repo/contents/",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                (
                    "PUT /repos/owner/repo/contents/",
                    httpx.Response(201, json={"content": {}}),
                ),
            ]
        )
        transport = httpx.MockTransport(recorder.handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
            branch="exp/my-branch",
            experiment_name="hover/test-exp",
        )

        url = await ch._upload_plot(plot, branch="exp/my-branch")

        assert url is not None
        assert "experiments/hover/test-exp/plots/fitness.png" in url
        assert url.startswith("https://raw.githubusercontent.com/")

        put_req = [r for r in recorder.requests if r.method == "PUT"][0]
        assert "experiments/hover/test-exp/plots/" in str(put_req.url)

    @pytest.mark.asyncio
    async def test_upload_falls_back_to_generic_path(self, tmp_path: Path) -> None:
        """Without experiment_name, uploads to plots/."""
        png_file = tmp_path / "fitness.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        plot = PlotAttachment(path=png_file, caption="Fitness")

        recorder = RequestRecorder(
            ordered_responses=[
                (
                    "GET /repos/owner/repo/contents/",
                    httpx.Response(404, json={"message": "Not Found"}),
                ),
                (
                    "PUT /repos/owner/repo/contents/",
                    httpx.Response(201, json={"content": {}}),
                ),
            ]
        )
        transport = httpx.MockTransport(recorder.handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
            branch="exp/my-branch",
        )

        url = await ch._upload_plot(plot, branch="exp/my-branch")

        assert url is not None
        assert "/plots/fitness.png" in url
        assert "experiments/" not in url


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Rolling comment Redis persistence tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRollingCommentRedis:
    def test_set_and_get_rolling_comment_id(self) -> None:
        """Rolling comment ID round-trips through Redis."""
        import fakeredis

        r = fakeredis.FakeRedis(decode_responses=True)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            experiment_name="hover/test",
            rolling_comment_redis=r,
        )

        ch._set_rolling_comment_id(999)
        result = ch._get_rolling_comment_id()
        assert result == 999

    def test_get_returns_none_when_no_key(self) -> None:
        """Returns None when no rolling comment ID stored in Redis."""
        import fakeredis

        r = fakeredis.FakeRedis(decode_responses=True)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            experiment_name="hover/test",
            rolling_comment_redis=r,
        )

        result = ch._get_rolling_comment_id()
        assert result is None

    def test_loads_existing_id_on_construction(self) -> None:
        """Constructor loads rolling comment ID from Redis if present."""
        import fakeredis

        r = fakeredis.FakeRedis(decode_responses=True)
        r.set("experiments:hover/test:rolling_comment_id", "555")

        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            experiment_name="hover/test",
            rolling_comment_redis=r,
        )

        assert ch._comment_id == 555

    def test_no_redis_falls_back_to_memory(self) -> None:
        """Without Redis, _get_rolling_comment_id returns in-memory comment_id."""
        ch = _make_channel()
        ch._comment_id = 123
        assert ch._get_rolling_comment_id() == 123

    def test_redis_key_includes_experiment_name(self) -> None:
        """Redis key uses experiments:{name}:rolling_comment_id format."""
        import fakeredis

        r = fakeredis.FakeRedis(decode_responses=True)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            experiment_name="hover/my-exp",
            rolling_comment_redis=r,
        )

        ch._set_rolling_comment_id(777)
        assert r.get("experiments:hover/my-exp:rolling_comment_id") == "777"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Rolling comment threshold tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRollingCommentThreshold:
    @pytest.mark.asyncio
    async def test_posts_new_comments_before_threshold(self) -> None:
        """Before threshold, always creates new comments even if comment_id set."""
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 10}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        ch._comment_id = 99  # Already has a comment
        ch._status_count = 0  # Below threshold
        ch._rolling_threshold_hours = 24

        result = await ch.send_status(_make_update())

        assert result is True
        assert recorder.requests[0].method == "POST"

    @pytest.mark.asyncio
    async def test_edits_comment_after_threshold(self) -> None:
        """After threshold, edits the rolling comment."""
        recorder = RequestRecorder(
            responses={
                "PATCH /repos/owner/repo/issues/comments/99": httpx.Response(
                    200, json={"id": 99}
                ),
            }
        )
        ch = _make_channel(recorder=recorder)
        ch._comment_id = 99
        ch._status_count = 24  # Will become 25, past threshold of 24
        ch._rolling_threshold_hours = 24

        result = await ch.send_status(_make_update())

        assert result is True
        assert recorder.requests[0].method == "PATCH"

    @pytest.mark.asyncio
    async def test_persists_rolling_id_at_threshold_boundary(self) -> None:
        """At exactly the threshold count, saves the comment ID to Redis."""
        import fakeredis

        r = fakeredis.FakeRedis(decode_responses=True)
        recorder = RequestRecorder(
            responses={
                "POST /repos/owner/repo/issues/42/comments": httpx.Response(
                    201, json={"id": 888}
                ),
            }
        )
        transport = httpx.MockTransport(recorder.handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
            experiment_name="hover/test",
            rolling_comment_redis=r,
            rolling_comment_threshold_hours=3,
        )
        ch._status_count = 2  # Next call will be count=3 (== threshold)

        await ch.send_status(_make_update())

        assert r.get("experiments:hover/test:rolling_comment_id") == "888"

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """Custom threshold of 1 means edit starts at cycle 2."""
        recorder = RequestRecorder(
            responses={
                "PATCH /repos/owner/repo/issues/comments/50": httpx.Response(
                    200, json={"id": 50}
                ),
            }
        )
        transport = httpx.MockTransport(recorder.handler)
        ch = GitHubPRChannel(
            repo="owner/repo",
            pr_number=42,
            token="ghp_test",
            transport=transport,
            rolling_comment_threshold_hours=1,
        )
        ch._comment_id = 50
        ch._status_count = 1  # Will become 2, past threshold of 1

        result = await ch.send_status(_make_update())

        assert result is True
        assert recorder.requests[0].method == "PATCH"
