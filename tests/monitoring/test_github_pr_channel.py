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
    """Records HTTP requests for assertion in tests."""

    def __init__(self, responses: dict[str, httpx.Response] | None = None):
        self.requests: list[httpx.Request] = []
        self._responses = responses or {}
        self._default_response = httpx.Response(200, json={"id": 1})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
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
                "GET /repos/owner/repo": httpx.Response(401, json={"message": "Bad credentials"}),
            }
        )
        ch = _make_channel(recorder=recorder)
        result = await ch.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_repo_not_found(self) -> None:
        recorder = RequestRecorder(
            responses={
                "GET /repos/owner/repo": httpx.Response(404, json={"message": "Not Found"}),
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
