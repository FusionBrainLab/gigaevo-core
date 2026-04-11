"""Tests for gigaevo.monitoring.telegram_channel — TelegramChannel with httpx."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from gigaevo.monitoring.alerts import Alert, AlertSeverity, AlertType
from gigaevo.monitoring.notifications import PlotAttachment, StatusUpdate
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.telegram_channel import TelegramChannel

# ── Test infrastructure ─────────────────────────────────────────────────────


def _make_transport(handler):
    """Create an httpx MockTransport from a handler function.

    handler signature: (request: httpx.Request) -> httpx.Response
    """
    return httpx.MockTransport(handler)


def _make_channel(handler) -> TelegramChannel:
    """Create a TelegramChannel with a mock transport."""
    transport = _make_transport(handler)
    return TelegramChannel(
        bot_token="test-token",
        chat_id="12345",
        transport=transport,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Construction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_creates_instance(self) -> None:
        channel = TelegramChannel(bot_token="test-token", chat_id="12345")
        assert isinstance(channel, TelegramChannel)

    def test_consecutive_failures_starts_at_zero(self) -> None:
        channel = TelegramChannel(bot_token="test-token", chat_id="12345")
        assert channel.consecutive_failures == 0

    def test_consecutive_failure_threshold_is_three(self) -> None:
        channel = TelegramChannel(bot_token="test-token", chat_id="12345")
        assert channel.CONSECUTIVE_FAILURE_THRESHOLD == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. check_health tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_check_health_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/bottest-token/getMe"
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": {
                        "id": 123,
                        "is_bot": True,
                        "first_name": "TestBot",
                    },
                },
            )

        channel = _make_channel(handler)
        result = await channel.check_health()
        assert result is True
        await channel.close()

    @pytest.mark.asyncio
    async def test_check_health_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"ok": False, "description": "Unauthorized"},
            )

        channel = _make_channel(handler)
        result = await channel.check_health()
        assert result is False
        await channel.close()

    @pytest.mark.asyncio
    async def test_check_health_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        channel = _make_channel(handler)
        result = await channel.check_health()
        assert result is False
        await channel.close()

    @pytest.mark.asyncio
    async def test_check_health_malformed_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False})

        channel = _make_channel(handler)
        result = await channel.check_health()
        assert result is False
        await channel.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Retry logic + _send_message + failure tracking tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSendMessageRetry:
    @pytest.mark.asyncio
    async def test_retry_on_429(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    429,
                    json={"ok": False, "description": "Too Many Requests"},
                    headers={"Retry-After": "1"},
                )
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        channel = _make_channel(handler)
        result = await channel._send_message("test")
        assert result is True
        assert call_count == 2
        await channel.close()

    @pytest.mark.asyncio
    async def test_retry_on_500(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(
                    500, json={"ok": False, "description": "Internal Server Error"}
                )
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        channel = _make_channel(handler)
        result = await channel._send_message("test")
        assert result is True
        assert call_count == 3
        await channel.close()

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                500, json={"ok": False, "description": "Internal Server Error"}
            )

        channel = _make_channel(handler)
        result = await channel._send_message("test")
        assert result is False
        assert call_count == 3
        assert channel.consecutive_failures == 1
        await channel.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"ok": False, "description": "Bad Request"})

        channel = _make_channel(handler)
        result = await channel._send_message("test")
        assert result is False
        assert call_count == 1
        await channel.close()

    @pytest.mark.asyncio
    async def test_network_error_retry(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        channel = _make_channel(handler)
        result = await channel._send_message("test")
        assert result is True
        assert call_count == 2
        await channel.close()


class TestConsecutiveFailures:
    @pytest.mark.asyncio
    async def test_reset_on_success(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            # First 3 calls (first _send_message): always 500 -> exhaust retries
            if call_count <= 3:
                return httpx.Response(
                    500,
                    json={"ok": False, "description": "Internal Server Error"},
                )
            # Second _send_message: success
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        channel = _make_channel(handler)

        # First send fails (exhausts retries)
        result1 = await channel._send_message("test1")
        assert result1 is False
        assert channel.consecutive_failures == 1

        # Second send succeeds
        result2 = await channel._send_message("test2")
        assert result2 is True
        assert channel.consecutive_failures == 0
        await channel.close()

    @pytest.mark.asyncio
    async def test_accumulates(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500, json={"ok": False, "description": "Internal Server Error"}
            )

        channel = _make_channel(handler)

        await channel._send_message("test1")
        await channel._send_message("test2")
        await channel._send_message("test3")

        assert channel.consecutive_failures == 3
        await channel.close()


# ── Factories for integration tests ─────────────────────────────────────────


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


def _recording_channel(responses=None):
    """Create a TelegramChannel that records all requests.

    Returns (channel, recorded_requests_list).
    Default: all requests return 200 + ok:true.
    """
    recorded: list[httpx.Request] = []

    if responses is None:
        responses = {}

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        # Check for custom responses by URL path
        path = request.url.path
        if path in responses:
            return responses[path]
        return httpx.Response(
            200, json={"ok": True, "result": {"message_id": 1}}
        )

    transport = _make_transport(handler)
    channel = TelegramChannel(
        bot_token="test-token",
        chat_id="12345",
        transport=transport,
    )
    return channel, recorded


# ═══════════════════════════════════════════════════════════════════════════════
# 4. send_status + send_alert integration tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSendStatus:
    @pytest.mark.asyncio
    async def test_full_table_message(self) -> None:
        channel, recorded = _recording_channel()
        update = StatusUpdate(
            experiment_name="hover/test-exp",
            snapshots=[
                _make_snapshot(label="A", generation=5, fitness=0.762),
                _make_snapshot(label="B", db=2, generation=8, fitness=0.831),
            ],
        )

        result = await channel.send_status(update)
        assert result is True

        # Should have sent exactly one message (no plots)
        assert len(recorded) == 1
        req = recorded[0]
        body = json.loads(req.content)
        assert body["chat_id"] == "12345"
        assert body["parse_mode"] == "HTML"
        text = body["text"]
        assert "hover/test-exp" in text
        assert "<pre>" in text
        assert "A" in text
        assert "B" in text
        assert "76.2%" in text
        assert "83.1%" in text
        await channel.close()

    @pytest.mark.asyncio
    async def test_with_alerts(self) -> None:
        channel, recorded = _recording_channel()
        update = StatusUpdate(
            experiment_name="hover/test-exp",
            snapshots=[_make_snapshot()],
            alerts=[_make_alert(severity=AlertSeverity.WARN, message="Run A stalled at gen 5")],
        )

        result = await channel.send_status(update)
        assert result is True

        body = json.loads(recorded[0].content)
        text = body["text"]
        assert "Alerts:" in text
        assert "stall" in text.lower() or "WARNING" in text
        await channel.close()

    @pytest.mark.asyncio
    async def test_with_plot_photos(self, tmp_path: Path) -> None:
        # Create a fake PNG file
        png_file = tmp_path / "plot.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        channel, recorded = _recording_channel()
        update = StatusUpdate(
            experiment_name="hover/test-exp",
            snapshots=[_make_snapshot()],
            plots=[PlotAttachment(path=png_file, caption="Fitness curves")],
        )

        result = await channel.send_status(update)
        assert result is True
        # Two requests: sendMessage + sendPhoto
        assert len(recorded) == 2
        assert "/sendMessage" in str(recorded[0].url)
        assert "/sendPhoto" in str(recorded[1].url)
        await channel.close()

    @pytest.mark.asyncio
    async def test_photo_failure_does_not_fail_call(self, tmp_path: Path) -> None:
        png_file = tmp_path / "plot.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        # sendPhoto always returns 500
        responses = {
            "/bottest-token/sendPhoto": httpx.Response(
                500, json={"ok": False, "description": "Internal Server Error"}
            ),
        }
        channel, recorded = _recording_channel(responses)
        update = StatusUpdate(
            experiment_name="hover/test-exp",
            snapshots=[_make_snapshot()],
            plots=[PlotAttachment(path=png_file, caption="Fitness curves")],
        )

        result = await channel.send_status(update)
        # Text succeeded even though photo failed
        assert result is True
        # consecutive_failures reset by successful text send
        assert channel.consecutive_failures == 0
        await channel.close()

    @pytest.mark.asyncio
    async def test_text_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500, json={"ok": False, "description": "Internal Server Error"}
            )

        channel = _make_channel(handler)
        update = StatusUpdate(
            experiment_name="hover/test-exp",
            snapshots=[_make_snapshot()],
        )

        result = await channel.send_status(update)
        assert result is False
        assert channel.consecutive_failures == 1
        await channel.close()


class TestSendAlert:
    @pytest.mark.asyncio
    async def test_sends_formatted_message(self) -> None:
        channel, recorded = _recording_channel()
        alert = _make_alert(
            alert_type=AlertType.CRASH,
            severity=AlertSeverity.ERROR,
            message="Run A process (PID 49341) is not alive.",
        )

        result = await channel.send_alert(alert)
        assert result is True

        body = json.loads(recorded[0].content)
        text = body["text"]
        assert "ERROR" in text
        assert "crash" in text
        await channel.close()
