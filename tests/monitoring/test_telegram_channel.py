"""Tests for gigaevo.monitoring.telegram_channel — TelegramChannel with httpx."""

from __future__ import annotations

import httpx
import pytest

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
