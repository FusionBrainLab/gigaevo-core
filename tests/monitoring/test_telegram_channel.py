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
