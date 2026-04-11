"""Telegram notification channel using httpx.

Replaces tools/telegram_notify.py with:
- httpx.AsyncClient instead of requests
- Retry with exponential backoff
- Consecutive failure tracking for cross-channel escalation
- Startup health probe via getMe
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from gigaevo.monitoring.alerts import Alert
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    StatusUpdate,
)

_log = logger.bind(component="telegram")

_DEFAULT_BASE_URL = "https://api.telegram.org"

# Retry config
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5  # seconds; multiplied by 2^attempt


class TelegramChannel(NotificationChannel):
    """Telegram Bot API notification channel.

    Uses httpx.AsyncClient for all API calls. Implements retry with
    exponential backoff on transient errors (HTTP 429, 5xx, network).
    Tracks consecutive failures for cross-channel escalation.
    """

    CONSECUTIVE_FAILURE_THRESHOLD = 3

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        base_url: str = _DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs: dict[str, Any] = {"timeout": httpx.Timeout(30.0)}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client. Safe to call multiple times."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _api_url(self, method: str) -> str:
        return f"{self._base_url}/bot{self._bot_token}/{method}"

    async def check_health(self) -> bool:
        """Call getMe to verify the bot token is valid and Telegram is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get(self._api_url("getMe"))
            if resp.status_code == 200:
                data = resp.json()
                return data.get("ok", False) is True
            return False
        except Exception as exc:
            _log.warning(f"Health check failed: {exc}")
            return False

    # send_status and send_alert implemented in Task 4
    async def send_status(self, update: StatusUpdate) -> bool:
        raise NotImplementedError  # placeholder until Task 4

    async def send_alert(self, alert: Alert) -> bool:
        raise NotImplementedError  # placeholder until Task 4
