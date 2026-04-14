"""Telegram notification channel using httpx.

Replaces tools/telegram_notify.py with:
- httpx.AsyncClient instead of requests
- Retry with exponential backoff
- Consecutive failure tracking for cross-channel escalation
- Startup health probe via getMe
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from gigaevo.monitoring.alerts import Alert
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    StatusUpdate,
    format_alert_message,
    format_status_table_telegram,
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
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._transport_factory = transport_factory
        self._client: httpx.AsyncClient | None = None
        # Track the event loop that owns the current client so we can detect
        # when the watchdog enters a new asyncio.run() cycle and rebuild the
        # client+transport in the new loop. Without this, httpx raises
        # "Event loop is closed" because the underlying connection pool is
        # bound to a loop that has since been torn down.
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._consecutive_failures = 0

    @classmethod
    def from_env(cls) -> TelegramChannel | None:
        """Build a TelegramChannel from standard env vars, or None if unconfigured.

        Reads:
          TELEGRAM_BOT_TOKEN  -- required
          TELEGRAM_CHAT_ID    -- required
          HTTPS_PROXY / https_proxy -- optional, wraps in AsyncHTTPTransport
        """
        import os

        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return None
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        # Use a factory (not a pre-built transport): the watchdog reuses this
        # channel across many asyncio.run() cycles, and AsyncHTTPTransport's
        # connection pool is loop-bound. A factory lets _get_client() rebuild
        # the transport whenever the event loop changes.
        factory = (lambda: httpx.AsyncHTTPTransport(proxy=proxy)) if proxy else None
        return cls(bot_token=token, chat_id=chat_id, transport_factory=factory)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def _get_client(self) -> httpx.AsyncClient:
        current_loop = asyncio.get_running_loop()
        stale_loop = (
            self._client_loop is not None and self._client_loop is not current_loop
        )

        if self._client is None or self._client.is_closed or stale_loop:
            # Drop the old client without aclose(): its loop is gone, we can't
            # await on it. The connection pool becomes garbage.
            self._client = None

            kwargs: dict[str, Any] = {"timeout": httpx.Timeout(30.0)}
            if self._transport_factory is not None:
                kwargs["transport"] = self._transport_factory()
            elif self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
            self._client_loop = current_loop
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

    # ── Core send methods ───────────────────────────────────────────────────

    async def _send_message(self, text: str, *, parse_mode: str = "HTML") -> bool:
        """Send a text message to Telegram with retry.

        Retries on HTTP 429, 5xx, and network errors. No retry on 4xx
        (except 429) because those indicate a bug in our request.
        """
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        return await self._post_with_retry("sendMessage", payload)

    async def _send_photo(self, photo_path: Path, caption: str = "") -> bool:
        """Send a photo to Telegram with retry.

        On HTTP 400 (often file-too-large), falls back to sendDocument which
        supports up to 50 MB vs sendPhoto's 10 MB limit.
        """
        try:
            client = await self._get_client()
            with open(photo_path, "rb") as f:
                files = {"photo": (photo_path.name, f, "image/png")}
                data = {
                    "chat_id": self._chat_id,
                    "caption": caption[:1024],
                    "parse_mode": "HTML",
                }
                for attempt in range(_MAX_RETRIES):
                    try:
                        f.seek(0)
                        resp = await client.post(
                            self._api_url("sendPhoto"),
                            data=data,
                            files=files,
                        )
                        if resp.status_code == 200:
                            return True
                        if resp.status_code == 429 or resp.status_code >= 500:
                            await self._backoff(attempt)
                            continue
                        if resp.status_code == 400:
                            _log.warning(
                                f"sendPhoto got 400 ({resp.text}), "
                                "falling back to sendDocument"
                            )
                            return await self._send_document(photo_path, caption)
                        _log.warning(
                            f"sendPhoto failed: {resp.status_code} {resp.text}"
                        )
                        return False
                    except httpx.HTTPError as exc:
                        _log.warning(
                            f"sendPhoto network error (attempt {attempt + 1}): {exc}"
                        )
                        if attempt < _MAX_RETRIES - 1:
                            await self._backoff(attempt)
                return False
        except Exception as exc:
            _log.error(f"sendPhoto error: {exc}")
            return False

    async def _send_document(self, doc_path: Path, caption: str = "") -> bool:
        """Send a file as a document (up to 50 MB). Used as fallback for large photos."""
        try:
            client = await self._get_client()
            with open(doc_path, "rb") as f:
                files = {"document": (doc_path.name, f, "application/octet-stream")}
                data = {
                    "chat_id": self._chat_id,
                    "caption": caption[:1024],
                    "parse_mode": "HTML",
                }
                for attempt in range(_MAX_RETRIES):
                    try:
                        f.seek(0)
                        resp = await client.post(
                            self._api_url("sendDocument"),
                            data=data,
                            files=files,
                        )
                        if resp.status_code == 200:
                            return True
                        if resp.status_code == 429 or resp.status_code >= 500:
                            await self._backoff(attempt)
                            continue
                        _log.warning(
                            f"sendDocument failed: {resp.status_code} {resp.text}"
                        )
                        return False
                    except httpx.HTTPError as exc:
                        _log.warning(
                            f"sendDocument network error (attempt {attempt + 1}): {exc}"
                        )
                        if attempt < _MAX_RETRIES - 1:
                            await self._backoff(attempt)
                return False
        except Exception as exc:
            _log.error(f"sendDocument error: {exc}")
            return False

    async def _post_with_retry(self, method: str, payload: dict) -> bool:
        """POST to Telegram Bot API with exponential backoff retry.

        Returns True on success (HTTP 200 + ok:true), False on failure.
        Updates consecutive_failures counter.
        """
        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await client.post(self._api_url(method), json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok", False):
                        self._consecutive_failures = 0
                        return True
                    _log.warning(f"{method} returned ok=false: {data}")
                    self._consecutive_failures += 1
                    return False

                if resp.status_code == 429 or resp.status_code >= 500:
                    _log.info(
                        f"{method} got {resp.status_code}, "
                        f"retry {attempt + 1}/{_MAX_RETRIES}"
                    )
                    await self._backoff(attempt)
                    continue

                # 4xx (except 429) -- our bug, no retry
                _log.warning(f"{method} failed: {resp.status_code} {resp.text}")
                self._consecutive_failures += 1
                return False

            except httpx.HTTPError as exc:
                last_exc = exc
                _log.warning(f"{method} network error (attempt {attempt + 1}): {exc}")
                if attempt < _MAX_RETRIES - 1:
                    await self._backoff(attempt)

        # All retries exhausted
        _log.error(f"{method} failed after {_MAX_RETRIES} attempts: {last_exc}")
        self._consecutive_failures += 1
        return False

    @staticmethod
    async def _backoff(attempt: int) -> None:
        delay = _BACKOFF_BASE * (2**attempt)
        await asyncio.sleep(delay)

    # ── Public channel methods ──────────────────────────────────────────────

    async def send_status(self, update: StatusUpdate) -> bool:
        """Send full status update: plugin body or table + alerts + photos."""
        if update.telegram_body:
            text_ok = await self._send_message(update.telegram_body, parse_mode="")
        else:
            parts: list[str] = []

            parts.append(f"<b>{update.experiment_name}</b>")
            if update.max_generations is not None:
                parts.append(f"Target: {update.max_generations} generations")
            parts.append("")

            table = format_status_table_telegram(update.snapshots)
            parts.append(table)

            if update.has_alerts:
                parts.append("")
                parts.append("<b>Alerts:</b>")
                for alert in update.alerts:
                    parts.append(f"  {format_alert_message(alert)}")

            message = "\n".join(parts)
            text_ok = await self._send_message(message)

        for plot in update.plots:
            photo_ok = await self._send_photo(plot.path, plot.caption)
            if not photo_ok:
                _log.warning(f"Failed to send plot: {plot.path}")

        return text_ok

    async def send_alert(self, alert: Alert) -> bool:
        """Send a single alert as a text message."""
        message = format_alert_message(alert)
        return await self._send_message(message)
