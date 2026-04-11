"""GitHub PR comment notification channel using httpx.

Replaces the urllib.request code in watchdog templates with:
- httpx.AsyncClient for all GitHub API calls
- Rolling comment (POST once, PATCH thereafter)
- Plot upload to GitHub Release assets with cache-busting URLs
- Cross-channel telegram_down header
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from gigaevo.monitoring.alerts import Alert
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    StatusUpdate,
    format_alert_message,
    format_status_table_markdown,
)

_log = logger.bind(component="github_pr")

_DEFAULT_BASE_URL = "https://api.github.com"


class GitHubPRChannel(NotificationChannel):
    """GitHub PR comment notification channel.

    Posts status tables and plot images as PR comments. Uses rolling
    comment pattern: first post creates, subsequent posts edit.
    """

    def __init__(
        self,
        repo: str,
        pr_number: int,
        token: str,
        base_url: str = _DEFAULT_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        branch: str | None = None,
    ) -> None:
        self._repo = repo
        self._pr_number = pr_number
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._branch = branch
        self._client: httpx.AsyncClient | None = None
        self._comment_id: int | None = None
        self._telegram_down = False

    @property
    def telegram_down(self) -> bool:
        return self._telegram_down

    @telegram_down.setter
    def telegram_down(self, value: bool) -> None:
        self._telegram_down = value

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(30.0),
                "headers": {
                    "Authorization": f"token {self._token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        """Verify the token and repo are valid by calling GET /repos/{owner}/{repo}."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self._base_url}/repos/{self._repo}")
            return resp.status_code == 200
        except Exception as exc:
            _log.warning(f"GitHub health check failed: {exc}")
            return False

    async def send_status(self, update: StatusUpdate) -> bool:
        """Post or edit the rolling PR comment with status table + alerts + plots."""
        raise NotImplementedError

    async def send_alert(self, alert: Alert) -> bool:
        """Post a new PR comment for an alert (never edits rolling comment)."""
        raise NotImplementedError
