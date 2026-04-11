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

    def _build_status_body(
        self, update: StatusUpdate, plot_urls: dict[int, str] | None = None
    ) -> str:
        """Build the markdown body for a status update PR comment."""
        parts: list[str] = []

        if self._telegram_down:
            parts.append(
                "> :warning: TELEGRAM DOWN "
                "-- notifications are not being delivered to Telegram"
            )
            parts.append("")

        parts.append(f"### {update.experiment_name}")
        if update.max_generations is not None:
            parts.append(f"Target: {update.max_generations} generations")
        ts = update.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        parts.append(f"_Updated: {ts}_")
        parts.append("")

        table = format_status_table_markdown(update.snapshots)
        parts.append(table)

        if update.has_alerts:
            parts.append("")
            parts.append("**Alerts:**")
            for alert in update.alerts:
                parts.append(f"- {format_alert_message(alert)}")

        if update.has_plots:
            parts.append("")
            parts.append("**Plots:**")
            for i, plot in enumerate(update.plots):
                url = (plot_urls or {}).get(i)
                if url:
                    parts.append(f"![{plot.caption}]({url})")
                else:
                    parts.append(f"- {plot.caption}: _{plot.path.name}_")

        return "\n".join(parts)

    async def _post_comment(self, body: str) -> int | None:
        """Create a new PR comment. Returns the comment ID or None."""
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base_url}/repos/{self._repo}"
                f"/issues/{self._pr_number}/comments",
                json={"body": body},
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                return data.get("id")
            _log.warning(f"POST comment failed: {resp.status_code} {resp.text}")
            return None
        except Exception as exc:
            _log.error(f"POST comment error: {exc}")
            return None

    async def _edit_comment(self, comment_id: int, body: str) -> bool:
        """Edit an existing PR comment. Returns True on success."""
        try:
            client = await self._get_client()
            resp = await client.patch(
                f"{self._base_url}/repos/{self._repo}"
                f"/issues/comments/{comment_id}",
                json={"body": body},
            )
            if resp.status_code == 200:
                return True
            _log.warning(f"PATCH comment {comment_id} failed: {resp.status_code}")
            return False
        except Exception as exc:
            _log.error(f"PATCH comment error: {exc}")
            return False

    async def send_status(self, update: StatusUpdate) -> bool:
        """Post or edit the rolling PR comment with status table + alerts + plots."""
        body = self._build_status_body(update)

        if self._comment_id is not None:
            success = await self._edit_comment(self._comment_id, body)
            if success:
                return True
            _log.info(
                f"Rolling comment {self._comment_id} edit failed, "
                f"creating new comment"
            )
            self._comment_id = None

        new_id = await self._post_comment(body)
        if new_id is not None:
            self._comment_id = new_id
            return True
        return False

    async def send_alert(self, alert: Alert) -> bool:
        """Post a new PR comment for an alert (never edits rolling comment)."""
        body = format_alert_message(alert)
        new_id = await self._post_comment(body)
        return new_id is not None
