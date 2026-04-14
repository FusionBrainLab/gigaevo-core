"""GitHub PR comment notification channel using httpx.

Replaces the urllib.request code in watchdog templates with:
- httpx.AsyncClient for all GitHub API calls
- Rolling comment (POST first N hours, PATCH thereafter using Redis-tracked ID)
- Plot upload to experiment branch with cache-busting URLs
- Cross-channel telegram_down header
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
import redis as redis_lib

from gigaevo.monitoring.alerts import Alert
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    PlotAttachment,
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
        experiment_name: str | None = None,
        rolling_comment_redis: redis_lib.Redis | None = None,
        rolling_comment_threshold_hours: int = 24,
    ) -> None:
        self._repo = repo
        self._pr_number = pr_number
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._branch = branch
        self._experiment_name = experiment_name
        self._rolling_redis = rolling_comment_redis
        self._rolling_threshold_hours = rolling_comment_threshold_hours
        self._client: httpx.AsyncClient | None = None
        self._comment_id: int | None = None
        self._status_count: int = 0
        self._telegram_down = False

        if self._rolling_redis and self._experiment_name:
            existing = self._get_rolling_comment_id()
            if existing:
                self._comment_id = existing
                _log.info(f"Loaded rolling comment ID from Redis: {existing}")

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
                f"{self._base_url}/repos/{self._repo}/issues/comments/{comment_id}",
                json={"body": body},
            )
            if resp.status_code == 200:
                return True
            _log.warning(f"PATCH comment {comment_id} failed: {resp.status_code}")
            return False
        except Exception as exc:
            _log.error(f"PATCH comment error: {exc}")
            return False

    async def _upload_plot(self, plot: PlotAttachment, branch: str) -> str | None:
        """Upload a plot file to the repo via GitHub Contents API.

        Returns the raw.githubusercontent.com URL on success, None on failure.
        The file is committed to the experiment branch so it persists.
        """
        import base64

        try:
            content = plot.path.read_bytes()
            encoded = base64.b64encode(content).decode()

            upload_path = (
                f"experiments/{self._experiment_name}/plots/{plot.path.name}"
                if self._experiment_name
                else f"plots/{plot.path.name}"
            )
            api_url = f"{self._base_url}/repos/{self._repo}/contents/{upload_path}"

            client = await self._get_client()
            get_resp = await client.get(api_url, params={"ref": branch})
            sha = None
            if get_resp.status_code == 200:
                sha = get_resp.json().get("sha")

            payload: dict[str, Any] = {
                "message": f"watchdog: upload {plot.path.name}",
                "content": encoded,
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha

            resp = await client.put(api_url, json=payload)
            if resp.status_code in (200, 201):
                raw_url = (
                    f"https://raw.githubusercontent.com"
                    f"/{self._repo}/{branch}/{upload_path}"
                )
                _log.info(f"Plot uploaded: {raw_url}")
                return raw_url
            _log.warning(f"Plot upload failed: {resp.status_code}")
            return None
        except Exception as exc:
            _log.error(f"Plot upload error: {exc}")
            return None

    @staticmethod
    def _cache_bust_url(url: str, timestamp: int) -> str:
        """Append cache-busting query parameter to a URL."""
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}v={timestamp}"

    def _get_rolling_comment_id(self) -> int | None:
        """Read rolling comment ID from Redis."""
        if self._rolling_redis is None:
            return self._comment_id
        try:
            key = f"experiments:{self._experiment_name}:rolling_comment_id"
            raw = self._rolling_redis.get(key)
            return int(raw) if raw else None
        except Exception as exc:
            _log.warning(f"Redis rolling comment read failed: {exc}")
            return self._comment_id

    def _set_rolling_comment_id(self, comment_id: int) -> None:
        """Write rolling comment ID to Redis for persistence across restarts."""
        if self._rolling_redis is None:
            return
        try:
            key = f"experiments:{self._experiment_name}:rolling_comment_id"
            self._rolling_redis.set(key, str(comment_id))
        except Exception as exc:
            _log.warning(f"Redis rolling comment write failed: {exc}")

    async def send_status(self, update: StatusUpdate) -> bool:
        """Post or edit the rolling PR comment with status table + alerts + plots.

        Rolling comment strategy: create new comments for the first
        ``rolling_comment_threshold_hours`` cycles (approx 1 cycle ≈ 1 hour),
        then switch to editing the last comment in-place.
        """
        # Upload plots first and collect URLs
        plot_urls: dict[int, str] = {}
        if update.has_plots and self._branch:
            for i, plot in enumerate(update.plots):
                url = await self._upload_plot(plot, self._branch)
                if url:
                    ts_bust = int(update.timestamp.timestamp())
                    plot_urls[i] = self._cache_bust_url(url, ts_bust)

        body = self._build_status_body(update, plot_urls=plot_urls)

        self._status_count += 1
        past_threshold = self._status_count > self._rolling_threshold_hours

        if past_threshold:
            comment_id = self._get_rolling_comment_id() or self._comment_id
            if comment_id:
                success = await self._edit_comment(comment_id, body)
                if success:
                    return True
                _log.info(f"Rolling comment {comment_id} edit failed, creating new")

        new_id = await self._post_comment(body)
        if new_id is not None:
            self._comment_id = new_id
            if self._status_count == self._rolling_threshold_hours:
                self._set_rolling_comment_id(new_id)
                _log.info(f"Rolling comment set: ID={new_id}")
            return True
        return False

    async def send_alert(self, alert: Alert) -> bool:
        """Post a new PR comment for an alert (never edits rolling comment)."""
        body = format_alert_message(alert)
        new_id = await self._post_comment(body)
        return new_id is not None
