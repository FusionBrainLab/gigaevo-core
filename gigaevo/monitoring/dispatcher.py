"""Notification fan-out dispatcher.

Sends StatusUpdate and Alert objects to all registered channels
concurrently. Implements cross-channel failure escalation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from gigaevo.monitoring.notifications import NotificationChannel, StatusUpdate

_log = logger.bind(component="dispatcher")


@dataclass(frozen=True)
class DispatchResult:
    """Result of dispatching a notification to all channels.

    Attributes:
        channel_results: Map of channel class name to success/failure.
        alerts_sent: Number of alerts successfully sent to at least one channel.
        alerts_suppressed: Number of alert deliveries that failed.
    """

    channel_results: dict[str, bool] = field(default_factory=dict)
    alerts_sent: int = 0
    alerts_suppressed: int = 0

    @property
    def all_succeeded(self) -> bool:
        return all(self.channel_results.values()) if self.channel_results else True

    @property
    def any_failed(self) -> bool:
        return any(not v for v in self.channel_results.values())


class NotificationDispatcher:
    """Fan-out dispatcher for notification channels.

    Sends the same StatusUpdate to every registered channel concurrently.
    Implements cross-channel failure escalation: if TelegramChannel fails
    consecutively, GitHubPRChannel gets a telegram_down header.
    """

    def __init__(self, channels: list[NotificationChannel]) -> None:
        self._channels = list(channels)

    @property
    def channels(self) -> list[NotificationChannel]:
        return list(self._channels)

    async def dispatch(self, update: StatusUpdate) -> DispatchResult:
        """Send update to all channels concurrently.

        1. Send status to all channels via asyncio.gather
        2. Send each alert to all channels
        3. Check for cross-channel failure escalation
        4. Return per-channel results
        """
        if not self._channels:
            return DispatchResult()

        # 1. Fan-out send_status to all channels concurrently
        status_tasks = [ch.send_status(update) for ch in self._channels]
        status_results = await asyncio.gather(*status_tasks, return_exceptions=True)

        channel_results: dict[str, bool] = {}
        for ch, result in zip(self._channels, status_results):
            name = type(ch).__name__
            if isinstance(result, BaseException):
                _log.error(f"Channel {name} raised: {result}")
                channel_results[name] = False
            else:
                channel_results[name] = bool(result)

        # 2. Send alerts to all channels
        alerts_sent = 0
        alerts_suppressed = 0
        for alert in update.alerts:
            alert_tasks = [ch.send_alert(alert) for ch in self._channels]
            alert_results = await asyncio.gather(*alert_tasks, return_exceptions=True)

            sent_to_any = False
            for ch, result in zip(self._channels, alert_results):
                name = type(ch).__name__
                if isinstance(result, BaseException):
                    _log.error(f"Alert delivery to {name} raised: {result}")
                    alerts_suppressed += 1
                elif result:
                    sent_to_any = True
                else:
                    alerts_suppressed += 1

            if sent_to_any:
                alerts_sent += 1

        # 3. Cross-channel failure escalation
        self._check_escalation()

        return DispatchResult(
            channel_results=channel_results,
            alerts_sent=alerts_sent,
            alerts_suppressed=alerts_suppressed,
        )

    def _check_escalation(self) -> None:
        """Check if Telegram failures should trigger PR channel escalation."""
        from gigaevo.monitoring.github_pr_channel import GitHubPRChannel
        from gigaevo.monitoring.telegram_channel import TelegramChannel

        telegram_down = False
        for ch in self._channels:
            if isinstance(ch, TelegramChannel):
                if (
                    ch.consecutive_failures
                    >= TelegramChannel.CONSECUTIVE_FAILURE_THRESHOLD
                ):
                    telegram_down = True
                    _log.warning(
                        f"Telegram consecutive failures: {ch.consecutive_failures} "
                        f">= threshold {TelegramChannel.CONSECUTIVE_FAILURE_THRESHOLD}"
                    )
                break

        for ch in self._channels:
            if isinstance(ch, GitHubPRChannel):
                if ch.telegram_down != telegram_down:
                    ch.telegram_down = telegram_down
                    _log.info(f"GitHubPRChannel.telegram_down = {telegram_down}")
