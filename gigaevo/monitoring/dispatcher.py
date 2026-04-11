"""Notification fan-out dispatcher.

Sends StatusUpdate and Alert objects to all registered channels
concurrently. Implements cross-channel failure escalation.
"""

from __future__ import annotations

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
        raise NotImplementedError  # placeholder until Task 4
