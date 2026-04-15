"""WatchdogPlugin ABC, plugin registry, and resolution.

Plugins control ONLY plot generation and status formatting.
Everything else (loop, heartbeat, Redis, notifications) is the engine.

Registry is a simple dict with @register decorator.
resolve_plugin() priority: manifest.watchdog.plugin > task heuristic > "solo" fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger

from gigaevo.experiment.manifest import ExperimentManifest
from gigaevo.monitoring.notifications import PlotAttachment
from gigaevo.monitoring.snapshot import RunSnapshot

_log = logger.bind(component="watchdog_plugin")


# ── Plugin ABC ───────────────────────────────────────────────────────────────


class WatchdogPlugin(ABC):
    """Abstract base for experiment-type watchdog plugins.

    Subclasses MUST implement:
      - generate_plots(): create plot files for this experiment type
      - format_status_body(): render status as markdown string

    Subclasses MAY override:
      - extra_telegram_content(): additional Telegram-specific content
      - extra_redis_queries(): additional Redis queries beyond snapshots
    """

    @abstractmethod
    def generate_plots(
        self,
        snapshots: list[RunSnapshot],
        output_dir: Path,
        cycle: int,
    ) -> list[PlotAttachment]:
        """Generate plot files for this cycle.

        Args:
            snapshots: Current RunSnapshot for each monitored run.
            output_dir: Directory to write plot PNGs into.
            cycle: Monotonically increasing cycle counter.

        Returns:
            List of PlotAttachment with paths to generated files.
        """
        ...

    @abstractmethod
    def format_status_body(
        self,
        snapshots: list[RunSnapshot],
        experiment_name: str,
        cycle: int,
        max_generations: int | None,
    ) -> str:
        """Render status as a markdown string for PR comments.

        Args:
            snapshots: Current RunSnapshot for each monitored run.
            experiment_name: Human-readable experiment identifier.
            cycle: Monotonically increasing cycle counter.
            max_generations: Target generation count (None in run mode).

        Returns:
            Markdown string ready for PR comment body.
        """
        ...

    def format_telegram_body(
        self,
        snapshots: list[RunSnapshot],
        experiment_name: str,
        cycle: int,
        max_generations: int | None,
        baseline: float | None = None,
    ) -> str | None:
        """Optional: plugin-specific Telegram message body.

        Returns None by default (uses generic table formatting).
        Override to produce plugin-specific formatting (e.g., G/D-separated
        summaries with emoji flags for adversarial).
        """
        return None

    def extra_telegram_content(self, snapshots: list[RunSnapshot]) -> str | None:
        """Optional: additional content for Telegram messages.

        Returns None by default (no extra content). Override to add
        experiment-specific Telegram photos or formatted text.
        """
        return None

    def extra_redis_queries(self) -> dict[str, str]:
        """Optional: additional Redis queries the plugin needs.

        Returns empty dict by default. Override to specify extra
        Redis keys to fetch beyond the standard snapshot queries.
        Keys: descriptive name, Values: Redis key pattern.
        """
        return {}


# ── Plugin Registry ──────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[WatchdogPlugin]] = {}

# Task name -> plugin name mapping for heuristic resolution
_TASK_HEURISTIC: dict[str, str] = {
    "adversarial": "adversarial",
    "heilbron": "heilbron",
    "hover": "solo",
    "hotpotqa": "solo",
    "toy": "solo",
}


def register(name: str):
    """Decorator to register a WatchdogPlugin subclass.

    Usage:
        @register("solo")
        class SoloPlugin(WatchdogPlugin):
            ...

    Raises ValueError if name is already registered.
    """

    def decorator(cls: type[WatchdogPlugin]) -> type[WatchdogPlugin]:
        if name in _REGISTRY:
            raise ValueError(
                f"Plugin '{name}' already registered by {_REGISTRY[name].__name__}. "
                f"Cannot register {cls.__name__} under the same name."
            )
        _REGISTRY[name] = cls
        _log.debug(f"Registered watchdog plugin: {name} -> {cls.__name__}")
        return cls

    return decorator


def get_registry() -> dict[str, type[WatchdogPlugin]]:
    """Return a copy of the plugin registry."""
    return dict(_REGISTRY)


def resolve_plugin(manifest: ExperimentManifest | None) -> type[WatchdogPlugin]:
    """Resolve the correct WatchdogPlugin class for an experiment.

    Priority:
      1. manifest.watchdog.plugin (explicit override)
      2. Task-prefix heuristic from manifest.experiment.task
      3. "solo" fallback

    Args:
        manifest: Validated ExperimentManifest, or None when running without one.

    Returns:
        The WatchdogPlugin subclass (not an instance).

    Raises:
        KeyError: If explicit plugin name is not in the registry.
    """
    if manifest is not None:
        # 1. Explicit plugin field
        explicit = manifest.control_plane.watchdog.plugin
        if explicit:
            if explicit not in _REGISTRY:
                raise KeyError(
                    f"Watchdog plugin '{explicit}' not found in registry. "
                    f"Available: {sorted(_REGISTRY.keys())}"
                )
            _log.info(
                f"Resolved plugin from manifest.control_plane.watchdog.plugin: {explicit}"
            )
            return _REGISTRY[explicit]

        # 2. Task-prefix heuristic
        task = manifest.contract.identity.task
        heuristic_name = _TASK_HEURISTIC.get(task)
        if heuristic_name and heuristic_name in _REGISTRY:
            _log.info(
                f"Resolved plugin from task heuristic: {task} -> {heuristic_name}"
            )
            return _REGISTRY[heuristic_name]

    # 3. Fallback to "solo"
    if "solo" not in _REGISTRY:
        raise KeyError(
            "No 'solo' plugin registered as fallback. "
            "Import gigaevo.monitoring.plugins.solo to register it."
        )
    _log.info("Resolved plugin: fallback to 'solo'")
    return _REGISTRY["solo"]
