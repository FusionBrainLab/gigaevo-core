"""One-line startup banner naming the memory components a run actually wired.

The three paper arms differ only in Hydra group overrides
(``pipeline``/``ideas_tracker``/``memory``), and a missed co-override
silently degrades to a ``Null*`` component. The banner puts the resolved
classes in the first log lines so the arm is verifiable at launch instead of
via ``.hydra/config.yaml`` archaeology after the run.
"""

from __future__ import annotations

from loguru import logger


def log_memory_arm_banner(
    *,
    provider: object | None,
    tracker: object | None,
    post_step_hook: object | None,
    pipeline_builder: object | None,
) -> None:
    """Log the resolved memory arm as class names (``None`` for absent)."""

    def _name(obj: object | None) -> str:
        return "None" if obj is None else type(obj).__name__

    logger.info(
        "[Memory][Arm] provider={} tracker={} post_step_hook={} pipeline_builder={}",
        _name(provider),
        _name(tracker),
        _name(post_step_hook),
        _name(pipeline_builder),
    )
