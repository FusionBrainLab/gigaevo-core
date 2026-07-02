"""Engine-facing memory provider adapters, injected via Hydra.

The provider is the read-side strategy object the DAG pipeline consumes
(``MemoryContextStage``). The ``memory={none,reader,writer,full,static}``
config arms swap the ``_target_``:

- ``NullMemoryProvider`` — no-op, returns empty selection (read side off)
- ``ReaderMemoryProvider`` — wraps the :class:`MemoryReader` facade
- ``StaticLeverMemoryProvider`` — fixed curated lever blocks, no bank
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re

from loguru import logger

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.read.exclusion import CardExcluder, NullExcluder
from gigaevo.memory.read.reader import MemoryReader, MemorySelection
from gigaevo.programs.program import Program


class MemoryProvider(ABC):
    """Abstract memory provider injected via Hydra."""

    @abstractmethod
    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        """Select memory cards relevant to this program."""


class NullMemoryProvider(MemoryProvider):
    """No-op provider. Returns empty selection. Default when ``memory=none``."""

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        return MemorySelection()


class ReaderMemoryProvider(MemoryProvider):
    """Adapts the :class:`MemoryReader` facade to the provider contract.

    The excluder prunes lineage-applied card ids from the research pass before
    the reader ranks candidates (filter-first lineage gate); ``NullExcluder``
    is the un-gated default.
    """

    def __init__(
        self,
        *,
        reader: MemoryReader,
        excluder: CardExcluder | None = None,
    ) -> None:
        self._reader = reader
        self._excluder = excluder if excluder is not None else NullExcluder()

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        return await self._reader.select(
            parents=[program],
            mutation_mode="rewrite",
            task_description=task_description,
            metrics_description=metrics_description,
            exclude_ids=self._excluder.exclude_for(program),
            parent_contexts=[parent_context] if parent_context is not None else None,
        )


class StaticLeverMemoryProvider(MemoryProvider):
    """Fixed lever-block provider for static-injection baselines (``memory=static``).

    Loads a curated levers file once and returns the same selection for every
    child: one card per ``---``-separated block, ids ``static:<stem>:<n>`` so
    gain-event stamping still attributes children to blocks post-hoc. A
    missing, empty, or wrong-block-count file fails the build — a levers-file
    typo must surface at launch, not silently run a degraded arm.
    """

    def __init__(
        self,
        *,
        levers_file: str | Path,
        expected_blocks: int | None = None,
    ) -> None:
        path = Path(levers_file)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise MemoryStorageError(f"levers file unreadable: {path}") from exc
        blocks = [b.strip() for b in re.split(r"^---\s*$", text, flags=re.M)]
        blocks = [b for b in blocks if b]
        if not blocks:
            raise MemoryStorageError(f"levers file has no lever blocks: {path}")
        # A mangled separator silently shrinks the treatment to fewer, merged
        # cards — the one failure mode that poisons an A/B arm undetected.
        if expected_blocks is not None and len(blocks) != expected_blocks:
            raise MemoryStorageError(
                f"levers file {path} parsed into {len(blocks)} blocks, "
                f"expected {expected_blocks}"
            )
        self._selection = MemorySelection(
            cards=tuple(blocks),
            card_ids=tuple(
                f"static:{path.stem}:{n}" for n in range(1, len(blocks) + 1)
            ),
        )
        logger.info(
            "[Memory][Static] serving {} lever blocks from {}", len(blocks), path
        )

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        return self._selection
