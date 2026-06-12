from __future__ import annotations

from datetime import UTC, datetime
import json
from math import ceil
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.core.protocols import Deduplicator, Evictor
from gigaevo.memory.efficacy import CardStatsStamper
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
    ConnectedIdea,
    MemoryCard,
    ProgramCard,
)
from gigaevo.memory.utils import to_float

_MAX_CONNECTED_DESCRIPTIONS = 5


class CardMemory(Protocol):
    def get_card(self, card_id: str) -> AnyCard | None: ...


def load_json(path: Path) -> Any:
    """Read a JSON document, failing loudly when the file is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Cards file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_latest_snapshot(payload: Any, required_key: str) -> dict[str, Any]:
    """Return the newest snapshot dict containing ``required_key``.

    Tracker logs are append-only lists of snapshots; a bare dict is accepted as
    a single-snapshot document.
    """
    if isinstance(payload, dict):
        if required_key in payload:
            return payload
        raise ValueError(f"Missing key '{required_key}' in snapshot payload")

    if isinstance(payload, list):
        snapshots = [
            item for item in payload if isinstance(item, dict) and required_key in item
        ]
        if snapshots:
            return snapshots[-1]
        raise ValueError(f"No snapshot with key '{required_key}' found in payload list")

    raise ValueError(
        "Invalid snapshot JSON format. Expected a dict or list of dict snapshots"
    )


def top_percent_count(total: int, percent: float) -> int:
    """How many items a top-``percent`` slice of ``total`` selects (>=1 when any qualify)."""
    if total <= 0 or percent <= 0:
        return 0
    return max(1, ceil(total * (percent / 100.0)))


def classify_card_type(card: AnyCard) -> str:
    """Bucket a typed card for write-stats reporting: ``programs`` or ``ideas``."""
    return "programs" if isinstance(card, ProgramCard) else "ideas"


class BankSnapshot(BaseModel):
    """Latest banks.json snapshot: the active idea bank."""

    model_config = ConfigDict(extra="ignore")

    active_bank: list[Any] = Field(
        description="Raw card payloads of the active idea bank."
    )


class BestIdeasSnapshot(BaseModel):
    """Latest best_ideas.json snapshot: ranked idea metric rows."""

    model_config = ConfigDict(extra="ignore")

    best_ideas: list[Any] = Field(
        description="IdeaStats.as_json_row() dumps, best ideas first."
    )


class ProgramsSnapshot(BaseModel):
    """Latest programs.json snapshot: exported program rows."""

    model_config = ConfigDict(extra="ignore")

    programs: list[Any] = Field(description="Raw exported program rows.")


class ProgramRow(BaseModel):
    """One row of programs.json's latest snapshot."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="Program id (ideas_tracker export key).")
    program_id: str = Field(
        default="", description="Program id (raw Redis export key)."
    )
    fitness: float | None = Field(
        default=None, description="Program fitness; None when absent or unparsable."
    )
    is_valid: float | None = Field(
        default=None,
        description="Validity flag from the raw export; None when pre-filtered.",
    )
    task_description: str = Field(
        default="", description="Task description active for the run."
    )
    task_description_summary: str = Field(
        default="", description="Condensed form of the task description."
    )
    code: str = Field(default="", description="Program source code.")

    @field_validator(
        "id",
        "program_id",
        "task_description",
        "task_description_summary",
        "code",
        mode="before",
    )
    @classmethod
    def coerce_text(cls, value: Any) -> str:
        return str(value or "")

    @field_validator("fitness", "is_valid", mode="before")
    @classmethod
    def coerce_float(cls, value: Any) -> float | None:
        return to_float(value, default=None)

    @property
    def resolved_program_id(self) -> str:
        return (self.id or self.program_id).strip()

    @property
    def passes_validity_gate(self) -> bool:
        """Absent is_valid means ideas_tracker already pre-filtered (valid-only);
        explicit 0 comes from the raw Redis export path."""
        return self.is_valid is None or self.is_valid > 0


class WriteStats(BaseModel):
    """Card-store write counters as reported by the backend's write ledger."""

    model_config = ConfigDict(extra="ignore")

    processed: int = Field(default=0, description="Cards submitted to the store.")
    added: int = Field(default=0, description="Cards added as new bank entries.")
    updated: int = Field(default=0, description="Cards that updated an existing entry.")
    rejected: int = Field(default=0, description="Cards rejected by a write gate.")
    updated_target_cards: int = Field(
        default=0, description="Existing cards modified as merge targets."
    )

    def delta_to(self, after: WriteStats) -> WriteStats:
        """Non-negative per-counter change from this snapshot to ``after``."""
        return WriteStats(
            processed=max(0, after.processed - self.processed),
            added=max(0, after.added - self.added),
            updated=max(0, after.updated - self.updated),
            rejected=max(0, after.rejected - self.rejected),
            updated_target_cards=max(
                0, after.updated_target_cards - self.updated_target_cards
            ),
        )

    def accumulate(self, delta: WriteStats) -> WriteStats:
        """New counters with ``delta`` added on top of this snapshot."""
        return WriteStats(
            processed=self.processed + delta.processed,
            added=self.added + delta.added,
            updated=self.updated + delta.updated,
            rejected=self.rejected + delta.rejected,
            updated_target_cards=self.updated_target_cards + delta.updated_target_cards,
        )


class CardTypeCounts(BaseModel):
    """Input card tally per write-stats bucket."""

    model_config = ConfigDict(extra="forbid")

    ideas: int = Field(default=0, description="Idea cards in the input batch.")
    programs: int = Field(default=0, description="Program cards in the input batch.")


class CardTypeWriteStats(BaseModel):
    """Write counters split per card-type bucket."""

    model_config = ConfigDict(extra="forbid")

    ideas: WriteStats = Field(
        default_factory=WriteStats, description="Write counters for idea cards."
    )
    programs: WriteStats = Field(
        default_factory=WriteStats, description="Write counters for program cards."
    )


class WriteStatsSnapshot(BaseModel):
    """One timestamped entry of memory_write_stats.json.

    Field names are the on-disk JSON keys — keep them stable.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str = Field(
        description="ISO-8601 UTC time the snapshot was recorded."
    )
    input_cards_count: int = Field(description="Cards in the input batch.")
    input_classify_card_type_counts: CardTypeCounts = Field(
        description="Input batch tally split by card type."
    )
    stats: WriteStats = Field(description="Combined write counters for the batch.")
    stats_by_classify_card_type: CardTypeWriteStats = Field(
        description="Write counters split by card type."
    )


def parse_best_ideas(path: Path) -> tuple[list[str], dict[str, IdeaStats]]:
    """Read best_ideas.json into (ordered unique idea ids, per-id stats rows).

    Rows are the ``IdeaStats.as_json_row()`` dumps the tracker writes, so they
    validate back into :class:`IdeaStats` directly.
    """
    payload = load_json(path)
    try:
        snapshot = BestIdeasSnapshot.model_validate(
            extract_latest_snapshot(payload, "best_ideas")
        )
    except ValidationError as exc:
        raise ValueError(
            f"Invalid best ideas format in {path}: expected list under 'best_ideas'"
        ) from exc

    idea_ids: list[str] = []
    best_by_id: dict[str, IdeaStats] = {}
    for item in snapshot.best_ideas:
        if not isinstance(item, dict):
            continue
        row = IdeaStats.model_validate(item)
        idea_id = row.idea_id.strip()
        if not idea_id or idea_id in best_by_id:
            continue
        idea_ids.append(idea_id)
        best_by_id[idea_id] = row

    return idea_ids, best_by_id


def parse_programs(path: Path) -> list[ProgramRow]:
    """Read programs.json's latest snapshot into typed program rows."""
    payload = load_json(path)
    try:
        snapshot = ProgramsSnapshot.model_validate(
            extract_latest_snapshot(payload, "programs")
        )
    except ValidationError as exc:
        raise ValueError(
            f"Invalid programs format in {path}: expected list under 'programs'"
        ) from exc
    return [
        ProgramRow.model_validate(program)
        for program in snapshot.programs
        if isinstance(program, dict)
    ]


def fold_best_idea_metrics(card: AnyCard, row: IdeaStats) -> AnyCard:
    """Fold a best_ideas stats row into a typed bank card.

    Fills a missing description from the row and stamps the row's metric
    vocabulary under ``evolution_statistics.best_ideas_snapshot``. The input
    card is never mutated.
    """
    updates: dict[str, Any] = {
        "evolution_statistics": card.evolution_statistics.model_copy(
            update={"best_ideas_snapshot": row.to_stats_block()}
        )
    }
    if not card.description:
        updates["description"] = row.description

    return card.model_copy(update=updates)


def load_best_idea_bank_cards(path: Path, best_ideas_path: Path) -> list[AnyCard]:
    """Select the banks.json cards named by best_ideas.json, metrics folded in."""
    if not best_ideas_path.exists():
        raise FileNotFoundError(f"Best ideas file not found: {best_ideas_path}")

    payload = load_json(path)
    try:
        snapshot = BankSnapshot.model_validate(
            extract_latest_snapshot(payload, "active_bank")
        )
    except ValidationError as exc:
        raise ValueError(
            f"Invalid banks format in {path}: expected 'active_bank' list"
        ) from exc

    all_cards = [
        normalize_memory_card(entry)
        for entry in snapshot.active_bank
        if isinstance(entry, dict)
    ]
    cards_by_id = {card.id.strip(): card for card in all_cards if card.id.strip()}
    best_idea_ids, best_by_id = parse_best_ideas(best_ideas_path)

    selected_cards: list[AnyCard] = []
    missing_cards: list[str] = []
    for idea_id in best_idea_ids:
        bank_card = cards_by_id.get(idea_id)
        if bank_card is None:
            missing_cards.append(idea_id)
            continue
        row = best_by_id.get(idea_id)
        selected_cards.append(
            fold_best_idea_metrics(bank_card, row) if row is not None else bank_card
        )

    if missing_cards:
        logger.warning(
            "[Memory][WritePipeline] {} best_ideas IDs were missing in banks and were skipped.",
            len(missing_cards),
        )

    return selected_cards


def load_latest_bank_cards(path: Path) -> list[AnyCard]:
    """Read the newest ``active_bank`` snapshot from banks.json as typed cards."""
    payload = load_json(path)
    try:
        snapshot = BankSnapshot.model_validate(
            extract_latest_snapshot(payload, "active_bank")
        )
    except ValidationError as exc:
        raise ValueError(
            f"Invalid banks format in {path}: expected 'active_bank' list"
        ) from exc
    return [
        normalize_memory_card(entry)
        for entry in snapshot.active_bank
        if isinstance(entry, dict)
    ]


def build_program_cards_from_top_programs(
    *,
    programs_path: Path | None,
    banks_path: Path,
    best_programs_percent: float,
    higher_is_better: bool = True,
    card_posterior: dict[str, CardStatsBlock] | None = None,
) -> list[ProgramCard]:
    """Turn the top-fitness slice of programs.json into typed program cards.

    Each card links back to the bank ideas that produced the program; cards
    that already carry an injection posterior are kept even when they drop out
    of the top slice, so their efficacy signal still reaches the auction.
    """
    if (
        programs_path is None
        or not programs_path.exists()
        or best_programs_percent <= 0
    ):
        return []

    rows = parse_programs(programs_path)
    eligible_rows = [
        row
        for row in rows
        if row.resolved_program_id
        and row.fitness is not None
        and row.passes_validity_gate
    ]
    if not eligible_rows:
        return []

    # When higher_is_better=True the best programs sit at the top of a
    # descending sort; for lower-is-better tasks (vartodd_ham_high, loss-style
    # metrics) the best programs are the LOWEST fitness, so we flip the sort.
    eligible_rows.sort(
        key=lambda row: (row.fitness, row.resolved_program_id),
        reverse=higher_is_better,
    )
    selected_count = top_percent_count(len(eligible_rows), best_programs_percent)
    selected_rows = eligible_rows[:selected_count]

    # A card accrues an injection posterior only after it is injected downstream,
    # by which point it has usually dropped out of the top-fitness slice; build it
    # anyway so its efficacy signal reaches the auction instead of being stranded.
    if card_posterior:
        selected_rows = selected_rows + [
            row
            for row in eligible_rows[selected_count:]
            if f"program-{row.resolved_program_id}" in card_posterior
        ]

    connected_ideas_by_program: dict[str, list[ConnectedIdea]] = {}
    for idea_card in load_latest_bank_cards(banks_path):
        if not isinstance(idea_card, MemoryCard):
            continue
        idea_id = idea_card.id.strip()
        if not idea_id:
            continue
        linked_idea = ConnectedIdea(
            idea_id=idea_id, description=idea_card.description.strip()
        )
        for raw_program_id in idea_card.programs:
            linked_program_id = str(raw_program_id or "").strip()
            if not linked_program_id:
                continue
            connected_ideas_by_program.setdefault(linked_program_id, []).append(
                linked_idea
            )

    cards: list[ProgramCard] = []
    for rank, row in enumerate(selected_rows, start=1):
        program_id = row.resolved_program_id
        connected_ideas = connected_ideas_by_program.get(program_id, [])
        connected_descriptions = [
            idea.description.strip()
            for idea in connected_ideas
            if idea.description.strip()
        ]
        connected_summary = "; ".join(
            connected_descriptions[:_MAX_CONNECTED_DESCRIPTIONS]
        ).strip()
        if connected_summary:
            description = connected_summary
            keywords = [f"program_rank:{rank}"]
        else:
            description = ""
            keywords = ["pending_analysis:true", f"program_rank:{rank}"]

        cards.append(
            ProgramCard(
                id=f"program-{program_id}",
                program_id=program_id,
                task_description=row.task_description.strip(),
                task_description_summary=row.task_description_summary.strip(),
                description=description,
                fitness=row.fitness,
                code=row.code,
                connected_ideas=connected_ideas,
                keywords=keywords,
            )
        )

    return cards


def load_memory_cards(
    path: Path,
    best_ideas_path: Path,
    *,
    programs_path: Path | None = None,
    best_programs_percent: float = 0.0,
    higher_is_better: bool = True,
    card_posterior: dict[str, CardStatsBlock] | None = None,
) -> list[AnyCard]:
    """Load idea and program cards from banks as typed cards, posteriors stamped.

    Raw JSON dicts cross into typed models inside the loaders called here;
    everything downstream operates on typed cards only.
    """
    payload = load_json(path)

    if isinstance(payload, dict) and "active_bank" in payload:
        idea_cards = load_best_idea_bank_cards(path, best_ideas_path)
    elif (
        isinstance(payload, list)
        and payload
        and isinstance(payload[0], dict)
        and "active_bank" in payload[0]
    ):
        idea_cards = load_best_idea_bank_cards(path, best_ideas_path)
    else:
        raise ValueError(
            "Invalid banks JSON format. Expected payload with 'active_bank'"
        )

    typed_cards: list[AnyCard] = [
        *idea_cards,
        *build_program_cards_from_top_programs(
            programs_path=programs_path,
            banks_path=path,
            best_programs_percent=best_programs_percent,
            higher_is_better=higher_is_better,
            card_posterior=card_posterior,
        ),
    ]
    if card_posterior:
        stamper = CardStatsStamper()
        typed_cards = [stamper.stamp_posterior(c, card_posterior) for c in typed_cards]
    return typed_cards


def append_write_stats_snapshot(
    *,
    stats_path: Path,
    input_cards_count: int,
    input_card_type_counts: CardTypeCounts,
    write_stats: WriteStats,
    write_stats_by_card_type: CardTypeWriteStats,
) -> WriteStatsSnapshot:
    """Append one timestamped write-stats snapshot to memory_write_stats.json."""
    snapshot = WriteStatsSnapshot(
        timestamp_utc=datetime.now(UTC).isoformat(),
        input_cards_count=int(input_cards_count),
        input_classify_card_type_counts=input_card_type_counts,
        stats=write_stats,
        stats_by_classify_card_type=write_stats_by_card_type,
    )

    existing: list[dict[str, Any]] = []
    if stats_path.exists():
        try:
            raw = load_json(stats_path)
            if isinstance(raw, list):
                existing = [item for item in raw if isinstance(item, dict)]
            elif isinstance(raw, dict):
                existing = [raw]
        except Exception as exc:
            logger.warning(
                "[Memory][WritePipeline] Failed to load existing write stats from {}: {}",
                stats_path,
                exc,
            )
            existing = []

    existing.append(snapshot.model_dump())
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=True, indent=2)
    return snapshot


def main(
    *,
    banks_path: Path,
    best_ideas_path: Path,
    programs_path: Path | None = None,
    backend: MemoryBackendFactory,
    checkpoint_dir: str | Path | None = None,
    best_programs_percent: float = 5.0,
    higher_is_better: bool = True,
    card_posterior: dict[str, CardStatsBlock] | None = None,
    evictor: Evictor | None = None,
    deduplicator: Deduplicator | None = None,
) -> WriteStatsSnapshot | None:
    """Load cards from banks, write them into the card bank, report stats.

    The bank is constructed via ``backend``, a required Hydra-composed
    :class:`MemoryBackendFactory` (``memory/backend`` group) — there is no
    implicit default. ``checkpoint_dir`` pins per-run memory artefacts under
    the Hydra output directory. ``evictor``/``deduplicator`` are the shared
    write-side components from the ``memory/evictor`` and ``memory/dedup``
    groups; ``None`` keeps the backend's built-in defaults.
    """
    memory = backend.build(
        checkpoint_dir=checkpoint_dir, evictor=evictor, deduplicator=deduplicator
    )

    try:
        if not banks_path.exists():
            raise FileNotFoundError(f"Banks file not found: {banks_path}")
        memory_cards = load_memory_cards(
            banks_path,
            best_ideas_path=best_ideas_path,
            programs_path=programs_path,
            best_programs_percent=best_programs_percent,
            higher_is_better=higher_is_better,
            card_posterior=card_posterior,
        )
        logger.info(
            "[Memory][WritePipeline] Loaded {} cards from banks: {} (filtered by: {})",
            len(memory_cards),
            banks_path,
            best_ideas_path,
        )

        idea_write_stats = WriteStats()
        program_write_stats = WriteStats()
        try:
            for idx, card in enumerate(memory_cards, start=1):
                before_stats = WriteStats.model_validate(memory.get_card_write_stats())
                memory_id = memory.save_card(card)
                after_stats = WriteStats.model_validate(memory.get_card_write_stats())
                stat_delta = before_stats.delta_to(after_stats)
                if isinstance(card, ProgramCard):
                    program_write_stats = program_write_stats.accumulate(stat_delta)
                else:
                    idea_write_stats = idea_write_stats.accumulate(stat_delta)
                stored = memory.get_card(memory_id)
                # memory_id is the ingest verdict's final id — possibly an existing
                # duplicate, not a fresh save; the write ledger holds the verdict.
                logger.debug(
                    "[Memory][WritePipeline] [{:03d}] ingested → {}: {}",
                    idx,
                    memory_id,
                    (stored.description if stored is not None else "")[:110],
                )
        except (RuntimeError, MemoryStorageError) as exc:
            logger.error("[Memory][WritePipeline] Write failed: {}", exc)
            return None

        evicted = memory.sweep_harmful()
        if evicted:
            logger.info(
                "[Memory][WritePipeline] Harm sweep evicted {} card(s): {}",
                len(evicted),
                evicted,
            )

        memory.rebuild()

        write_stats = WriteStats.model_validate(memory.get_card_write_stats())
        write_stats_by_card_type = CardTypeWriteStats(
            ideas=idea_write_stats, programs=program_write_stats
        )
        input_card_type_counts = CardTypeCounts(
            ideas=sum(1 for card in memory_cards if not isinstance(card, ProgramCard)),
            programs=sum(1 for card in memory_cards if isinstance(card, ProgramCard)),
        )
        logger.info(
            "[Memory][WritePipeline] Write stats: processed={} added={} updated={} rejected={} "
            "ideas(proc={} add={} upd={} rej={}) "
            "programs(proc={} add={} upd={} rej={}) updated_target_cards={}",
            write_stats.processed,
            write_stats.added,
            write_stats.updated,
            write_stats.rejected,
            idea_write_stats.processed,
            idea_write_stats.added,
            idea_write_stats.updated,
            idea_write_stats.rejected,
            program_write_stats.processed,
            program_write_stats.added,
            program_write_stats.updated,
            program_write_stats.rejected,
            write_stats.updated_target_cards,
        )

        stats_path = banks_path.parent / "memory_write_stats.json"
        snapshot = append_write_stats_snapshot(
            stats_path=stats_path,
            input_cards_count=len(memory_cards),
            input_card_type_counts=input_card_type_counts,
            write_stats=write_stats,
            write_stats_by_card_type=write_stats_by_card_type,
        )
        logger.info(
            "[Memory][WritePipeline] Memory write stats saved to: {}", stats_path
        )
        return snapshot
    finally:
        memory.close()
