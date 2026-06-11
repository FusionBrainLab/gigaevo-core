from __future__ import annotations

from datetime import UTC, datetime
import json
from math import ceil
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.core.protocols import Deduplicator, Evictor
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import AnyCard, ProgramCard
from gigaevo.memory.utils import to_float
from gigaevo.programs.metrics.context import VALIDITY_KEY

_MAX_CONNECTED_DESCRIPTIONS = 5


class CardMemory(Protocol):
    def get_card(self, card_id: str) -> AnyCard | None: ...


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Cards file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_latest_snapshot(payload: Any, required_key: str) -> dict[str, Any]:
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


def _top_percent_count(total: int, percent: float) -> int:
    if total <= 0 or percent <= 0:
        return 0
    return max(1, ceil(total * (percent / 100.0)))


def _classify_card_type(card: dict[str, Any] | AnyCard) -> str:
    if isinstance(card, ProgramCard):
        return "programs"
    if isinstance(card, dict):
        if str(card.get("category") or "").strip().lower() == "program":
            return "programs"
        if str(card.get("program_id") or "").strip():
            return "programs"
    return "ideas"


def _zero_write_stats() -> dict[str, int]:
    return {
        "processed": 0,
        "added": 0,
        "updated": 0,
        "rejected": 0,
        "updated_target_cards": 0,
    }


def _diff_write_stats(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    keys = set(before) | set(after)
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in keys
    }


def _parse_best_ideas(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    payload = _load_json(path)
    snapshot = _extract_latest_snapshot(payload, "best_ideas")
    best_ideas = snapshot.get("best_ideas")
    if not isinstance(best_ideas, list):
        raise ValueError(
            f"Invalid best ideas format in {path}: expected list under 'best_ideas'"
        )

    idea_ids: list[str] = []
    best_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for item in best_ideas:
        if not isinstance(item, dict):
            continue
        idea_id = str(item.get("idea_id") or item.get("id") or "").strip()
        if not idea_id or idea_id in seen_ids:
            continue
        seen_ids.add(idea_id)
        idea_ids.append(idea_id)
        best_by_id[idea_id] = item

    return idea_ids, best_by_id


def _parse_programs(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    snapshot = _extract_latest_snapshot(payload, "programs")
    programs = snapshot.get("programs")
    if not isinstance(programs, list):
        raise ValueError(
            f"Invalid programs format in {path}: expected list under 'programs'"
        )
    return [program for program in programs if isinstance(program, dict)]


def _merge_best_idea_metrics_into_card(
    card: dict[str, Any], best_entry: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(card)
    if not merged.get("description"):
        merged["description"] = str(best_entry.get("description") or "")

    best_metrics = {
        key: value
        for key, value in best_entry.items()
        if key not in {"idea_id", "id", "description"}
    }
    if best_metrics:
        evolution_stats = merged.get("evolution_statistics")
        if not isinstance(evolution_stats, dict):
            evolution_stats = {}
        evolution_stats["best_ideas_snapshot"] = best_metrics
        merged["evolution_statistics"] = evolution_stats

    return merged


def _load_banks_cards(path: Path, best_ideas_path: Path) -> list[dict]:
    if not best_ideas_path.exists():
        raise FileNotFoundError(f"Best ideas file not found: {best_ideas_path}")

    payload = _load_json(path)
    snapshot = _extract_latest_snapshot(payload, "active_bank")
    active_bank = snapshot.get("active_bank")
    if not isinstance(active_bank, list):
        raise ValueError(f"Invalid banks format in {path}: expected 'active_bank' list")

    all_cards = [card for card in active_bank if isinstance(card, dict)]
    cards_by_id = {
        str(card.get("id")).strip(): card
        for card in all_cards
        if str(card.get("id") or "").strip()
    }
    best_idea_ids, best_by_id = _parse_best_ideas(best_ideas_path)

    selected_cards: list[dict] = []
    missing_cards: list[str] = []
    for idea_id in best_idea_ids:
        bank_card = cards_by_id.get(idea_id)
        best_entry = best_by_id.get(idea_id, {})
        if bank_card is None:
            missing_cards.append(idea_id)
            continue
        selected_cards.append(_merge_best_idea_metrics_into_card(bank_card, best_entry))

    if missing_cards:
        logger.warning(
            "[Memory][WritePipeline] {} best_ideas IDs were missing in banks and were skipped.",
            len(missing_cards),
        )

    return selected_cards


def _load_latest_bank_cards(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    snapshot = _extract_latest_snapshot(payload, "active_bank")
    active_bank = snapshot.get("active_bank")
    if not isinstance(active_bank, list):
        raise ValueError(f"Invalid banks format in {path}: expected 'active_bank' list")
    return [card for card in active_bank if isinstance(card, dict)]


def _build_program_cards_from_top_programs(
    *,
    programs_path: Path | None,
    banks_path: Path,
    best_programs_percent: float,
    higher_is_better: bool = True,
    card_posterior: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if (
        programs_path is None
        or not programs_path.exists()
        or best_programs_percent <= 0
    ):
        return []

    programs = _parse_programs(programs_path)
    if not programs:
        return []

    eligible_programs: list[dict[str, Any]] = []
    for program in programs:
        program_id = str(program.get("id") or program.get("program_id") or "").strip()
        fitness = to_float(program.get("fitness"))
        if not program_id or fitness is None:
            continue
        # Absent is_valid means ideas_tracker already pre-filtered (valid-only).
        # Only skip when is_valid is explicitly 0 (raw Redis export path).
        is_valid = to_float(program.get(VALIDITY_KEY))
        if is_valid is not None and is_valid <= 0:
            continue
        enriched = dict(program)
        enriched["program_id"] = program_id
        enriched["fitness"] = fitness
        eligible_programs.append(enriched)

    if not eligible_programs:
        return []

    # When higher_is_better=True the best programs sit at the top of a
    # descending sort; for lower-is-better tasks (vartodd_ham_high, loss-style
    # metrics) the best programs are the LOWEST fitness, so we flip the sort.
    eligible_programs.sort(
        key=lambda program: (
            float(program.get("fitness", 0.0)),
            str(program["program_id"]),
        ),
        reverse=higher_is_better,
    )
    selected_count = _top_percent_count(len(eligible_programs), best_programs_percent)
    selected_programs = eligible_programs[:selected_count]

    # A card accrues an injection posterior only after it is injected downstream,
    # by which point it has usually dropped out of the top-fitness slice; build it
    # anyway so its efficacy signal reaches the auction instead of being stranded.
    if card_posterior:
        selected_programs = selected_programs + [
            program
            for program in eligible_programs[selected_count:]
            if f"program-{program['program_id']}" in card_posterior
        ]

    connected_ideas_by_program: dict[str, list[dict[str, str]]] = {}
    for idea_card in _load_latest_bank_cards(banks_path):
        idea_id = str(idea_card.get("id") or "").strip()
        idea_description = str(idea_card.get("description") or "").strip()
        if not idea_id:
            continue
        programs_for_idea = idea_card.get("programs")
        if not isinstance(programs_for_idea, list):
            continue
        linked_idea = {
            "idea_id": idea_id,
            "description": idea_description,
        }
        for raw_program_id in programs_for_idea:
            linked_program_id = str(raw_program_id or "").strip()
            if not linked_program_id:
                continue
            connected_ideas_by_program.setdefault(linked_program_id, []).append(
                linked_idea
            )

    cards: list[dict[str, Any]] = []
    for rank, program in enumerate(selected_programs, start=1):
        program_id = str(program["program_id"])
        task_description = str(program.get("task_description") or "").strip()
        task_description_summary = str(
            program.get("task_description_summary") or ""
        ).strip()
        connected_ideas = connected_ideas_by_program.get(program_id, [])
        connected_descriptions = [
            str(item.get("description") or "").strip()
            for item in connected_ideas
            if isinstance(item, dict)
        ]
        connected_descriptions = [text for text in connected_descriptions if text]
        connected_summary = "; ".join(
            connected_descriptions[:_MAX_CONNECTED_DESCRIPTIONS]
        )
        connected_summary = connected_summary.strip()
        if connected_summary:
            description = connected_summary
            keywords = [f"program_rank:{rank}"]
        else:
            description = ""
            keywords = ["pending_analysis:true", f"program_rank:{rank}"]

        card_id = f"program-{program_id}"
        card: dict[str, Any] = {
            "id": card_id,
            "category": "program",
            "program_id": program_id,
            "task_description": task_description,
            "task_description_summary": task_description_summary,
            "description": description,
            "fitness": float(program["fitness"]),
            "code": str(program.get("code") or ""),
            "connected_ideas": connected_ideas,
            "keywords": keywords,
        }
        cards.append(card)

    return cards


def stamp_card_posterior(
    card: dict[str, Any], card_posterior: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Merge a card's injection posterior into ``evolution_statistics["ALL"]``.

    Applies to ANY card the selector can inject — idea and program cards alike —
    keyed by the card's own id. Existing statistics (e.g. the
    ``best_ideas_snapshot`` block) are preserved; cards without a posterior are
    returned unchanged so the auction treats them as COLD.
    """
    posterior = card_posterior.get(str(card.get("id") or "").strip())
    if not posterior:
        return card
    stats = card.get("evolution_statistics")
    merged = dict(stats) if isinstance(stats, dict) else {}
    merged["ALL"] = posterior
    stamped = dict(card)
    stamped["evolution_statistics"] = merged
    return stamped


def load_memory_cards(
    path: Path,
    best_ideas_path: Path,
    *,
    programs_path: Path | None = None,
    best_programs_percent: float = 0.0,
    higher_is_better: bool = True,
    card_posterior: dict[str, dict[str, Any]] | None = None,
) -> list:
    """Load idea and program cards from banks, apply filters, stamp posteriors."""
    payload = _load_json(path)

    cards: list[dict]
    if isinstance(payload, dict) and "active_bank" in payload:
        cards = _load_banks_cards(path, best_ideas_path)
    elif (
        isinstance(payload, list)
        and payload
        and isinstance(payload[0], dict)
        and "active_bank" in payload[0]
    ):
        cards = _load_banks_cards(path, best_ideas_path)
    else:
        raise ValueError(
            "Invalid banks JSON format. Expected payload with 'active_bank'"
        )

    all_cards = cards + _build_program_cards_from_top_programs(
        programs_path=programs_path,
        banks_path=path,
        best_programs_percent=best_programs_percent,
        higher_is_better=higher_is_better,
        card_posterior=card_posterior,
    )
    if card_posterior:
        all_cards = [stamp_card_posterior(c, card_posterior) for c in all_cards]
    return [normalize_memory_card(c) for c in all_cards]


def _write_memory_write_stats(
    *,
    stats_path: Path,
    input_cards_count: int,
    input_classify_card_type_counts: dict[str, int],
    write_stats: dict[str, int],
    write_stats_by_classify_card_type: dict[str, dict[str, int]],
) -> dict[str, Any]:
    snapshot = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "input_cards_count": int(input_cards_count),
        "input_classify_card_type_counts": input_classify_card_type_counts,
        "stats": write_stats,
        "stats_by_classify_card_type": write_stats_by_classify_card_type,
    }

    existing: list[dict[str, Any]] = []
    if stats_path.exists():
        try:
            raw = _load_json(stats_path)
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

    existing.append(snapshot)
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
    card_posterior: dict[str, dict[str, Any]] | None = None,
    evictor: Evictor | None = None,
    deduplicator: Deduplicator | None = None,
) -> dict[str, Any] | None:
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

        write_stats_by_classify_card_type = {
            "ideas": _zero_write_stats(),
            "programs": _zero_write_stats(),
        }
        try:
            for idx, card in enumerate(memory_cards, start=1):
                card_type = _classify_card_type(card)
                before_stats = memory.get_card_write_stats()
                memory_id = memory.save_card(card)
                after_stats = memory.get_card_write_stats()
                stat_diff = _diff_write_stats(before_stats, after_stats)
                for stat_name, stat_value in stat_diff.items():
                    write_stats_by_classify_card_type[card_type][stat_name] = int(
                        write_stats_by_classify_card_type[card_type].get(stat_name, 0)
                    ) + int(stat_value)
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

        write_stats = memory.get_card_write_stats()
        input_classify_card_type_counts = {
            "ideas": sum(
                1 for card in memory_cards if _classify_card_type(card) == "ideas"
            ),
            "programs": sum(
                1 for card in memory_cards if _classify_card_type(card) == "programs"
            ),
        }
        logger.info(
            "[Memory][WritePipeline] Write stats: processed={} added={} updated={} rejected={} "
            "ideas(proc={} add={} upd={} rej={}) "
            "programs(proc={} add={} upd={} rej={}) updated_target_cards={}",
            write_stats.get("processed", 0),
            write_stats.get("added", 0),
            write_stats.get("updated", 0),
            write_stats.get("rejected", 0),
            write_stats_by_classify_card_type["ideas"].get("processed", 0),
            write_stats_by_classify_card_type["ideas"].get("added", 0),
            write_stats_by_classify_card_type["ideas"].get("updated", 0),
            write_stats_by_classify_card_type["ideas"].get("rejected", 0),
            write_stats_by_classify_card_type["programs"].get("processed", 0),
            write_stats_by_classify_card_type["programs"].get("added", 0),
            write_stats_by_classify_card_type["programs"].get("updated", 0),
            write_stats_by_classify_card_type["programs"].get("rejected", 0),
            write_stats.get("updated_target_cards", 0),
        )

        stats_path = banks_path.parent / "memory_write_stats.json"
        snapshot = _write_memory_write_stats(
            stats_path=stats_path,
            input_cards_count=len(memory_cards),
            input_classify_card_type_counts=input_classify_card_type_counts,
            write_stats=write_stats,
            write_stats_by_classify_card_type=write_stats_by_classify_card_type,
        )
        logger.info(
            "[Memory][WritePipeline] Memory write stats saved to: {}", stats_path
        )
        return snapshot
    finally:
        memory.close()
