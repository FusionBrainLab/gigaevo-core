"""Summarize canonical memory event telemetry for debugging.

Usage examples:

    python tools/memory_event_report.py outputs/run/memory
    python tools/memory_event_report.py outputs/run --top-n 20
    python tools/memory_event_report.py --events /tmp/memory_events.jsonl --json

Reads the three artifacts a memory-enabled run leaves under its checkpoint dir:
``memory_events.jsonl`` (flat rows, one per ``MEMORY_*`` canonical event),
``write_ledger.jsonl`` (one row per admission verdict) and ``cards.json``
(the card bank). The tool is intentionally read-only and tolerant of missing
files. A partial run can still answer useful questions such as why memory was
empty, which cards won the auction, whether a few cards dominate prompts, and
whether program cards are consuming most of the budget.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from statistics import median
from typing import Any

from gigaevo.memory.cards import CardStatsBlock, ContextualGain
from gigaevo.memory.read.reputation import block_from_events

DEFAULT_EVENTS = "memory_events.jsonl"
DEFAULT_LEDGER = "write_ledger.jsonl"
DEFAULT_BANK = "cards.json"


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            rows.append(
                {
                    "_invalid_json": True,
                    "_line_no": line_no,
                    "_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_bank(path: Path | None) -> list[dict[str, Any]]:
    """Card dicts from a ``cards.json`` bank file; empty on missing/corrupt."""
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cards = payload.get("cards") if isinstance(payload, Mapping) else None
    if not isinstance(cards, Mapping):
        return []
    return [card for card in cards.values() if isinstance(card, Mapping)]


def _resolve_checkpoint_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    if (path / DEFAULT_EVENTS).exists() or (path / DEFAULT_BANK).exists():
        return path
    memory_dir = path / "memory"
    if (memory_dir / DEFAULT_EVENTS).exists() or (memory_dir / DEFAULT_BANK).exists():
        return memory_dir
    return path


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _card_id(card: Mapping[str, Any]) -> str:
    value = card.get("id", "")
    return str(value) if value is not None else ""


def _card_kind(card: Mapping[str, Any]) -> str:
    return str(card.get("kind") or "insight")


def _card_kind_for_id(
    card_id: str, cards_by_id: Mapping[str, Mapping[str, Any]]
) -> str:
    card = cards_by_id.get(card_id)
    if card is not None:
        return _card_kind(card)
    if card_id.startswith("static:"):
        return "static"
    return "unknown"


def _card_block(card: Mapping[str, Any]) -> CardStatsBlock | None:
    """Live reputation block from a card's persisted ``gain_events``.

    The same ``block_from_events`` math the auction reads on its own gain
    events, so the monitor reports the real posterior rather than a stale
    persisted field. ``None`` for a card with no events (cold prior only)."""
    raw = card.get("gain_events")
    if not isinstance(raw, list):
        return None
    events: list[ContextualGain] = []
    for entry in raw:
        if isinstance(entry, Mapping):
            events.append(ContextualGain.model_validate(entry))
    return block_from_events(events)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percent(part: int | float, total: int | float) -> float:
    return 100.0 * float(part) / float(total) if total else 0.0


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(counter.most_common())


def _top_counts(items: Iterable[str], top_n: int) -> list[dict[str, Any]]:
    return [
        {"id": key, "count": count} for key, count in Counter(items).most_common(top_n)
    ]


def _flatten_ids(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    ids: list[str] = []
    for row in rows:
        ids.extend(str(card_id) for card_id in _as_list(row.get(key)) if card_id)
    return ids


def _by_event(
    events: Sequence[Mapping[str, Any]], name: str
) -> list[Mapping[str, Any]]:
    return [row for row in events if row.get("event") == name]


def summarize_memory_events(
    *,
    events: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
    top_n: int = 10,
) -> dict[str, Any]:
    cards_by_id = {_card_id(card): card for card in cards if _card_id(card)}
    read_events = _by_event(events, "MEMORY_READ_SELECTION")
    research_events = _by_event(events, "MEMORY_RESEARCH")
    research_steps = _by_event(events, "MEMORY_RESEARCH_STEP")
    auction_events = _by_event(events, "MEMORY_AUCTION_RUN")
    budget_events = _by_event(events, "MEMORY_BUDGET_CAP")
    store_write_events = _by_event(events, "MEMORY_STORE_WRITE")
    store_sync_events = _by_event(events, "MEMORY_STORE_SYNC")
    eviction_events = _by_event(events, "MEMORY_EVICTION_SWEEP")
    consolidation_events = _by_event(events, "MEMORY_CONSOLIDATION_PASS")
    restamp_events = _by_event(events, "MEMORY_GAIN_RESTAMP")

    selected_ids = _flatten_ids(read_events, "selected_ids")
    candidate_ids = _flatten_ids(read_events, "candidate_ids")
    render_dropped_ids = _flatten_ids(read_events, "render_dropped_ids")

    empty_reasons: Counter[str] = Counter()
    selected_decisions = 0
    empty_after_candidates = 0
    slate_total = 0
    slate_selected = 0
    read_total_ms: list[float] = []
    for row in read_events:
        if _as_list(row.get("selected_ids")):
            selected_decisions += 1
            empty_reasons["selected"] += 1
        else:
            empty_reasons[str(row.get("empty_reason") or "unknown_empty")] += 1
            if _as_list(row.get("candidate_ids")):
                empty_after_candidates += 1
        slate = _as_list(row.get("slate"))
        slate_total += len(slate)
        slate_selected += sum(
            1 for bid in slate if isinstance(bid, Mapping) and bool(bid.get("selected"))
        )
        total = _safe_float(_as_dict(row.get("timing_ms")).get("total"))
        if total is not None:
            read_total_ms.append(total)

    research_ms = [
        duration
        for row in research_events
        if (duration := _safe_float(row.get("duration_ms"))) is not None
    ]
    research_outcomes = Counter(
        str(row.get("outcome")) for row in research_events if row.get("outcome")
    )
    research_iterations = [
        n for row in research_events if (n := _safe_int(row.get("iterations")))
    ]
    step_decisions = Counter(
        str(row.get("decision")) for row in research_steps if row.get("decision")
    )
    step_scopes: Counter[str] = Counter()
    for row in research_steps:
        step_scopes.update(str(scope) for scope in _as_list(row.get("scopes")))

    selected_kind_counts = Counter(
        _card_kind_for_id(card_id, cards_by_id) for card_id in selected_ids
    )
    candidate_kind_counts = Counter(
        _card_kind_for_id(card_id, cards_by_id) for card_id in candidate_ids
    )

    auction_kinds = Counter(
        str(row.get("auction")) for row in auction_events if row.get("auction")
    )
    budget_dropped_ids = _flatten_ids(budget_events, "dropped_ids")

    ledger_outcomes = Counter(
        str(row.get("outcome")) for row in ledger if row.get("outcome") is not None
    )
    ledger_categories = Counter(
        str(row.get("category")) for row in ledger if row.get("category")
    )
    store_ops = Counter(
        f"{row.get('op')}:{row.get('outcome')}" for row in store_write_events
    )
    sync_ops = Counter(
        f"{row.get('op')}:{row.get('outcome')}" for row in store_sync_events
    )
    sync_durations = [
        duration
        for row in store_sync_events
        if (duration := _safe_float(row.get("duration_ms"))) is not None
    ]
    bank_counts = [
        n for row in store_write_events if (n := _safe_int(row.get("bank_count")))
    ]
    evicted_ids = _flatten_ids(eviction_events, "evicted_ids")
    consolidation_outcomes = Counter(
        str(row.get("outcome")) for row in consolidation_events if row.get("outcome")
    )
    consolidation_merged = sum(
        _safe_int(row.get("merged")) or 0 for row in consolidation_events
    )

    intro_events: list[int] = []
    posterior_count = 0
    confident_count = 0
    bank_kind_counts: Counter[str] = Counter()
    for card in cards:
        bank_kind_counts[_card_kind(card)] += 1
        block = _card_block(card)
        if block is None:
            continue
        posterior_count += 1
        if block.efficacy_confident:
            confident_count += 1
        intro_events.append(block.intro_events)

    last_restamp = restamp_events[-1] if restamp_events else {}

    total_selected = len(selected_ids)
    top_selected = _top_counts(selected_ids, top_n)
    top1_count = top_selected[0]["count"] if top_selected else 0
    top5_count = sum(item["count"] for item in top_selected[:5])

    return {
        "events": {
            "total": len(events),
            "invalid_json": sum(1 for row in events if row.get("_invalid_json")),
            "by_event": _counter_dict(
                Counter(
                    str(row.get("event"))
                    for row in events
                    if not row.get("_invalid_json")
                )
            ),
        },
        "read": {
            "decisions": len(read_events),
            "selected_decisions": selected_decisions,
            "empty_decisions": len(read_events) - selected_decisions,
            "empty_after_candidates": empty_after_candidates,
            "empty_reasons": _counter_dict(empty_reasons),
            "candidate_total": len(candidate_ids),
            "render_dropped_total": len(render_dropped_ids),
            "selected_total": total_selected,
            "unique_selected": len(set(selected_ids)),
            "top_selected": top_selected,
            "top1_share_pct": _percent(top1_count, total_selected),
            "top5_share_pct": _percent(top5_count, total_selected),
            "avg_total_ms": _avg(read_total_ms),
        },
        "research": {
            "events": len(research_events),
            "outcomes": _counter_dict(research_outcomes),
            "avg_iterations": _avg([float(n) for n in research_iterations]),
            "avg_duration_ms": _avg(research_ms),
            "max_duration_ms": max(research_ms) if research_ms else None,
            "steps": len(research_steps),
            "step_decisions": _counter_dict(step_decisions),
            "step_scopes": _counter_dict(step_scopes),
        },
        "auction": {
            "event_count": len(auction_events),
            "by_auction": _counter_dict(auction_kinds),
            "slate_total": slate_total,
            "slate_selected": slate_selected,
            "slate_rejected": slate_total - slate_selected,
            "rejection_rate_pct": _percent(slate_total - slate_selected, slate_total),
        },
        "budget": {
            "cap_events": len(budget_events),
            "dropped_total": len(budget_dropped_ids),
            "top_dropped": _top_counts(budget_dropped_ids, top_n),
        },
        "card_kinds": {
            "candidate": _counter_dict(candidate_kind_counts),
            "selected": _counter_dict(selected_kind_counts),
            "bank": _counter_dict(bank_kind_counts),
        },
        "write_ledger": {
            "rows": len(ledger),
            "invalid_json": sum(1 for row in ledger if row.get("_invalid_json")),
            "outcomes": _counter_dict(ledger_outcomes),
            "categories": _counter_dict(ledger_categories),
        },
        "store": {
            "write_events": len(store_write_events),
            "write_ops": _counter_dict(store_ops),
            "last_bank_count": bank_counts[-1] if bank_counts else None,
            "sync_events": len(store_sync_events),
            "sync_ops": _counter_dict(sync_ops),
            "avg_sync_ms": _avg(sync_durations),
            "max_sync_ms": max(sync_durations) if sync_durations else None,
        },
        "maintenance": {
            "eviction_sweeps": len(eviction_events),
            "evicted_total": len(evicted_ids),
            "top_evicted": _top_counts(evicted_ids, top_n),
            "consolidation_passes": len(consolidation_events),
            "consolidation_outcomes": _counter_dict(consolidation_outcomes),
            "consolidation_merged": consolidation_merged,
        },
        "bank": {
            "cards": len(cards),
            "posterior_cards": posterior_count,
            "confident_cards": confident_count,
            "intro_events_median": median(intro_events) if intro_events else None,
            "intro_events_max": max(intro_events) if intro_events else None,
        },
        "gain_restamp": {
            "events": len(restamp_events),
            "last_credited_card_count": last_restamp.get("credited_card_count"),
            "last_event_count": sum(
                _safe_int(count) or 0
                for count in _as_dict(
                    last_restamp.get("event_count_by_card_id")
                ).values()
            ),
        },
    }


def build_report(
    checkpoint_dir: Path | None = None,
    *,
    events_path: Path | None = None,
    ledger_path: Path | None = None,
    cards_path: Path | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint_dir(checkpoint_dir)
    events_file = events_path or (checkpoint / DEFAULT_EVENTS if checkpoint else None)
    ledger_file = ledger_path or (checkpoint / DEFAULT_LEDGER if checkpoint else None)
    cards_file = cards_path or (checkpoint / DEFAULT_BANK if checkpoint else None)

    events = _read_jsonl(events_file)
    ledger = _read_jsonl(ledger_file)
    cards = _read_bank(cards_file)
    summary = summarize_memory_events(
        events=events, ledger=ledger, cards=cards, top_n=top_n
    )
    summary["files"] = {
        "checkpoint_dir": str(checkpoint) if checkpoint else "",
        "events": str(events_file) if events_file else "",
        "events_exists": bool(events_file and events_file.exists()),
        "write_ledger": str(ledger_file) if ledger_file else "",
        "write_ledger_exists": bool(ledger_file and ledger_file.exists()),
        "cards": str(cards_file) if cards_file else "",
        "cards_exists": bool(cards_file and cards_file.exists()),
    }
    return summary


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.1f} ms"
    except (TypeError, ValueError):
        return "n/a"


def _format_counts(counts: Mapping[str, int], *, empty: str = "none") -> list[str]:
    if not counts:
        return [f"  {empty}"]
    return [f"  {key}: {value}" for key, value in counts.items()]


def _format_top(
    items: Sequence[Mapping[str, Any]], *, empty: str = "none"
) -> list[str]:
    if not items:
        return [f"  {empty}"]
    return [f"  {item['id']}: {item['count']}" for item in items]


def format_report(summary: Mapping[str, Any]) -> str:
    files = summary["files"]
    events = summary["events"]
    read = summary["read"]
    research = summary["research"]
    auction = summary["auction"]
    budget = summary["budget"]
    card_kinds = summary["card_kinds"]
    ledger = summary["write_ledger"]
    store = summary["store"]
    maintenance = summary["maintenance"]
    bank = summary["bank"]
    restamp = summary["gain_restamp"]

    lines = [
        "Memory Event Audit",
        "",
        "Files",
        f"  events: {files['events']} ({'found' if files['events_exists'] else 'missing'})",
        f"  write ledger: {files['write_ledger']} ({'found' if files['write_ledger_exists'] else 'missing'})",
        f"  cards: {files['cards']} ({'found' if files['cards_exists'] else 'missing'})",
        "",
        "Event Stream",
        f"  rows: {events['total']} invalid_json: {events['invalid_json']}",
        "  by event:",
    ]
    lines.extend(_format_counts(events["by_event"], empty="no event rows"))
    lines.extend(
        [
            "",
            "Read Decisions",
            f"  decisions: {read['decisions']}",
            f"  selected decisions: {read['selected_decisions']}",
            f"  empty decisions: {read['empty_decisions']}",
            f"  empty after candidates: {read['empty_after_candidates']}",
            f"  candidates: {read['candidate_total']} render-dropped: {read['render_dropped_total']}",
            f"  selected cards: {read['selected_total']} unique: {read['unique_selected']}",
            f"  top1 share: {_fmt_pct(read['top1_share_pct'])} top5 share: {_fmt_pct(read['top5_share_pct'])}",
            f"  avg total: {_fmt_ms(read['avg_total_ms'])}",
            "  empty reasons:",
        ]
    )
    lines.extend(_format_counts(read["empty_reasons"], empty="none"))
    lines.extend(["", "Top Selected Cards"])
    lines.extend(_format_top(read["top_selected"]))
    lines.extend(
        [
            "",
            "Research",
            f"  events: {research['events']} steps: {research['steps']}",
            f"  avg iterations: {research['avg_iterations']}",
            f"  avg duration: {_fmt_ms(research['avg_duration_ms'])} max: {_fmt_ms(research['max_duration_ms'])}",
            "  outcomes:",
        ]
    )
    lines.extend(_format_counts(research["outcomes"], empty="none"))
    lines.append("  step decisions:")
    lines.extend(_format_counts(research["step_decisions"], empty="none"))
    lines.append("  step scopes:")
    lines.extend(_format_counts(research["step_scopes"], empty="none"))
    lines.extend(
        [
            "",
            "Auction",
            f"  auction events: {auction['event_count']}",
            "  by auction:",
        ]
    )
    lines.extend(_format_counts(auction["by_auction"], empty="none"))
    lines.extend(
        [
            f"  slate bids: {auction['slate_total']}",
            f"  slate selected: {auction['slate_selected']}",
            f"  slate rejected: {auction['slate_rejected']}",
            f"  rejection rate: {_fmt_pct(auction['rejection_rate_pct'])}",
            "",
            "Budget",
            f"  cap events: {budget['cap_events']}",
            f"  dropped cards: {budget['dropped_total']}",
            "  top dropped:",
        ]
    )
    lines.extend(_format_top(budget["top_dropped"]))
    lines.extend(
        [
            "",
            "Card Kind Mix",
            "  candidates:",
        ]
    )
    lines.extend(_format_counts(card_kinds["candidate"]))
    lines.append("  selected:")
    lines.extend(_format_counts(card_kinds["selected"]))
    lines.append("  bank:")
    lines.extend(_format_counts(card_kinds["bank"]))
    lines.extend(
        [
            "",
            "Write Ledger",
            f"  rows: {ledger['rows']} invalid_json: {ledger['invalid_json']}",
            "  outcomes:",
        ]
    )
    lines.extend(_format_counts(ledger["outcomes"]))
    lines.append("  categories:")
    lines.extend(_format_counts(ledger["categories"]))
    lines.extend(
        [
            "",
            "Store",
            f"  write events: {store['write_events']} last bank count: {store['last_bank_count']}",
            "  write ops:",
        ]
    )
    lines.extend(_format_counts(store["write_ops"], empty="none"))
    lines.extend(
        [
            f"  sync events: {store['sync_events']}",
            f"  avg sync: {_fmt_ms(store['avg_sync_ms'])} max: {_fmt_ms(store['max_sync_ms'])}",
            "  sync ops:",
        ]
    )
    lines.extend(_format_counts(store["sync_ops"], empty="none"))
    lines.extend(
        [
            "",
            "Maintenance",
            f"  eviction sweeps: {maintenance['eviction_sweeps']} evicted: {maintenance['evicted_total']}",
            "  top evicted:",
        ]
    )
    lines.extend(_format_top(maintenance["top_evicted"]))
    lines.extend(
        [
            f"  consolidation passes: {maintenance['consolidation_passes']} merged: {maintenance['consolidation_merged']}",
            "  consolidation outcomes:",
        ]
    )
    lines.extend(_format_counts(maintenance["consolidation_outcomes"], empty="none"))
    lines.extend(
        [
            "",
            "Card Bank",
            f"  cards: {bank['cards']}",
            f"  posterior cards: {bank['posterior_cards']}",
            f"  confident cards: {bank['confident_cards']}",
            f"  median intro events: {bank['intro_events_median']}",
            f"  max intro events: {bank['intro_events_max']}",
            "",
            "Gain Restamp",
            f"  events: {restamp['events']}",
            f"  last credited cards: {restamp['last_credited_card_count']}",
            f"  last event count: {restamp['last_event_count']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        type=Path,
        help="Memory checkpoint dir, or a run dir containing memory/.",
    )
    parser.add_argument(
        "--events", type=Path, help="Explicit memory_events.jsonl path."
    )
    parser.add_argument(
        "--write-ledger", type=Path, help="Explicit write_ledger.jsonl path."
    )
    parser.add_argument("--cards", type=Path, help="Explicit cards.json bank path.")
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of card IDs to show."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_report(
        args.checkpoint_dir,
        events_path=args.events,
        ledger_path=args.write_ledger,
        cards_path=args.cards,
        top_n=args.top_n,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_report(summary), end="")


if __name__ == "__main__":
    main()
