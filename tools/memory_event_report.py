"""Summarize canonical memory event telemetry for debugging.

Usage examples:

    python tools/memory_event_report.py outputs/run/memory
    python tools/memory_event_report.py outputs/run --top-n 20
    python tools/memory_event_report.py --events /tmp/memory_events.jsonl --json

The tool is intentionally read-only and tolerant of missing files. A partial
run can still answer useful questions such as why memory was empty, which cards
won the auction, whether a few cards dominate prompts, and whether program cards
are consuming most of the budget.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from statistics import median
from typing import Any

from gigaevo.memory.context import ContextualGain
from gigaevo.memory.efficacy import block_from_events
from gigaevo.memory.shared_memory.models import CardStatsBlock

DEFAULT_EVENTS = "memory_events.jsonl"
DEFAULT_LEDGER = "write_ledger.jsonl"
DEFAULT_EXPORT = "amem_exports/amem_memories.jsonl"


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


def _resolve_checkpoint_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    if (path / DEFAULT_EVENTS).exists() or (path / DEFAULT_LEDGER).exists():
        return path
    memory_dir = path / "memory"
    if (memory_dir / DEFAULT_EVENTS).exists() or (memory_dir / DEFAULT_LEDGER).exists():
        return memory_dir
    return path


def _event_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _card_id(card: Mapping[str, Any]) -> str:
    value = card.get("id", "")
    return str(value) if value is not None else ""


def _card_type_from_id(card_id: str) -> str:
    return "program" if card_id.startswith("program-") else "idea"


def _card_type(card: Mapping[str, Any]) -> str:
    category = str(card.get("category", "") or "").lower()
    card_id = _card_id(card)
    if category == "program" or "program_id" in card or card_id.startswith("program-"):
        return "program"
    return "idea"


def _card_type_for_id(
    card_id: str, cards_by_id: Mapping[str, Mapping[str, Any]]
) -> str:
    card = cards_by_id.get(card_id)
    if card is not None:
        return _card_type(card)
    return _card_type_from_id(card_id)


def _card_block(card: Mapping[str, Any]) -> CardStatsBlock | None:
    """Live reputation block from a card's persisted ``gain_events``.

    The same ``block_from_events`` math the auction reads on its own gain
    events, so the monitor reports the real posterior rather than a removed
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


def _flatten_ids(read_events: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    ids: list[str] = []
    for row in read_events:
        payload = _event_payload(row)
        ids.extend(str(card_id) for card_id in _as_list(payload.get(key)) if card_id)
    return ids


def summarize_memory_events(
    *,
    events: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
    top_n: int = 10,
) -> dict[str, Any]:
    cards_by_id = {_card_id(card): card for card in cards if _card_id(card)}
    read_events = [row for row in events if row.get("event_type") == "read.selection"]
    read_request_events = [
        row for row in events if row.get("event_type") == "read.request"
    ]
    read_retrieval_events = [
        row for row in events if row.get("event_type") == "read.retrieval"
    ]
    auction_events = [row for row in events if row.get("event_type") == "auction.run"]
    budget_events = [row for row in events if row.get("event_type") == "budget.cap"]
    write_events = [
        row for row in events if str(row.get("event_type", "")).startswith("write.")
    ]
    store_events = [
        row for row in events if str(row.get("event_type", "")).startswith("store.")
    ]
    gam_events = [
        row for row in events if str(row.get("event_type", "")).startswith("gam.")
    ]
    bridge_events = [
        row for row in events if row.get("event_type") == "injection_posterior.compute"
    ]

    selected_ids = _flatten_ids(read_events, "selected_ids")
    candidate_ids = _flatten_ids(read_events, "candidate_ids")
    fetched_ids = _flatten_ids(read_events, "fetched_ids")
    missing_ids = _flatten_ids(read_events, "missing_ids")

    empty_reasons: Counter[str] = Counter()
    selected_decisions = 0
    empty_after_candidates = 0
    slate_total = 0
    slate_selected = 0
    read_total_ms: list[float] = []
    for row in read_events:
        payload = _event_payload(row)
        selected_count = _safe_int(payload.get("selected_count")) or len(
            _as_list(payload.get("selected_ids"))
        )
        if selected_count > 0:
            selected_decisions += 1
            empty_reasons["selected"] += 1
        else:
            reason = str(payload.get("empty_reason") or "unknown_empty")
            empty_reasons[reason] += 1
            if (_safe_int(payload.get("candidate_count")) or 0) > 0:
                empty_after_candidates += 1
        slate = _as_list(payload.get("slate"))
        slate_total += len(slate)
        slate_selected += sum(
            1 for bid in slate if isinstance(bid, Mapping) and bool(bid.get("selected"))
        )
        timing = payload.get("timing_ms")
        if isinstance(timing, Mapping):
            total = _safe_float(timing.get("total"))
            if total is not None:
                read_total_ms.append(total)

    retrieval_ms = [
        duration
        for row in read_retrieval_events
        if (duration := _safe_float(_event_payload(row).get("duration_ms"))) is not None
    ]

    selected_type_counts = Counter(
        _card_type_for_id(card_id, cards_by_id) for card_id in selected_ids
    )
    candidate_type_counts = Counter(
        _card_type_for_id(card_id, cards_by_id) for card_id in candidate_ids
    )

    budget_dropped_ids: list[str] = []
    for row in budget_events:
        budget_dropped_ids.extend(
            str(card_id)
            for card_id in _as_list(_event_payload(row).get("dropped_ids"))
            if card_id
        )

    ledger_outcomes = Counter(
        str(row.get("outcome")) for row in ledger if row.get("outcome") is not None
    )
    ledger_categories = Counter(
        str(row.get("category")) for row in ledger if row.get("category")
    )
    write_event_outcomes = Counter(
        str(_event_payload(row).get("outcome"))
        for row in write_events
        if _event_payload(row).get("outcome") is not None
    )
    write_event_categories = Counter(
        str(_event_payload(row).get("category"))
        for row in write_events
        if _event_payload(row).get("category")
    )
    write_event_final_ids = [
        str(_event_payload(row).get("final_id"))
        for row in write_events
        if _event_payload(row).get("final_id")
    ]
    store_event_types = Counter(str(row.get("event_type")) for row in store_events)
    store_outcomes = Counter(
        str(_event_payload(row).get("outcome"))
        for row in store_events
        if _event_payload(row).get("outcome") is not None
    )
    store_modes = Counter(
        str(_event_payload(row).get("mode"))
        for row in store_events
        if _event_payload(row).get("mode") is not None
    )
    store_durations = [
        duration
        for row in store_events
        if (duration := _safe_float(_event_payload(row).get("duration_ms"))) is not None
    ]
    gam_tools: list[str] = []
    gam_modes: Counter[str] = Counter()
    for row in gam_events:
        payload = _event_payload(row)
        tool = payload.get("tool")
        if tool is not None:
            gam_tools.append(str(tool))
        for key in ("selected_tools", "filtered_tools", "plan_tools"):
            gam_tools.extend(str(item) for item in _as_list(payload.get(key)) if item)
        mode = payload.get("mode")
        if mode is not None:
            gam_modes[str(mode)] += 1
    gam_outcomes = Counter(
        str(_event_payload(row).get("outcome"))
        for row in gam_events
        if _event_payload(row).get("outcome") is not None
    )
    gam_durations = [
        duration
        for row in gam_events
        if (duration := _safe_float(_event_payload(row).get("duration_ms"))) is not None
    ]

    intro_events: list[int] = []
    posterior_count = 0
    confident_count = 0
    bank_type_counts: Counter[str] = Counter()
    for card in cards:
        bank_type_counts[_card_type(card)] += 1
        block = _card_block(card)
        if block is None:
            continue
        posterior_count += 1
        if block.efficacy_confident:
            confident_count += 1
        intro_events.append(block.intro_events)

    bridge_payloads = [_event_payload(row) for row in bridge_events]
    last_bridge = bridge_payloads[-1] if bridge_payloads else {}

    total_selected = len(selected_ids)
    top_selected = _top_counts(selected_ids, top_n)
    top1_count = top_selected[0]["count"] if top_selected else 0
    top5_count = sum(item["count"] for item in top_selected[:5])

    return {
        "events": {
            "total": len(events),
            "invalid_json": sum(1 for row in events if row.get("_invalid_json")),
            "by_type": _counter_dict(
                Counter(str(row.get("event_type")) for row in events)
            ),
            "by_component": _counter_dict(
                Counter(str(row.get("component")) for row in events)
            ),
        },
        "read": {
            "request_events": len(read_request_events),
            "retrieval_events": len(read_retrieval_events),
            "decisions": len(read_events),
            "selected_decisions": selected_decisions,
            "empty_decisions": len(read_events) - selected_decisions,
            "empty_after_candidates": empty_after_candidates,
            "empty_reasons": _counter_dict(empty_reasons),
            "candidate_total": len(candidate_ids),
            "fetched_total": len(fetched_ids),
            "missing_total": len(missing_ids),
            "selected_total": total_selected,
            "unique_selected": len(set(selected_ids)),
            "top_selected": top_selected,
            "top1_share_pct": _percent(top1_count, total_selected),
            "top5_share_pct": _percent(top5_count, total_selected),
            "avg_retrieval_ms": _avg(retrieval_ms),
            "avg_total_ms": _avg(read_total_ms),
        },
        "auction": {
            "event_count": len(auction_events),
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
        "card_types": {
            "candidate": _counter_dict(candidate_type_counts),
            "selected": _counter_dict(selected_type_counts),
            "bank": _counter_dict(bank_type_counts),
        },
        "write_ledger": {
            "rows": len(ledger),
            "invalid_json": sum(1 for row in ledger if row.get("_invalid_json")),
            "outcomes": _counter_dict(ledger_outcomes),
            "categories": _counter_dict(ledger_categories),
        },
        "write_events": {
            "events": len(write_events),
            "by_type": _counter_dict(
                Counter(str(row.get("event_type")) for row in write_events)
            ),
            "outcomes": _counter_dict(write_event_outcomes),
            "categories": _counter_dict(write_event_categories),
            "top_final_ids": _top_counts(write_event_final_ids, top_n),
        },
        "store_events": {
            "events": len(store_events),
            "by_type": _counter_dict(store_event_types),
            "outcomes": _counter_dict(store_outcomes),
            "modes": _counter_dict(store_modes),
            "avg_duration_ms": _avg(store_durations),
            "max_duration_ms": max(store_durations) if store_durations else None,
        },
        "gam_events": {
            "events": len(gam_events),
            "by_type": _counter_dict(
                Counter(str(row.get("event_type")) for row in gam_events)
            ),
            "outcomes": _counter_dict(gam_outcomes),
            "modes": _counter_dict(gam_modes),
            "tools": _counter_dict(Counter(gam_tools)),
            "avg_duration_ms": _avg(gam_durations),
            "max_duration_ms": max(gam_durations) if gam_durations else None,
        },
        "bank": {
            "cards": len(cards),
            "posterior_cards": posterior_count,
            "confident_cards": confident_count,
            "intro_events_median": median(intro_events) if intro_events else None,
            "intro_events_max": max(intro_events) if intro_events else None,
        },
        "posterior_bridge": {
            "events": len(bridge_events),
            "last_card_count": last_bridge.get("card_count"),
            "last_event_count": sum(
                _safe_int(count) or 0
                for count in _as_dict(
                    last_bridge.get("event_count_by_card_id")
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
    cards_file = cards_path or (checkpoint / DEFAULT_EXPORT if checkpoint else None)

    events = _read_jsonl(events_file)
    ledger = _read_jsonl(ledger_file)
    cards = _read_jsonl(cards_file)
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
    auction = summary["auction"]
    budget = summary["budget"]
    card_types = summary["card_types"]
    ledger = summary["write_ledger"]
    write_events = summary["write_events"]
    store_events = summary["store_events"]
    gam_events = summary["gam_events"]
    bank = summary["bank"]
    bridge = summary["posterior_bridge"]

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
        "  by type:",
    ]
    lines.extend(_format_counts(events["by_type"], empty="no event rows"))
    lines.append("  by component:")
    lines.extend(_format_counts(events["by_component"], empty="no event rows"))
    lines.extend(
        [
            "",
            "Read Decisions",
            f"  request events: {read['request_events']}",
            f"  retrieval events: {read['retrieval_events']}",
            f"  decisions: {read['decisions']}",
            f"  selected decisions: {read['selected_decisions']}",
            f"  empty decisions: {read['empty_decisions']}",
            f"  empty after candidates: {read['empty_after_candidates']}",
            f"  candidates: {read['candidate_total']} fetched: {read['fetched_total']} missing: {read['missing_total']}",
            f"  selected cards: {read['selected_total']} unique: {read['unique_selected']}",
            f"  top1 share: {_fmt_pct(read['top1_share_pct'])} top5 share: {_fmt_pct(read['top5_share_pct'])}",
            f"  avg retrieval: {_fmt_ms(read['avg_retrieval_ms'])} avg total: {_fmt_ms(read['avg_total_ms'])}",
            "  empty reasons:",
        ]
    )
    lines.extend(_format_counts(read["empty_reasons"], empty="none"))
    lines.extend(["", "Top Selected Cards"])
    lines.extend(_format_top(read["top_selected"]))
    lines.extend(
        [
            "",
            "Auction",
            f"  auction events: {auction['event_count']}",
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
            "Card Type Mix",
            "  candidates:",
        ]
    )
    lines.extend(_format_counts(card_types["candidate"]))
    lines.append("  selected:")
    lines.extend(_format_counts(card_types["selected"]))
    lines.append("  bank:")
    lines.extend(_format_counts(card_types["bank"]))
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
            "Write Events",
            f"  events: {write_events['events']}",
            "  by type:",
        ]
    )
    lines.extend(_format_counts(write_events["by_type"], empty="none"))
    lines.append("  outcomes:")
    lines.extend(_format_counts(write_events["outcomes"], empty="none"))
    lines.append("  categories:")
    lines.extend(_format_counts(write_events["categories"], empty="none"))
    lines.append("  top final ids:")
    lines.extend(_format_top(write_events["top_final_ids"]))
    lines.extend(
        [
            "",
            "Store Events",
            f"  events: {store_events['events']}",
            f"  avg duration: {_fmt_ms(store_events['avg_duration_ms'])}",
            f"  max duration: {_fmt_ms(store_events['max_duration_ms'])}",
            "  by type:",
        ]
    )
    lines.extend(_format_counts(store_events["by_type"], empty="none"))
    lines.append("  outcomes:")
    lines.extend(_format_counts(store_events["outcomes"], empty="none"))
    lines.append("  modes:")
    lines.extend(_format_counts(store_events["modes"], empty="none"))
    lines.extend(
        [
            "",
            "GAM Events",
            f"  events: {gam_events['events']}",
            f"  avg duration: {_fmt_ms(gam_events['avg_duration_ms'])}",
            f"  max duration: {_fmt_ms(gam_events['max_duration_ms'])}",
            "  by type:",
        ]
    )
    lines.extend(_format_counts(gam_events["by_type"], empty="none"))
    lines.append("  outcomes:")
    lines.extend(_format_counts(gam_events["outcomes"], empty="none"))
    lines.append("  modes:")
    lines.extend(_format_counts(gam_events["modes"], empty="none"))
    lines.append("  tools:")
    lines.extend(_format_counts(gam_events["tools"], empty="none"))
    lines.extend(
        [
            "",
            "Exported Bank",
            f"  cards: {bank['cards']}",
            f"  posterior cards: {bank['posterior_cards']}",
            f"  confident cards: {bank['confident_cards']}",
            f"  median intro events: {bank['intro_events_median']}",
            f"  max intro events: {bank['intro_events_max']}",
            "",
            "Injection Posterior Bridge",
            f"  events: {bridge['events']}",
            f"  last card count: {bridge['last_card_count']}",
            f"  last event count: {bridge['last_event_count']}",
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
    parser.add_argument("--cards", type=Path, help="Explicit amem_memories.jsonl path.")
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
