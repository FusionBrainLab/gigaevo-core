#!/usr/bin/env python3
"""Memory-card health snapshot for live memory runs.

Read-only. Gathers a compact, faithful snapshot of the on-disk card bank
(``<run>/memory/cards.json``) across one or more runs so a reviewer (human
or LLM) can judge how cards look and whether their attributes are adequate.

The objective integrity checks here target the absorbed-id alias layer (the new
gain-attribution machinery): an absorbed id must reference a *removed* card, be
claimed by exactly one survivor, and never name its own survivor. Subjective
quality (specificity, tautology, dedup) is left to the existing
``memory_quality_audit.py`` and to the reviewer's verdict; the event side
(auction winners, dominance, budget) is delegated to ``memory_event_report.py``.

Usage:
    python tools/memory_card_health.py outputs/run_a outputs/run_b --json out.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVENT_REPORT = _REPO_ROOT / "tools" / "memory_event_report.py"


class HealthFlag(NamedTuple):
    """One objective integrity concern about a card. ``detail`` is the offending
    value (an absorbed id, a partner id, a description preview)."""

    kind: str
    card_id: str
    detail: str


@dataclass(frozen=True)
class CardHealth:
    card_id: str
    card_type: str
    missing_description: bool
    n_keywords: int
    n_programs: int
    n_gain_events: int
    absorbed_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunHealth:
    run: str
    n_cards: int
    n_mem: int
    n_program: int
    n_missing_description: int
    n_zero_keywords: int
    n_with_gain_events: int
    n_with_absorbed: int
    cards: tuple[CardHealth, ...] = field(default_factory=tuple)
    flags: tuple[HealthFlag, ...] = field(default_factory=tuple)


def _card_type(card_id: str, card: Mapping[str, Any] | None = None) -> str:
    kind = (card or {}).get("kind")
    if kind == "program":
        return "program"
    if kind == "insight":
        return "mem"
    if card_id.startswith("mem-"):
        return "mem"
    if card_id.startswith(("prog-", "program-")):
        return "program"
    return "other"


def _as_seq(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def assess_card(card_id: str, card: Mapping[str, Any]) -> CardHealth:
    desc = card.get("description")
    return CardHealth(
        card_id=card_id,
        card_type=_card_type(card_id, card),
        missing_description=not (isinstance(desc, str) and desc.strip()),
        n_keywords=len(_as_seq(card.get("keywords"))),
        n_programs=len(_as_seq(card.get("programs"))),
        n_gain_events=len(_as_seq(card.get("gain_events"))),
        absorbed_ids=tuple(str(a) for a in _as_seq(card.get("absorbed_ids"))),
    )


def assess_run(run: str, cards: Mapping[str, Mapping[str, Any]]) -> RunHealth:
    healths = tuple(assess_card(cid, cards[cid]) for cid in sorted(cards))
    live_ids = set(cards)
    flags: list[HealthFlag] = []

    survivors_of: dict[str, list[str]] = defaultdict(list)
    for h in healths:
        if h.missing_description:
            flags.append(HealthFlag("missing_description", h.card_id, ""))
        for aid in h.absorbed_ids:
            if aid == h.card_id:
                flags.append(HealthFlag("self_absorbed", h.card_id, aid))
            elif aid in live_ids:
                flags.append(HealthFlag("absorbed_id_still_live", h.card_id, aid))
            survivors_of[aid].append(h.card_id)

    for aid, survivors in sorted(survivors_of.items()):
        if len(survivors) > 1:
            flags.append(HealthFlag("cross_absorbed", ",".join(sorted(survivors)), aid))

    by_desc: dict[str, list[str]] = defaultdict(list)
    for cid in sorted(cards):
        desc = cards[cid].get("description")
        if isinstance(desc, str) and desc.strip():
            by_desc[desc.strip()].append(cid)
    for desc, ids in sorted(by_desc.items()):
        if len(ids) > 1:
            preview = desc[:60] + ("…" if len(desc) > 60 else "")
            for cid in ids:
                flags.append(HealthFlag("duplicate_description", cid, preview))

    return RunHealth(
        run=run,
        n_cards=len(healths),
        n_mem=sum(h.card_type == "mem" for h in healths),
        n_program=sum(h.card_type == "program" for h in healths),
        n_missing_description=sum(h.missing_description for h in healths),
        n_zero_keywords=sum(
            h.card_type == "mem" and h.n_keywords == 0 for h in healths
        ),
        n_with_gain_events=sum(h.n_gain_events > 0 for h in healths),
        n_with_absorbed=sum(bool(h.absorbed_ids) for h in healths),
        cards=healths,
        flags=tuple(flags),
    )


def load_card_bank(run_root: Path) -> dict[str, dict[str, Any]]:
    bank = run_root / "memory" / "cards.json"
    if not bank.exists():
        return {}
    try:
        payload = json.loads(bank.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    cards = payload.get("cards") if isinstance(payload, dict) else {}
    if not isinstance(cards, dict):
        return {}
    return {cid: c for cid, c in cards.items() if isinstance(c, dict)}


def _event_summary(run_root: Path) -> dict[str, Any] | None:
    """Delegate the event side (auction/dominance/budget) to memory_event_report."""
    try:
        out = subprocess.run(
            [sys.executable, str(_EVENT_REPORT), str(run_root), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode == 0 and out.stdout.strip():
            return json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    return None


def _run_to_dict(h: RunHealth, events: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "run": h.run,
        "rollup": {
            "n_cards": h.n_cards,
            "n_mem": h.n_mem,
            "n_program": h.n_program,
            "n_missing_description": h.n_missing_description,
            "n_zero_keywords": h.n_zero_keywords,
            "n_with_gain_events": h.n_with_gain_events,
            "n_with_absorbed": h.n_with_absorbed,
        },
        "flags": [f._asdict() for f in h.flags],
        "cards": [
            {
                "id": c.card_id,
                "type": c.card_type,
                "missing_description": c.missing_description,
                "n_keywords": c.n_keywords,
                "n_programs": c.n_programs,
                "n_gain_events": c.n_gain_events,
                "absorbed_ids": list(c.absorbed_ids),
            }
            for c in h.cards
        ],
        "events": events,
    }


def _format_markdown(runs: Sequence[dict[str, Any]]) -> str:
    lines = ["# Memory-card health snapshot", ""]
    for r in runs:
        roll = r["rollup"]
        lines.append(f"## {r['run']}")
        if roll["n_cards"] == 0:
            lines.append("- no cards yet (cards.json absent or empty)")
            lines.append("")
            continue
        lines.append(
            f"- cards: {roll['n_cards']} ({roll['n_mem']} mem / {roll['n_program']} program)"
        )
        lines.append(
            f"- attribute adequacy: missing_description={roll['n_missing_description']}, "
            f"zero_keywords(mem)={roll['n_zero_keywords']}, "
            f"with_gain_events={roll['n_with_gain_events']}, "
            f"with_absorbed_ids={roll['n_with_absorbed']}"
        )
        flags = r["flags"]
        if flags:
            lines.append(f"- 🚨 SUSPICIOUS ({len(flags)}):")
            for f in flags:
                lines.append(f"    - {f['kind']}: {f['card_id']} → {f['detail']}")
        else:
            lines.append("- ✅ no structural/integrity flags")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_roots", type=Path, nargs="+", help="Run output directories.")
    ap.add_argument("--json", type=Path, default=None, help="Write combined JSON here.")
    ap.add_argument(
        "--no-events",
        action="store_true",
        help="Skip the memory_event_report delegation (cards only).",
    )
    args = ap.parse_args()

    runs: list[dict[str, Any]] = []
    for root in args.run_roots:
        bank = load_card_bank(root)
        health = assess_run(root.name, bank)
        events = None if args.no_events else _event_summary(root)
        runs.append(_run_to_dict(health, events))

    combined = {"runs": runs}
    if args.json:
        args.json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(_format_markdown(runs))


if __name__ == "__main__":
    main()
