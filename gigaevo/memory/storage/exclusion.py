"""Card-id alias helpers shared by storage retrieval and read-side excluders."""

from __future__ import annotations

from collections.abc import Iterable

from gigaevo.memory.cards import Card


def card_alias_ids(card: Card) -> frozenset[str]:
    """Every id that should refer to ``card`` for read-side exclusion."""
    return frozenset(
        cid.strip() for cid in (card.id, *card.absorbed_ids) if cid.strip()
    )


def is_card_excluded(card: Card, exclude_ids: frozenset[str]) -> bool:
    """True when the card's live id or any absorbed alias is excluded."""
    return bool(card_alias_ids(card) & exclude_ids)


def expand_exclude_ids(
    cards: Iterable[Card], exclude_ids: frozenset[str]
) -> frozenset[str]:
    """Expand lineage/bench exclusions through merge aliases.

    A program may have used ``mem-old`` before consolidation merged that card into
    survivor ``mem-new``. Filtering only the live id would let the same idea
    re-enter the prompt under the survivor id, so any excluded alias excludes the
    survivor and every alias it carries.
    """
    if not exclude_ids:
        return frozenset()
    expanded = set(exclude_ids)
    changed = True
    bank = tuple(cards)
    while changed:
        changed = False
        for card in bank:
            aliases = card_alias_ids(card)
            if aliases & expanded and not aliases <= expanded:
                expanded.update(aliases)
                changed = True
    return frozenset(expanded)
