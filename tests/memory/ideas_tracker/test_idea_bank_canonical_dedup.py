"""Tests for IdeaBank canonical-key dedup-merge (CARD_STRUCTURE_v2 §2 Stage A).

RED-phase TDD tests written before implementation.

Contract: when an Idea is added whose `keywords` contain a canonical:* token
that matches an existing idea, the bank MUST merge the new idea into the
existing one (programs append, explanation entry append, alias archive)
rather than insert a duplicate.
"""

from __future__ import annotations

from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank
from gigaevo.memory.ideas_tracker.models import Idea, IdeaExplanation


def _idea(
    description: str,
    *,
    canonical_key: str,
    programs: list[str] | None = None,
    extra_keywords: list[str] | None = None,
    motivation: str = "",
) -> Idea:
    keywords = [f"canonical:{canonical_key}"] + (extra_keywords or [])
    return Idea(
        description=description,
        keywords=keywords,
        programs=programs or [],
        explanation=IdeaExplanation(entries=[motivation] if motivation else []),
    )


class TestIdeaBankCanonicalDedup:
    def test_duplicate_canonical_key_merges_programs(self) -> None:
        bank = IdeaBank()
        a = _idea(
            "REMOVE target_log_transform: floor at 0.15 penalty",
            canonical_key="REMOVE:target_log_transform:_:_",
            programs=["p1"],
        )
        b = _idea(
            "REMOVE log target: hurts at low fitness",
            canonical_key="REMOVE:target_log_transform:_:_",
            programs=["p2"],
        )
        bank.add(a)
        bank.add(b)
        assert len(bank.all_ideas()) == 1
        merged = bank.all_ideas()[0]
        assert set(merged.programs) == {"p1", "p2"}

    def test_duplicate_canonical_key_archives_second_description(self) -> None:
        bank = IdeaBank()
        a = _idea(
            "REMOVE target_log_transform: floor at 0.15 penalty",
            canonical_key="REMOVE:target_log_transform:_:_",
            programs=["p1"],
        )
        b = _idea(
            "REMOVE log target: hurts at low fitness",
            canonical_key="REMOVE:target_log_transform:_:_",
            programs=["p2"],
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        # First-wins on description
        assert merged.description == a.description
        # Second's description preserved in aliases
        assert any(
            isinstance(alias, dict) and b.description in str(alias)
            for alias in merged.aliases
        )

    def test_duplicate_canonical_key_appends_motivation(self) -> None:
        bank = IdeaBank()
        a = _idea(
            "REMOVE log target",
            canonical_key="REMOVE:log_target:_:_",
            motivation="first motivation",
        )
        b = _idea(
            "REMOVE log target alt",
            canonical_key="REMOVE:log_target:_:_",
            motivation="second motivation",
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        assert "first motivation" in merged.explanation.entries
        assert "second motivation" in merged.explanation.entries

    def test_different_canonical_keys_inserted_separately(self) -> None:
        bank = IdeaBank()
        a = _idea("ADD log1p_pop", canonical_key="ADD:log1p_pop:_:_")
        b = _idea("ADD room_ratio", canonical_key="ADD:room_ratio:_:_")
        bank.add(a)
        bank.add(b)
        assert len(bank.all_ideas()) == 2

    def test_no_canonical_keyword_falls_through_to_uuid_dedup(self) -> None:
        # When neither idea has a canonical:* keyword, current behavior
        # (UUID reassignment on id collision) must still hold.
        bank = IdeaBank()
        a = Idea(description="no canonical keyword A")
        b = Idea(description="no canonical keyword B")
        b = b.model_copy(update={"id": a.id})  # force collision
        bank.add(a)
        bank.add(b)
        assert len(bank.all_ideas()) == 2

    def test_canonical_dedup_preserves_first_strategy(self) -> None:
        bank = IdeaBank()
        a = Idea(
            description="A",
            strategy="exploitation",
            keywords=["canonical:UPDATE:depth:6:7"],
            programs=["p1"],
        )
        b = Idea(
            description="B",
            strategy="exploration",
            keywords=["canonical:UPDATE:depth:6:7"],
            programs=["p2"],
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        assert merged.strategy == "exploitation"

    def test_canonical_dedup_dedup_programs(self) -> None:
        bank = IdeaBank()
        a = _idea(
            "A",
            canonical_key="UPDATE:depth:6:7",
            programs=["p1", "p2"],
        )
        b = _idea(
            "B",
            canonical_key="UPDATE:depth:6:7",
            programs=["p2", "p3"],
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        assert sorted(merged.programs) == ["p1", "p2", "p3"]
