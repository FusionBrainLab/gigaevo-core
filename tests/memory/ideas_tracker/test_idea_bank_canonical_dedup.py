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
        assert any(alias.description == b.description for alias in merged.aliases)
        assert merged.aliases[0].key == f"{a.id}-canonical-merge"
        assert merged.aliases[0].programs == ["p2"]

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

    def test_merge_bumps_last_generation(self) -> None:
        bank = IdeaBank()
        a = _idea("A", canonical_key="UPDATE:depth:6:7", programs=["p1"]).model_copy(
            update={"last_generation": 5}
        )
        b = _idea("B", canonical_key="UPDATE:depth:6:7", programs=["p2"]).model_copy(
            update={"last_generation": 12}
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        assert merged.last_generation == 12

    def test_merge_does_not_regress_last_generation(self) -> None:
        bank = IdeaBank()
        a = _idea("A", canonical_key="UPDATE:depth:6:7").model_copy(
            update={"last_generation": 12}
        )
        b = _idea("B", canonical_key="UPDATE:depth:6:7").model_copy(
            update={"last_generation": 5}
        )
        bank.add(a)
        bank.add(b)
        assert bank.all_ideas()[0].last_generation == 12

    def test_merge_unions_topical_keywords(self) -> None:
        bank = IdeaBank()
        a = _idea(
            "A",
            canonical_key="UPDATE:depth:6:7",
            extra_keywords=["regularization"],
        )
        b = _idea(
            "B",
            canonical_key="UPDATE:depth:6:7",
            extra_keywords=["leaf-variance", "regularization"],
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        assert "regularization" in merged.keywords
        assert "leaf-variance" in merged.keywords
        canonical_count = sum(
            1 for kw in merged.keywords if kw.startswith("canonical:")
        )
        assert canonical_count == 1

    def test_merge_keeps_existing_machine_tags_only(self) -> None:
        """Contradictory verification tags from the merged-away idea must not
        ride in: the incumbent's machine tags are the card's verdict."""
        bank = IdeaBank()
        a = _idea(
            "A",
            canonical_key="UPDATE:depth:6:7",
            extra_keywords=["verified:true"],
        )
        b = _idea(
            "B",
            canonical_key="UPDATE:depth:6:7",
            extra_keywords=["verified:false", "mechanism_unverified:true"],
        )
        bank.add(a)
        bank.add(b)
        merged = bank.all_ideas()[0]
        assert "verified:true" in merged.keywords
        assert "verified:false" not in merged.keywords
        assert "mechanism_unverified:true" not in merged.keywords

    def test_repeated_merges_keep_alias_keys_unique(self) -> None:
        bank = IdeaBank()
        for label in ("A", "B", "C"):
            bank.add(
                _idea(label, canonical_key="UPDATE:depth:6:7", programs=[f"p-{label}"])
            )
        merged = bank.all_ideas()[0]
        keys = [alias.key for alias in merged.aliases]
        assert len(keys) == 2
        assert len(set(keys)) == 2

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
