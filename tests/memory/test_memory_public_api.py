"""Tests for gigaevo.memory public API exports.

Verifies that all public names are importable from the package root.
"""


def test_all_exports_complete():
    """__all__ matches actual exported names."""
    import gigaevo.memory as mem_pkg

    expected = {
        "AmemGamMemory",
        "AnyCard",
        "ConnectedIdea",
        "GigaEvoMemoryBase",
        "LocalMemorySnapshot",
        "MemoryCard",
        "MemoryCardExplanation",
        "ProgramCard",
        "Strategy",
        "normalize_memory_card",
    }
    assert set(mem_pkg.__all__) == expected


def test_import_from_package_root():
    """All public API names importable from gigaevo.memory."""
    from gigaevo.memory import (
        AmemGamMemory,
        AnyCard,
        ConnectedIdea,
        GigaEvoMemoryBase,
        LocalMemorySnapshot,
        MemoryCard,
        MemoryCardExplanation,
        ProgramCard,
        Strategy,
        normalize_memory_card,
    )

    assert AmemGamMemory is not None
    assert MemoryCard is not None
    assert ProgramCard is not None
    assert AnyCard is not None
    assert normalize_memory_card is not None
    assert GigaEvoMemoryBase is not None
    assert ConnectedIdea is not None
    assert MemoryCardExplanation is not None
    assert LocalMemorySnapshot is not None
    assert Strategy is not None


def test_import_from_shared_memory():
    """All names also importable from gigaevo.memory.shared_memory."""
    from gigaevo.memory.shared_memory import (  # noqa: F401
        AmemGamMemory,
        AnyCard,
        ConnectedIdea,
        GigaEvoMemoryBase,
        LocalMemorySnapshot,
        MemoryCard,
        MemoryCardExplanation,
        ProgramCard,
        Strategy,
        normalize_memory_card,
    )


def test_normalize_from_package(tmp_path):
    """normalize_memory_card works when imported from package root."""
    from gigaevo.memory import normalize_memory_card

    card = normalize_memory_card({"id": "c1", "description": "test"})
    assert card is not None
    # Access id — works on both dict and Pydantic model
    card_id = card.id if not isinstance(card, dict) else card["id"]
    assert card_id == "c1"


def test_amem_gam_memory_from_package(tmp_path):
    """AmemGamMemory constructible from package-level import."""
    from gigaevo.memory import AmemGamMemory

    mem = AmemGamMemory(
        checkpoint_path=str(tmp_path / "mem"),
        use_api=False,
        sync_on_init=False,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )
    assert mem is not None
    mem.save_card({"id": "c1", "description": "test"})
    assert mem.get_card("c1") is not None
