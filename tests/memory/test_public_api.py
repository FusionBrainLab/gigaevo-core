"""Tests for gigaevo.memory public API exports.

Verifies the lightweight public names are importable from the package root.
The heavy backend nodes ``AmemGamMemory`` and ``MemorySystem`` are deliberately
NOT re-exported from the root (they drag the Redis/Chroma/engine closure); they
live at their submodules so ``import gigaevo`` stays light for leaf tools.
"""


def test_all_exports_complete():
    """__all__ matches actual exported names."""
    import gigaevo.memory as mem_pkg

    expected = {
        "ApiConfig",
        "AnyCard",
        "GamConfig",
        "GigaEvoMemoryBase",
        "MemoryCard",
        "MemoryConfig",
        "ProgramCard",
        "Strategy",
        "normalize_memory_card",
    }
    assert set(mem_pkg.__all__) == expected


def test_import_from_package_root():
    """All lightweight public API names importable from gigaevo.memory."""
    from gigaevo.memory import (
        AnyCard,
        ApiConfig,
        GamConfig,
        GigaEvoMemoryBase,
        MemoryCard,
        MemoryConfig,
        ProgramCard,
        Strategy,
        normalize_memory_card,
    )

    assert ApiConfig is not None
    assert MemoryCard is not None
    assert ProgramCard is not None
    assert AnyCard is not None
    assert normalize_memory_card is not None
    assert GamConfig is not None
    assert GigaEvoMemoryBase is not None
    assert MemoryConfig is not None
    assert Strategy is not None


def test_import_from_shared_memory():
    """Lightweight names also importable from gigaevo.memory.shared_memory."""
    from gigaevo.memory.shared_memory import (  # noqa: F401
        AnyCard,
        GigaEvoMemoryBase,
        MemoryCard,
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
    """AmemGamMemory constructible from its submodule with MemoryConfig."""
    from gigaevo.memory.shared_memory.memory import AmemGamMemory
    from gigaevo.memory.shared_memory.memory_config import MemoryConfig

    cfg = MemoryConfig(checkpoint_path=tmp_path / "mem")
    mem = AmemGamMemory(config=cfg)
    assert mem is not None
    mem.save_card({"id": "c1", "description": "test"})
    assert mem.get_card("c1") is not None
