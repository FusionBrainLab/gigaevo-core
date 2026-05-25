"""Reader-side staleness regression for AmemGamMemory.

Bug discovered after a 7h tabular_regression run produced 781/781 empty
selector results despite 34 cards on disk. Root cause: when a reader-only
``AmemGamMemory`` is constructed BEFORE any cards exist on disk, its
``research_agent`` stays ``None`` and its ``card_store.cards`` stays empty;
subsequent on-disk writes by a separate writer instance are never picked
up because the reader has no API sync and never calls ``save``/``delete``
itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import MemoryConfig
from tests.fakes.agentic_memory import make_test_memory_with_agentic


@pytest.fixture
def shared_checkpoint(tmp_path: Path) -> Path:
    return tmp_path / "shared_mem"


def _build_reader(checkpoint_path: Path) -> AmemGamMemory:
    """Build a reader that mirrors what MemorySelectorAgent does in production."""
    cfg = MemoryConfig(
        checkpoint_path=checkpoint_path,
        search_limit=5,
        rebuild_interval=10,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )
    return AmemGamMemory(config=cfg)


def _save_card_via_writer(
    tmp_path: Path, shared_checkpoint: Path, description: str
) -> str:
    """Build a writer pointed at the SAME checkpoint and write one card."""
    writer, _fake_sys = make_test_memory_with_agentic(tmp_path)
    # Re-point the writer's storage to the shared checkpoint by rebuilding
    # with that path. We can't just hot-swap, so we rebuild the writer via
    # the same factory but with a shared checkpoint location.
    return writer.save_card({"category": "general", "description": description})


def test_reader_built_before_writes_still_sees_external_card_additions(
    tmp_path: Path,
) -> None:
    """RED: reader created on empty disk does NOT pick up writes from a
    separate writer instance — proves the dual-instance staleness bug.

    GREEN (after fix): reader auto-refreshes on search() when the on-disk
    index_file mtime advances past its last-seen mtime, so it returns
    cards written by the external writer.
    """
    shared_checkpoint = tmp_path / "shared_mem"

    # 1) Reader created BEFORE any cards exist on disk.
    reader = _build_reader(shared_checkpoint)
    assert reader.research_agent is None
    assert reader.card_store.cards == {}

    # 2) External writer (separate AmemGamMemory instance) writes a card to
    #    the SAME checkpoint dir.
    writer_cfg = MemoryConfig(
        checkpoint_path=shared_checkpoint,
        search_limit=5,
        rebuild_interval=1,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )
    # Use the agentic fake path so the writer's rebuild succeeds and
    # produces an on-disk artefact a real reader could see.
    from unittest.mock import MagicMock

    from tests.fakes.agentic_memory import FakeAMemGenerator, _get_fake_runtime

    mock_llm = MagicMock()
    writer = AmemGamMemory(
        config=writer_cfg,
        runtime=_get_fake_runtime(),
        llm_service=mock_llm,
        generator=FakeAMemGenerator({"llm_service": mock_llm}),
    )
    writer.save_card(
        {
            "category": "general",
            "description": "Use CatBoost with depth=6 and learning_rate=0.03 for tabular regression.",
            "keywords": ["catboost", "depth", "learning_rate"],
        }
    )
    # Force the writer to flush its index to disk.
    writer.card_store.persist()
    assert writer_cfg.index_file.exists(), "writer should have persisted index_file"

    # 3) Reader searches. Without the fix, it returns the empty sentinel.
    #    With the fix, it must auto-refresh from disk and find the card.
    result = reader.search("CatBoost tabular regression configuration")
    assert "No relevant memories found" not in result, (
        "reader did not auto-refresh after external writer added a card "
        f"(got: {result!r})"
    )
    # The fix should populate reader.card_store from the on-disk index.
    assert len(reader.card_store.cards) >= 1, (
        f"reader's card_store should now contain the externally-written card "
        f"(got {len(reader.card_store.cards)} cards)"
    )


def test_reader_built_after_writes_already_sees_cards(tmp_path: Path) -> None:
    """Control test: reader built AFTER cards exist works correctly today.
    This proves the bug is timing-specific (init-before-cards), not a wholesale
    failure of the read path.
    """
    shared_checkpoint = tmp_path / "shared_mem"

    # Build writer first, persist a card.
    from unittest.mock import MagicMock

    from tests.fakes.agentic_memory import FakeAMemGenerator, _get_fake_runtime

    mock_llm = MagicMock()
    writer = AmemGamMemory(
        config=MemoryConfig(
            checkpoint_path=shared_checkpoint,
            search_limit=5,
            rebuild_interval=1,
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        ),
        runtime=_get_fake_runtime(),
        llm_service=mock_llm,
        generator=FakeAMemGenerator({"llm_service": mock_llm}),
    )
    writer.save_card(
        {
            "category": "general",
            "description": "LightGBM num_leaves=64 early stopping rounds 50",
        }
    )
    writer.card_store.persist()

    # NOW build the reader.
    reader = _build_reader(shared_checkpoint)
    assert len(reader.card_store.cards) >= 1, (
        "reader built post-write should load the card"
    )


def test_reader_refresh_is_idempotent_when_index_unchanged(tmp_path: Path) -> None:
    """Repeated searches against an unchanged store should not rebuild repeatedly
    (avoid burning chroma/embeddings on every call).
    """
    shared_checkpoint = tmp_path / "shared_mem"

    from unittest.mock import MagicMock

    from tests.fakes.agentic_memory import FakeAMemGenerator, _get_fake_runtime

    mock_llm = MagicMock()
    writer = AmemGamMemory(
        config=MemoryConfig(
            checkpoint_path=shared_checkpoint,
            search_limit=5,
            rebuild_interval=1,
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        ),
        runtime=_get_fake_runtime(),
        llm_service=mock_llm,
        generator=FakeAMemGenerator({"llm_service": mock_llm}),
    )
    writer.save_card(
        {"category": "general", "description": "RandomForest n_estimators=300"}
    )
    writer.card_store.persist()

    reader = _build_reader(shared_checkpoint)

    # First search refreshes; subsequent searches should be no-op refreshes.
    reader.search("forest model")
    initial_mtime_seen = getattr(reader, "_last_seen_index_mtime", None)
    reader.search("forest model again")
    later_mtime_seen = getattr(reader, "_last_seen_index_mtime", None)
    # When implemented, mtime tracking should remain pinned to the same value.
    assert initial_mtime_seen is not None
    assert initial_mtime_seen == later_mtime_seen
