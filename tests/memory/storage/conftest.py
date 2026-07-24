"""Fixtures for the storage layer: deterministic embeddings + test-data factories.

Every test in this directory runs against real in-memory Chroma, but with a
token-hash embedding function instead of a sentence-transformer: identical text
embeds to distance 0, disjoint tokens embed (near-)orthogonally, and the suite
stays fast and network-free.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.config import StoreConfig
from tests.fakes.embedding import FakeEmbeddingFunction


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()
    FakeEmbeddingFunction.batches.clear()
    return FakeEmbeddingFunction


@pytest.fixture
def make_card():
    counter = iter(range(10_000))

    def _make_card(**overrides) -> Card:
        n = next(counter)
        params = {
            "id": f"mem-test{n:04d}",
            "kind": CardKind.INSIGHT,
            "description": f"idea-{n} exploits problem structure",
            "explanation_summary": f"works because of invariant-{n}",
        }
        params.update(overrides)
        return Card(**params)

    return _make_card


@pytest.fixture
def make_store_config(tmp_path):
    def _make(**overrides) -> StoreConfig:
        params = {"path": tmp_path / "store"}
        params.update(overrides)
        return StoreConfig(**params)

    return _make
