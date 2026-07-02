"""Fixtures for the storage layer: deterministic embeddings + test-data factories.

Every test in this directory runs against real Chroma in a tmpdir, but with a
token-hash embedding function instead of a sentence-transformer: identical
text embeds to distance 0, disjoint tokens embed (near-)orthogonally, and the
suite stays fast and network-free.
"""

from __future__ import annotations

import hashlib
import math

from chromadb.api.types import EmbeddingFunction
import pytest

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.config import StoreConfig


class FakeEmbeddingFunction(EmbeddingFunction):
    embedded: list[str] = []

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name

    @staticmethod
    def name() -> str:
        return "fake-token-hash"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> FakeEmbeddingFunction:
        return FakeEmbeddingFunction(**config)

    def __call__(self, input):
        type(self).embedded.extend(input)
        return [self._embed(text) for text in input]

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * 64
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % 64
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()
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
