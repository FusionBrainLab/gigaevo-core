"""Deterministic token-hash embedding double for real-Chroma tests.

Identical text embeds to distance 0, disjoint tokens embed (near-)orthogonally,
and suites stay fast and network-free.
"""

from __future__ import annotations

import hashlib
import math

from chromadb.api.types import EmbeddingFunction


class FakeEmbeddingFunction(EmbeddingFunction):
    embedded: list[str] = []
    batches: list[list[str]] = []

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
        batch = list(input)
        type(self).embedded.extend(batch)
        type(self).batches.append(batch)
        return [self._embed(text) for text in batch]

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * 64
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % 64
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
