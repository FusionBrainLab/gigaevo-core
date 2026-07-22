"""Frozen semantic features for the embedding-informed card prior.

The card prior wants a low-dimensional, replayable summary of each card's
meaning. This module supplies two collaborators the posterior never has to
know the internals of:

* ``FrozenProjection`` — a seeded Johnson-Lindenstrauss map from a raw sentence
  embedding down to ``output_dim`` features. It is a pure function of its
  version stamp and dimensions, so a serialized reduced vector can always be
  reconstructed from the card text and the version alone.
* ``CardEmbedder`` — the seam that turns card text into a raw embedding. The
  live path reuses the retrieval model; tests substitute a deterministic fake.

Only reduced vectors ever reach ``FeatureSpace``; the heavy sentence-transformer
dependency stays behind ``CardEmbedder`` and is never imported by the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import hashlib
from typing import TYPE_CHECKING

import numpy as np

from gigaevo.memory_v2.models import CardSnapshot

if TYPE_CHECKING:
    from gigaevo.memory_v2.features import EmbeddingPriorConfig


class FrozenProjection:
    """Deterministic Johnson-Lindenstrauss projection ``input_dim -> output_dim``.

    The Gaussian projection matrix is seeded from ``version`` and the two
    dimensions, so two independently constructed projections with the same
    stamp are byte-for-byte identical and a version bump is a real change.
    Entries are drawn ``N(0, 1 / output_dim)`` so the map approximately
    preserves Euclidean geometry.
    """

    def __init__(self, *, version: str, input_dim: int, output_dim: int) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"projection dimensions must be positive, got "
                f"input_dim={input_dim}, output_dim={output_dim}"
            )
        self.version = version
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._matrix = self._build_matrix()

    def _seed(self) -> int:
        stamp = f"{self.version}:{self.input_dim}:{self.output_dim}".encode()
        return int.from_bytes(hashlib.sha256(stamp).digest()[:8], "big")

    def _build_matrix(self) -> np.ndarray:
        rng = np.random.default_rng(self._seed())
        scale = 1.0 / np.sqrt(self.output_dim)
        matrix = rng.normal(scale=scale, size=(self.input_dim, self.output_dim))
        return np.ascontiguousarray(matrix, dtype=float)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def project(self, vectors: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(vectors, dtype=float)
        if array.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected trailing dimension {self.input_dim}, got {array.shape}"
            )
        return array @ self._matrix


class CardEmbedder(ABC):
    """Turns card text into a raw sentence embedding.

    Implementations own the heavy embedding model; the posterior consumes only
    the projected features and never depends on this class directly.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the raw embedding vectors this embedder returns."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed each text, returning a ``(len(texts), dimension)`` array."""


class SentenceTransformerCardEmbedder(CardEmbedder):
    """Raw card embeddings from the same model the retrieval index uses.

    The sentence-transformer is loaded lazily on first use so importing this
    module stays cheap and the model is only paid for when the prior is active.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._embedding_fn: object | None = None
        self._dimension: int | None = None

    def _function(self) -> object:
        if self._embedding_fn is None:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            from gigaevo.memory.storage.hf_cache import ensure_writable_hf_cache

            ensure_writable_hf_cache()
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._model_name
            )
        return self._embedding_fn

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.embed(("",)).shape[1])
        return self._dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=float)
        vectors = self._function()(list(texts))  # type: ignore[operator]
        return np.asarray(vectors, dtype=float)


class CardEmbeddingReducer:
    """Map candidate cards to their frozen, reduced embedding features.

    The reduced vector is a pure function of the embedded text, so identical
    payloads are embedded once and cached: fitting re-runs over overlapping
    candidate sets never re-pay the sentence-transformer for a seen card. The
    returned mapping is keyed by ``bank_card_id`` — the arm the posterior sees.
    """

    def __init__(self, *, embedder: CardEmbedder, projection: FrozenProjection) -> None:
        self._embedder = embedder
        self._projection = projection
        self._cache: dict[str, np.ndarray] = {}

    def reduce(self, cards: Sequence[CardSnapshot]) -> dict[str, np.ndarray]:
        pending = [card.payload for card in cards if card.payload not in self._cache]
        unseen = list(dict.fromkeys(pending))
        if unseen:
            raw = _unit_normalize(self._embedder.embed(unseen))
            reduced = self._projection.project(raw)
            for text, vector in zip(unseen, reduced, strict=True):
                self._cache[text] = np.ascontiguousarray(vector, dtype=float)
        return {card.bank_card_id: self._cache[card.payload] for card in cards}


def build_embedding_reducer(
    prior: EmbeddingPriorConfig | None,
    card_embedder: CardEmbedder | None,
) -> CardEmbeddingReducer | None:
    """Build the reducer both memory-v2 seams share, or ``None`` when disabled.

    The read (provider) and write (retirement) seams fit the same feature
    config, so both must agree on whether the prior is active. Returning ``None``
    for a disabled prior keeps the control byte-identical; raising when the prior
    is on but no embedder was wired fails loud instead of silently degrading.
    """

    if prior is None:
        return None
    if card_embedder is None:
        raise ValueError(
            "memory-v2 embedding_prior is configured but no card_embedder was "
            "provided to build its projected card features"
        )
    projection = FrozenProjection(
        version=prior.projection_version,
        input_dim=prior.raw_dimension,
        output_dim=prior.dimension,
    )
    return CardEmbeddingReducer(embedder=card_embedder, projection=projection)


def _unit_normalize(rows: np.ndarray) -> np.ndarray:
    # The induced per-card prior variance scales with ||phi||^2, so the effect
    # prior width would otherwise track the embedder's arbitrary vector norm.
    # Row-normalizing the raw embedding pins that width to the card's semantic
    # direction alone (empty-text zero vectors stay zero).
    array = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms > 0.0, norms, 1.0)
