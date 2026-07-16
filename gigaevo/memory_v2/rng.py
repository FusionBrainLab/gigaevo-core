"""Order-independent event-keyed random streams."""

from __future__ import annotations

import hashlib

import numpy as np


class EventRNG:
    """Derive an independent PCG64DXSM stream for each decision domain.

    A caller never shares mutable generator state between acquisition, posterior
    Monte Carlo, and the treatment draw.  Adding a draw in one domain therefore
    cannot perturb any other decision during replay.
    """

    def __init__(self, key: str) -> None:
        if not key:
            raise ValueError("EventRNG requires a non-empty key")
        self._key = key

    def generator(self, domain: str, counter: int = 0) -> np.random.Generator:
        if not domain:
            raise ValueError("RNG domain must be non-empty")
        if counter < 0:
            raise ValueError("RNG counter must be non-negative")
        digest = hashlib.sha256(f"{self._key}\0{domain}\0{counter}".encode()).digest()
        entropy = np.frombuffer(digest, dtype=np.uint32).tolist()
        return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))

    def uniform(self, domain: str, counter: int = 0) -> float:
        return float(self.generator(domain, counter).random())
