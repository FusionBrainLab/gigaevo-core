"""Cohn spherical-codes catalogue loader: the 90-config (d, N) benchmark set.

Mirrors Table 7 of the ImprovEvolve paper (Appendix E.3): 90 configurations,
10 per dimension d in [8, 16], N drawn from [26, 1021]. Each entry is the
best-known code from Henry Cohn's online catalogue
(https://cohn.mit.edu/spherical-codes/, mirror https://spherical-codes.org),
downloaded into cohn_cache/ on first use. Cache contents are generated data and
are intentionally not committed.

mu(X) = max_{i<j} <x_i, x_j>  (signed maximum inner product; lower is better).
mu_Cohn is computed LIVE from the frozen coordinates, never transcribed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "cohn_cache"
_CATALOGUE_URL = "https://spherical-codes.org/data/{d}/{n}"
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# 90 configurations (d, N): 10 per dimension d in [8, 16] (paper Table 7 set).
CONFIGS: tuple[tuple[int, int], ...] = (
    (8, 26), (8, 51), (8, 86), (8, 283), (8, 322), (8, 528), (8, 656), (8, 669), (8, 835), (8, 873),
    (9, 32), (9, 44), (9, 521), (9, 625), (9, 750), (9, 786), (9, 838), (9, 869), (9, 880), (9, 886),
    (10, 91), (10, 136), (10, 271), (10, 314), (10, 324), (10, 785), (10, 827), (10, 862), (10, 868), (10, 912),
    (11, 84), (11, 268), (11, 352), (11, 373), (11, 407), (11, 521), (11, 590), (11, 613), (11, 692), (11, 740),
    (12, 95), (12, 240), (12, 331), (12, 343), (12, 416), (12, 642), (12, 734), (12, 855), (12, 899), (12, 913),
    (13, 69), (13, 90), (13, 162), (13, 244), (13, 351), (13, 592), (13, 691), (13, 725), (13, 818), (13, 882),
    (14, 102), (14, 211), (14, 212), (14, 642), (14, 692), (14, 913), (14, 922), (14, 925), (14, 970), (14, 1021),
    (15, 208), (15, 243), (15, 380), (15, 425), (15, 436), (15, 456), (15, 491), (15, 514), (15, 517), (15, 527),
    (16, 88), (16, 126), (16, 160), (16, 197), (16, 296), (16, 341), (16, 534), (16, 688), (16, 770), (16, 807),
)

_LOAD_CACHE: dict[tuple[int, int], tuple[np.ndarray, float]] = {}


def _packing_path(d: int, n: int) -> Path:
    return CACHE_DIR / f"packing_{d}_{n}.txt"


def _download(d: int, n: int) -> str:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy else urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(
        _CATALOGUE_URL.format(d=d, n=n), headers={"User-Agent": "gigaevo-spherical-general"}
    )
    return opener.open(req, timeout=60).read().decode("utf-8", "replace")


def _parse(text: str, d: int, n: int) -> np.ndarray:
    vals = _FLOAT_RE.findall(text)
    arr = np.asarray([float(v) for v in vals], dtype=np.float64)
    if arr.size != d * n:
        raise ValueError(f"packing (d={d}, N={n}): parsed {arr.size} floats, expected {d * n}")
    return arr.reshape(n, d)


def _renormalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def mu(X: np.ndarray) -> float:
    """Signed maximum pairwise inner product max_{i<j} <x_i, x_j>."""
    A = np.asarray(X, dtype=np.float64)
    G = A @ A.T
    np.fill_diagonal(G, -np.inf)
    return float(G.max())


def load(d: int, n: int) -> np.ndarray:
    """Downloaded/cached (N, d) Cohn config, renormalized to unit rows in float64.

    Reads cohn_cache/; downloads and caches on miss (needs HTTPS_PROXY).
    """
    path = _packing_path(d, n)
    if path.exists():
        text = path.read_text()
    else:
        text = _download(d, n)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    X = _parse(text, d, n)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"packing (d={d}, N={n}) has non-finite entries")
    return _renormalize(X)


def load_frozen(d: int, n: int) -> tuple[np.ndarray, float]:
    """Memoized (copy-of-array, mu_Cohn). The copy is safe for the caller to mutate."""
    key = (d, n)
    if key not in _LOAD_CACHE:
        X = load(d, n)
        _LOAD_CACHE[key] = (X, mu(X))
    X, m = _LOAD_CACHE[key]
    return X.copy(), m


def _stratified(by_index: tuple[int, ...]) -> list[tuple[int, int]]:
    by_d: dict[int, list[int]] = {}
    for d, n in CONFIGS:
        by_d.setdefault(d, []).append(n)
    out: list[tuple[int, int]] = []
    for d in sorted(by_d):
        ns = sorted(by_d[d])
        for i in by_index:
            if i < len(ns):
                out.append((d, ns[i]))
    return out


# panel: 14 high-headroom configs for evolution, chosen from the R=1 full90 headroom
# maps of the paper baselines (E7/E8) -> the most-improvable (d, N) per dimension, with
# all of d in [8, 16] represented (d=14 is near-Cohn everywhere; its single best config is
# kept for coverage). The headline stays full90; the panel only steers where evolution
# spends per-mutant gradient. Provenance: select_panel.py over the R=1 maps; see
# docs/superpowers/specs/2026-06-17-spherical-codes-general-improver-design.md.
PANEL: tuple[tuple[int, int], ...] = (
    (8, 669),
    (9, 521), (9, 625),
    (10, 785),
    (11, 613), (11, 692),
    (12, 343),
    (13, 244), (13, 818),
    (14, 692),
    (15, 380), (15, 527),
    (16, 296), (16, 807),
)
# smoke: smallest N in the first 6 dimensions -> 6 fast configs.
SMOKE: tuple[tuple[int, int], ...] = tuple(_stratified((0,))[:6])


def eval_configs(name: str) -> list[tuple[int, int]]:
    sets = {"full90": list(CONFIGS), "panel": list(PANEL), "smoke": list(SMOKE)}
    if name not in sets:
        raise ValueError(f"unknown eval set {name!r}; choose from {sorted(sets)}")
    return sets[name]
