"""Synthetic noisy-evaluation problem for smoke-testing the paired archive gate.

Per-sample score = exp(-2|pred - target|) + deterministic pseudo-noise seeded
from the prediction vector itself: the same program always re-evaluates to the
same vector, while distinct programs draw independent noise — mimicking hover's
single-eval variance (sigma_mean ~ 0.01 at N=64, NOISE_STD=0.08).

Returns (metrics, artifact) with the per-sample vector under the
"_program_metadata" reserved namespace, which standard pipelines route.
"""

import hashlib

import numpy as np

N_SAMPLES = 64
NOISE_STD = 0.08

X_GRID = np.linspace(-2.0, 2.0, N_SAMPLES)
Y_TRUE = np.sin(2.5 * X_GRID) * np.exp(-0.15 * X_GRID**2) + 0.4 * X_GRID


def validate(data) -> tuple[dict, dict]:
    preds = np.asarray(data, dtype=float)
    if preds.shape != (N_SAMPLES,):
        raise ValueError(f"Expected {N_SAMPLES} predictions, got shape {preds.shape}")
    if not np.all(np.isfinite(preds)):
        raise ValueError("Predictions contain NaN or infinite values")

    base = np.exp(-2.0 * np.abs(preds - Y_TRUE))
    seed = int.from_bytes(hashlib.sha256(preds.tobytes()).digest()[:4], "big")
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD, N_SAMPLES)
    scores = np.clip(base + noise, 0.0, 1.0)

    return {
        "fitness": float(scores.mean()),
        "is_valid": 1.0,
    }, {"_program_metadata": {"per_sample_scores": [float(s) for s in scores]}}
