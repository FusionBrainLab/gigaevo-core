"""Independent ACI verifier (runbook §7.3) — numpy only, no JAX.

C(f) = ||f*f||_2^2 / (||f*f||_1 · ||f*f||_inf) for nonnegative f on a 1-D grid.

Metric convention calibrated 2026-07-18 against both published anchors: the
canonical validate_f formulas (interval-rule L2^2 over [-0.5, 0.5] with
zero-padded endpoints, L1 = sum|conv|/(len+1), Linf = max|conv|) applied to the
full FFT autoconvolution reproduce the stored adapted record
C=0.9625839725411991 to ~1e-12 and the AlphaEvolve start's 0.96102.
Plain discrete sums do NOT match the published numbers — never use them here.

Sensitivity checks per §7.3: alternate FFT padding (2x next-pow2) and a
direct-convolution cross-check on a reduced-resolution copy. Verified
artifacts go to results/verified/aci/<sha256>.{json,npz} via write_verified.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np

MIN_RESOLUTION = 100
NONNEG_TOL = -1e-6
REDUCED_N = 2049


def canonical_sha256(f: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(f, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def autoconvolution(f: np.ndarray, pad_factor: int = 1) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    m = 2 * len(f) - 1
    nfft = (1 << (m - 1).bit_length()) * pad_factor
    fft = np.fft.rfft(f, n=nfft)
    return np.fft.irfft(fft * fft, n=nfft)[:m]


def metrics_from_conv(conv: np.ndarray) -> dict:
    m = len(conv)
    x = np.linspace(-0.5, 0.5, m + 2)
    dx = np.diff(x)
    y = np.concatenate(([0.0], conv, [0.0]))
    y1, y2 = y[:-1], y[1:]
    l2_sq = float(np.sum((dx / 3.0) * (y1**2 + y1 * y2 + y2**2)))
    l1 = float(np.sum(np.abs(conv)) / (m + 1))
    linf = float(np.max(np.abs(conv)))
    denom = l1 * linf
    c = l2_sq / denom if denom > 0.0 else float("nan")
    return {"l2_sq": l2_sq, "l1": l1, "linf": linf, "c": c}


def verify(f: np.ndarray, reduced_check: bool = True) -> dict:
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"f must be 1-D, got shape {f.shape}")
    n = len(f)
    if n < MIN_RESOLUTION:
        raise ValueError(f"resolution {n} < hard floor {MIN_RESOLUTION}")
    if not np.all(np.isfinite(f)):
        raise ValueError("f contains non-finite values")
    if float(f.min()) < NONNEG_TOL:
        raise ValueError(f"negative beyond tolerance: min={f.min()}")
    if float(f.max()) <= 0.0:
        raise ValueError("trivial (all-zero) f rejected")
    f = np.maximum(f, 0.0)

    report = {"resolution": n, "sha256": canonical_sha256(f)}
    report.update(metrics_from_conv(autoconvolution(f)))
    alt = metrics_from_conv(autoconvolution(f, pad_factor=2))
    report["c_alt_padding"] = alt["c"]
    report["padding_sensitivity_abs"] = abs(alt["c"] - report["c"])

    if reduced_check:
        if n <= REDUCED_N:
            fr = f
        else:
            fr = np.interp(np.linspace(0.0, n - 1.0, REDUCED_N), np.arange(n), f)
        c_direct = metrics_from_conv(np.convolve(fr, fr, mode="full"))["c"]
        c_fft = metrics_from_conv(autoconvolution(fr))["c"]
        report["reduced_fft_vs_direct_abs_diff"] = abs(c_direct - c_fft)
    return report


def write_verified(f: np.ndarray, out_dir: Path, extra: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = verify(f)
    sha = report["sha256"]
    npz = out_dir / f"{sha}.npz"
    np.savez_compressed(npz, f=np.asarray(f, dtype=np.float64))
    payload = {
        "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "array_file": npz.name,
        **report,
    }
    if extra:
        payload.update(extra)
    path = out_dir / f"{sha}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path
