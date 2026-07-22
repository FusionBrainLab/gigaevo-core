"""Cross-process randomized GPU leasing for the TabM evaluator.

Each validation process sees the same CUDA devices.  A small ``flock``-based
lease prevents independently scheduled candidates from choosing the same GPU,
while shuffling the probe order avoids always preferring device zero.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

_DEVICE_ENV = "GIGAEVO_TABM_DEVICE"
_DEVICES_ENV = "GIGAEVO_TABM_GPU_DEVICES"
_LOCK_DIR_ENV = "GIGAEVO_TABM_GPU_LOCK_DIR"
_LOCK_TIMEOUT_ENV = "GIGAEVO_TABM_GPU_LOCK_TIMEOUT"
_DEFAULT_LOCK_DIR = "/tmp/gigaevo-tabm-gpu-leases"


@dataclass(frozen=True)
class GpuLease:
    """A leased torch device and its stable lock identity."""

    device: str
    logical_index: int | None
    visible_token: str | None


def _cuda_device_count() -> int:
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _visible_tokens(count: int) -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value and value.strip() not in {"", "-1"}:
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if len(tokens) >= count:
            return tokens[:count]
    return [str(index) for index in range(count)]


def _allowed_indices(count: int) -> list[int]:
    requested = os.environ.get(_DEVICES_ENV)
    if not requested:
        return list(range(count))
    try:
        indices = [int(value.strip()) for value in requested.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{_DEVICES_ENV} must be a comma-separated list of logical CUDA indices"
        ) from exc
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"{_DEVICES_ENV} must contain unique CUDA indices")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(
            f"{_DEVICES_ENV}={requested!r} is outside the visible CUDA range [0, {count})"
        )
    return indices


def _lock_path(lock_dir: Path, token: str) -> Path:
    digest = hashlib.sha256(token.encode()).hexdigest()[:16]
    return lock_dir / f"gpu-{digest}.lock"


@contextmanager
def random_gpu_lease() -> Iterator[GpuLease]:
    """Lease one available GPU, randomized and exclusive across processes.

    ``GIGAEVO_TABM_DEVICE=cpu`` bypasses CUDA for tests.  A concrete
    ``cuda:N`` value restricts acquisition to that logical device.  Otherwise
    all visible devices (or ``GIGAEVO_TABM_GPU_DEVICES``) participate.
    Advisory locks are released automatically if a worker exits unexpectedly.
    """

    forced = os.environ.get(_DEVICE_ENV, "auto").strip().lower()
    if forced == "cpu":
        yield GpuLease("cpu", None, None)
        return

    count = _cuda_device_count()
    if count < 1:
        raise RuntimeError(
            "TabM evaluation requires CUDA; set GIGAEVO_TABM_DEVICE=cpu only for "
            "small tests"
        )
    tokens = _visible_tokens(count)
    indices = _allowed_indices(count)
    if forced != "auto":
        if not forced.startswith("cuda:"):
            raise ValueError(f"{_DEVICE_ENV} must be 'auto', 'cpu', or 'cuda:N'")
        try:
            forced_index = int(forced.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid {_DEVICE_ENV} value: {forced!r}") from exc
        if forced_index not in indices:
            raise ValueError(
                f"forced device {forced!r} is not in the allowed logical devices {indices}"
            )
        indices = [forced_index]

    lock_dir = Path(os.environ.get(_LOCK_DIR_ENV, _DEFAULT_LOCK_DIR))
    lock_dir.mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get(_LOCK_TIMEOUT_ENV, "3600"))
    if timeout <= 0:
        raise ValueError(f"{_LOCK_TIMEOUT_ENV} must be positive")

    rng = random.SystemRandom()
    deadline = time.monotonic() + timeout
    while True:
        probe_order = list(indices)
        rng.shuffle(probe_order)
        for index in probe_order:
            token = tokens[index]
            path = _lock_path(lock_dir, token)
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue

            metadata = json.dumps(
                {
                    "pid": os.getpid(),
                    "logical_index": index,
                    "visible_token": token,
                    "acquired_at": time.time(),
                }
            ).encode()
            os.ftruncate(descriptor, 0)
            os.write(descriptor, metadata)
            try:
                yield GpuLease(f"cuda:{index}", index, token)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"no TabM GPU lease became available within {timeout:g} seconds"
            )
        time.sleep(rng.uniform(0.05, 0.20))
