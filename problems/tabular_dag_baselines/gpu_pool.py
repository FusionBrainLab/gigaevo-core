"""Cross-process randomized GPU leasing shared by neural DAG baselines."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

_GENERIC_PREFIX = "GIGAEVO_TABULAR_DAG_"
_DEFAULT_LOCK_DIR = "/tmp/gigaevo-tabular-dag-gpu-leases"
_CAMPAIGN_GPU_SLOTS = 2
_GPU_SLOTS_PER_DEVICE = 2


@dataclass(frozen=True)
class GpuLease:
    """A leased torch device and its stable lock identity."""

    device: str
    logical_index: int | None
    visible_token: str | None


def _model_prefix(model_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()
    if not normalized:
        raise ValueError("model_name must contain an alphanumeric character")
    return f"GIGAEVO_{normalized}_"


def _setting(model_name: str, key: str, default: str | None = None) -> str | None:
    model_value = os.environ.get(_model_prefix(model_name) + key)
    if model_value is not None:
        return model_value
    return os.environ.get(_GENERIC_PREFIX + key, default)


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


def _allowed_indices(model_name: str, count: int) -> list[int]:
    requested = _setting(model_name, "GPU_DEVICES")
    if not requested:
        return list(range(count))
    setting_name = _model_prefix(model_name) + "GPU_DEVICES"
    try:
        indices = [int(value.strip()) for value in requested.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"{setting_name} must be a comma-separated list of logical CUDA indices"
        ) from exc
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"{setting_name} must contain unique CUDA indices")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(
            f"{setting_name}={requested!r} is outside the visible CUDA range [0, {count})"
        )
    return indices


def _lock_path(lock_dir: Path, token: str, slot: int) -> Path:
    digest = hashlib.sha256(token.encode()).hexdigest()[:16]
    return lock_dir / f"gpu-{digest}-{slot}.lock"


@contextmanager
def _campaign_gpu_gate(
    lock_dir: Path, *, deadline: float, model_name: str, timeout: float
) -> Iterator[None]:
    """Bound the number of workers one campaign can place in the GPU pool."""

    campaign_id = os.environ.get("GIGAEVO_EXEC_POOL_ID")
    if not campaign_id:
        yield
        return

    digest = hashlib.sha256(campaign_id.encode()).hexdigest()[:16]
    rng = random.SystemRandom()
    while True:
        slots = list(range(_CAMPAIGN_GPU_SLOTS))
        rng.shuffle(slots)
        for slot in slots:
            descriptor = os.open(
                lock_dir / f"campaign-{digest}-{slot}.lock",
                os.O_CREAT | os.O_RDWR,
                0o666,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"no {model_name} GPU lease became available within {timeout:g} seconds"
            )
        time.sleep(rng.uniform(0.05, 0.20))


@contextmanager
def random_gpu_lease(model_name: str) -> Iterator[GpuLease]:
    """Lease one GPU for ``model_name``, or honor its explicit CPU override.

    Model-specific variables such as ``GIGAEVO_TABM_DEVICE`` take precedence
    over shared ``GIGAEVO_TABULAR_DAG_*`` settings.  The default lock directory
    is shared by every model so TabM, RealMLP, TabICL, TabPFN, and TabFM all
    honor the same per-device capacity.
    """

    prefix = _model_prefix(model_name)
    forced = (_setting(model_name, "DEVICE", "auto") or "auto").strip().lower()
    if forced == "cpu":
        yield GpuLease("cpu", None, None)
        return

    count = _cuda_device_count()
    if count < 1:
        raise RuntimeError(
            f"{model_name} evaluation requires CUDA; set {prefix}DEVICE=cpu only "
            "for small tests"
        )
    tokens = _visible_tokens(count)
    indices = _allowed_indices(model_name, count)
    if forced != "auto":
        if not forced.startswith("cuda:"):
            raise ValueError(f"{prefix}DEVICE must be 'auto', 'cpu', or 'cuda:N'")
        try:
            forced_index = int(forced.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid {prefix}DEVICE value: {forced!r}") from exc
        if forced_index not in indices:
            raise ValueError(
                f"forced device {forced!r} is not in the allowed logical devices {indices}"
            )
        indices = [forced_index]

    lock_dir = Path(_setting(model_name, "GPU_LOCK_DIR", _DEFAULT_LOCK_DIR) or "")
    lock_dir.mkdir(parents=True, exist_ok=True)
    timeout_name = prefix + "GPU_LOCK_TIMEOUT"
    timeout = float(_setting(model_name, "GPU_LOCK_TIMEOUT", "3600") or "3600")
    if timeout <= 0:
        raise ValueError(f"{timeout_name} must be positive")

    deadline = time.monotonic() + timeout
    with _campaign_gpu_gate(
        lock_dir, deadline=deadline, model_name=model_name, timeout=timeout
    ):
        rng = random.SystemRandom()
        while True:
            probe_order = list(indices)
            rng.shuffle(probe_order)
            for index in probe_order:
                token = tokens[index]
                slots = list(range(_GPU_SLOTS_PER_DEVICE))
                rng.shuffle(slots)
                for slot in slots:
                    descriptor = os.open(
                        _lock_path(lock_dir, token, slot),
                        os.O_CREAT | os.O_RDWR,
                        0o666,
                    )
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(descriptor)
                        continue

                    metadata = json.dumps(
                        {
                            "pid": os.getpid(),
                            "model": model_name,
                            "logical_index": index,
                            "visible_token": token,
                            "slot": slot,
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
                    f"no {model_name} GPU lease became available within "
                    f"{timeout:g} seconds"
                )
            time.sleep(rng.uniform(0.05, 0.20))


def release_cuda(lease: GpuLease) -> None:
    """Release cached tensors after a model's lease-scoped evaluation."""

    if lease.logical_index is None:
        return
    import torch

    gc.collect()
    with torch.cuda.device(lease.logical_index):
        torch.cuda.empty_cache()
