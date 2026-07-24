from __future__ import annotations

import multiprocessing
import os

import pytest

from problems.tabular_dag_baselines import gpu_pool


def _hold_campaign_gpu(
    lock_dir: str,
    entered: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    os.environ["GIGAEVO_EXEC_POOL_ID"] = "shared-campaign"
    os.environ["GIGAEVO_TABULAR_DAG_GPU_LOCK_DIR"] = lock_dir
    gpu_pool._cuda_device_count = lambda: 2
    with gpu_pool.random_gpu_lease("tabm"):
        entered.set()
        assert release.wait(5)


def test_each_gpu_allows_two_concurrent_leases(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_pool, "_cuda_device_count", lambda: 1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GIGAEVO_EXEC_POOL_ID", raising=False)
    monkeypatch.delenv("GIGAEVO_TABM_DEVICE", raising=False)
    monkeypatch.delenv("GIGAEVO_TABM_GPU_DEVICES", raising=False)
    monkeypatch.delenv("GIGAEVO_TABM_GPU_LOCK_DIR", raising=False)
    monkeypatch.delenv("GIGAEVO_REALMLP_GPU_LOCK_DIR", raising=False)
    monkeypatch.delenv("GIGAEVO_TABICL_GPU_LOCK_DIR", raising=False)
    monkeypatch.setenv("GIGAEVO_TABULAR_DAG_GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("GIGAEVO_TABULAR_DAG_GPU_LOCK_TIMEOUT", "0.1")

    with gpu_pool.random_gpu_lease("tabm") as first:
        with gpu_pool.random_gpu_lease("realmlp") as second:
            assert first.logical_index == second.logical_index == 0
            with pytest.raises(TimeoutError):
                with gpu_pool.random_gpu_lease("tabicl"):
                    pass

    with gpu_pool.random_gpu_lease("tabicl") as released:
        assert released.logical_index == 0


def test_campaign_gpu_concurrency_is_capped_at_two(tmp_path):
    context = multiprocessing.get_context("fork")
    first_entered = context.Event()
    second_entered = context.Event()
    third_entered = context.Event()
    release_first = context.Event()
    release_second = context.Event()
    release_third = context.Event()
    first = context.Process(
        target=_hold_campaign_gpu,
        args=(str(tmp_path), first_entered, release_first),
    )
    second = context.Process(
        target=_hold_campaign_gpu,
        args=(str(tmp_path), second_entered, release_second),
    )
    third = context.Process(
        target=_hold_campaign_gpu,
        args=(str(tmp_path), third_entered, release_third),
    )

    first.start()
    assert first_entered.wait(2)
    second.start()
    assert second_entered.wait(2)
    third.start()
    assert not third_entered.wait(0.25)
    release_first.set()
    assert third_entered.wait(2)
    release_second.set()
    release_third.set()
    first.join(2)
    second.join(2)
    third.join(2)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert third.exitcode == 0


def test_cpu_override_needs_no_cuda(monkeypatch):
    monkeypatch.setenv("GIGAEVO_TABM_DEVICE", "cpu")
    monkeypatch.setattr(
        gpu_pool,
        "_cuda_device_count",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA should not be inspected")),
    )

    with gpu_pool.random_gpu_lease("tabm") as lease:
        assert lease.device == "cpu"
        assert lease.logical_index is None
