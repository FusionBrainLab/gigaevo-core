from __future__ import annotations

from problems.dag_tabm import gpu_pool


def test_random_gpu_leases_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_pool, "_cuda_device_count", lambda: 2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GIGAEVO_TABM_DEVICE", raising=False)
    monkeypatch.delenv("GIGAEVO_TABM_GPU_DEVICES", raising=False)
    monkeypatch.setenv("GIGAEVO_TABM_GPU_LOCK_DIR", str(tmp_path))

    with gpu_pool.random_gpu_lease() as first:
        with gpu_pool.random_gpu_lease() as second:
            assert {first.logical_index, second.logical_index} == {0, 1}

    with gpu_pool.random_gpu_lease() as released:
        assert released.logical_index in {0, 1}


def test_cpu_override_needs_no_cuda(monkeypatch):
    monkeypatch.setenv("GIGAEVO_TABM_DEVICE", "cpu")
    monkeypatch.setattr(
        gpu_pool,
        "_cuda_device_count",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA should not be inspected")),
    )

    with gpu_pool.random_gpu_lease() as lease:
        assert lease.device == "cpu"
        assert lease.logical_index is None
