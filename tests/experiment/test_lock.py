"""Tests for gigaevo.experiment.lock — Redis locking + atomic write primitives.

The lock module was extracted from ``gigaevo.experiment.manifest`` in the recent
refactor. These tests cover the four package-private symbols directly, using
``fakeredis`` to exercise the real Redis protocol without requiring a daemon.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
import redis as redis_pkg
import yaml

from gigaevo.experiment.lock import (
    _acquire_lock,
    _get_redis,
    _release_lock,
    _write_manifest_atomic,
)


# ---------------------------------------------------------------------------
# _get_redis
# ---------------------------------------------------------------------------


class TestGetRedis:
    def test_uses_env_vars(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "some-host")
        monkeypatch.setenv("REDIS_PORT", "6380")

        captured: dict = {}

        class FakeRedisClient:
            def __init__(self, host, port, db):
                captured["host"] = host
                captured["port"] = port
                captured["db"] = db

            def ping(self):
                return True

        with patch.object(redis_pkg, "Redis", FakeRedisClient):
            _get_redis()

        assert captured == {"host": "some-host", "port": 6380, "db": 0}

    def test_invalid_port_raises_runtimeerror(self, monkeypatch):
        monkeypatch.setenv("REDIS_PORT", "not-an-int")
        with pytest.raises(RuntimeError, match="Invalid REDIS_PORT"):
            _get_redis()

    def test_connection_failure_raises_actionable_runtimeerror(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "nonexistent-host")
        monkeypatch.setenv("REDIS_PORT", "6379")

        class FailingRedis:
            def __init__(self, **_):
                pass

            def ping(self):
                raise redis_pkg.ConnectionError("Connection refused")

        with patch.object(redis_pkg, "Redis", FailingRedis):
            with pytest.raises(RuntimeError, match="Cannot connect to Redis"):
                _get_redis()


# ---------------------------------------------------------------------------
# _acquire_lock / _release_lock
# ---------------------------------------------------------------------------


class TestAcquireLock:
    def test_acquire_success_records_pid(self):
        r = fakeredis.FakeRedis()
        key = _acquire_lock(r, "hover/test-exp", timeout=1.0)

        assert key == "experiments:hover/test-exp:yaml_lock"
        assert r.get(key) == str(os.getpid()).encode()

    def test_timeout_raises_runtimeerror_with_holder(self):
        r = fakeredis.FakeRedis()
        # Pre-populate the lock key
        r.set("experiments:hover/test-exp:yaml_lock", "99999")

        start = time.monotonic()
        with pytest.raises(RuntimeError, match="Could not acquire lock"):
            _acquire_lock(r, "hover/test-exp", timeout=0.5)
        elapsed = time.monotonic() - start

        # Should respect timeout (allow some slop for fakeredis + scheduler)
        assert elapsed < 2.0

    def test_lock_has_ttl(self):
        """Lock must have an expiry to prevent zombie locks."""
        r = fakeredis.FakeRedis()
        _acquire_lock(r, "hover/test-exp", timeout=1.0)

        ttl = r.ttl("experiments:hover/test-exp:yaml_lock")
        assert 0 < ttl <= 30

    def test_acquire_after_release_succeeds(self):
        r = fakeredis.FakeRedis()
        key = _acquire_lock(r, "hover/test-exp", timeout=1.0)
        _release_lock(r, key)

        # Should be acquirable again
        key2 = _acquire_lock(r, "hover/test-exp", timeout=1.0)
        assert key2 == key


class TestReleaseLock:
    def test_deletes_key(self):
        r = fakeredis.FakeRedis()
        r.set("experiments:hover/test-exp:yaml_lock", "12345")
        _release_lock(r, "experiments:hover/test-exp:yaml_lock")
        assert r.get("experiments:hover/test-exp:yaml_lock") is None

    def test_is_idempotent(self):
        r = fakeredis.FakeRedis()
        # Second release on a non-existent key should not raise
        _release_lock(r, "experiments:hover/test-exp:yaml_lock")
        _release_lock(r, "experiments:hover/test-exp:yaml_lock")


# ---------------------------------------------------------------------------
# _write_manifest_atomic
# ---------------------------------------------------------------------------


class TestWriteManifestAtomic:
    def test_writes_yaml_round_trip(self, tmp_path):
        target = tmp_path / "experiment.yaml"
        data = {"schema_version": 1, "experiment": {"name": "hover/foo"}}

        _write_manifest_atomic(target, data)

        assert target.exists()
        loaded = yaml.safe_load(target.read_text())
        assert loaded == data

    def test_tmp_file_cleaned_up_on_success(self, tmp_path):
        target = tmp_path / "experiment.yaml"
        _write_manifest_atomic(target, {"key": "value"})

        assert not target.with_suffix(".yaml.tmp").exists()

    def test_preserves_key_order(self, tmp_path):
        """Manifest writes use sort_keys=False to preserve authored ordering."""
        target = tmp_path / "experiment.yaml"
        data = {"zebra": 1, "alpha": 2, "middle": 3}

        _write_manifest_atomic(target, data)

        # Read raw text, check order is preserved
        text = target.read_text()
        assert text.index("zebra") < text.index("alpha") < text.index("middle")

    def test_survives_rename_failure_without_corrupting_original(self, tmp_path):
        """If rename fails, the original file (if any) must remain intact."""
        target = tmp_path / "experiment.yaml"
        # Seed the target with known-good content
        target.write_text("original: value\n")
        original = target.read_text()

        # Simulate rename failure
        with patch.object(Path, "rename", side_effect=OSError("simulated")):
            with pytest.raises(OSError):
                _write_manifest_atomic(target, {"new": "content"})

        # Original content is preserved
        assert target.read_text() == original
