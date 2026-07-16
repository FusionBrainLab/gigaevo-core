"""Tests for DiskMetricsBackend and the MetricsHistoryReader protocol.

The disk backend mirrors RedisMetricsBackend's history contract: same
entry schema (``{"s", "t", "v", "k"}``), same tag sanitization
(``/`` → ``:``, `` `` → ``_``), same LRANGE-style slicing in
``get_history`` — so the live frontier monitor can read either backend
through the shared protocol.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from gigaevo.utils.trackers.backends.disk import DiskMetricsBackend
from gigaevo.utils.trackers.backends.redis import RedisMetricsBackend
from gigaevo.utils.trackers.base import MetricsHistoryReader
from gigaevo.utils.trackers.configs import DiskMetricsConfig, RedisMetricsConfig


def make_backend(tmp_path) -> DiskMetricsBackend:
    backend = DiskMetricsBackend(DiskMetricsConfig(root_dir=tmp_path / "metrics"))
    backend.open()
    return backend


class TestDiskWriteReadRoundtrip:
    def test_scalar_roundtrip(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("program_metrics/valid_frontier_fitness", 0.5, 1, 100.0)
        backend.write_scalar("program_metrics/valid_frontier_fitness", 0.7, 2, 101.0)
        backend.flush()
        history = backend.get_history("program_metrics/valid_frontier_fitness")
        assert history == [
            {"s": 1, "t": 100.0, "v": 0.5, "k": "scalar"},
            {"s": 2, "t": 101.0, "v": 0.7, "k": "scalar"},
        ]
        backend.close()

    def test_unflushed_writes_not_visible(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("m", 1.0, 0, 100.0)
        assert backend.get_history("m") == []
        backend.close()

    def test_close_flushes_buffer(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("m", 1.0, 0, 100.0)
        backend.close()
        assert backend.get_history("m") == [
            {"s": 0, "t": 100.0, "v": 1.0, "k": "scalar"}
        ]

    def test_readable_after_close(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("m", 1.0, 0, 100.0)
        backend.close()
        reopened = DiskMetricsBackend(DiskMetricsConfig(root_dir=tmp_path / "metrics"))
        assert reopened.get_history("m") == [
            {"s": 0, "t": 100.0, "v": 1.0, "k": "scalar"}
        ]

    def test_missing_tag_returns_empty(self, tmp_path):
        backend = make_backend(tmp_path)
        assert backend.get_history("never_written") == []
        backend.close()

    def test_hist_and_text_entries(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_hist("h", [1.0, 2.0], 0, 100.0)
        backend.write_text("t", "hello", 0, 100.0)
        backend.flush()
        assert backend.get_history("h") == [
            {"s": 0, "t": 100.0, "v": [1.0, 2.0], "k": "hist"}
        ]
        assert backend.get_history("t") == [
            {"s": 0, "t": 100.0, "v": "hello", "k": "text"}
        ]
        backend.close()


class TestDiskTagSanitization:
    def test_file_name_matches_redis_key_sanitization(self, tmp_path):
        # Redis: tag "a/b c" → key suffix "a:b_c". Disk mirrors it so the
        # on-disk layout corresponds 1:1 with the Redis history keys.
        backend = make_backend(tmp_path)
        backend.write_scalar("program_metrics/valid frontier", 1.0, 0, 100.0)
        backend.flush()
        backend.close()
        expected = tmp_path / "metrics" / "program_metrics:valid_frontier.jsonl"
        assert expected.exists()

    def test_lists_persisted_metrics(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("program_metrics/frontier", 1.0, 0, 100.0)
        backend.write_text("diagnostics/message", "ok", 0, 100.0)
        backend.flush()

        assert backend.list_metrics() == [
            "diagnostics/message",
            "program_metrics/frontier",
        ]


class TestDiskLiveReads:
    def test_malformed_trailing_line_does_not_hide_valid_history(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("m", 1.0, 0, 100.0)
        backend.flush()
        path = tmp_path / "metrics" / "m.jsonl"
        with path.open("a") as stream:
            stream.write('{"s": 1')

        assert backend.get_history("m") == [
            {"s": 0, "t": 100.0, "v": 1.0, "k": "scalar"}
        ]


class TestDiskLrangeSlicing:
    def fill(self, tmp_path) -> DiskMetricsBackend:
        backend = make_backend(tmp_path)
        for i in range(5):
            backend.write_scalar("m", float(i), i, 100.0 + i)
        backend.flush()
        return backend

    def values(self, entries) -> list[float]:
        return [e["v"] for e in entries]

    def test_full_range(self, tmp_path):
        backend = self.fill(tmp_path)
        assert self.values(backend.get_history("m", 0, -1)) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_inclusive_end(self, tmp_path):
        backend = self.fill(tmp_path)
        assert self.values(backend.get_history("m", 1, 3)) == [1.0, 2.0, 3.0]

    def test_negative_indices(self, tmp_path):
        backend = self.fill(tmp_path)
        assert self.values(backend.get_history("m", -2, -1)) == [3.0, 4.0]

    def test_empty_when_start_past_end(self, tmp_path):
        backend = self.fill(tmp_path)
        assert backend.get_history("m", 4, 2) == []


class TestDiskClearSeries:
    def test_clear_removes_history(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("m", 1.0, 0, 100.0)
        backend.flush()
        backend.clear_series("m")
        assert backend.get_history("m") == []
        backend.close()

    def test_clear_drops_buffered_entries(self, tmp_path):
        backend = make_backend(tmp_path)
        backend.write_scalar("m", 1.0, 0, 100.0)
        backend.clear_series("m")
        backend.flush()
        assert backend.get_history("m") == []
        backend.close()


class TestHistoryReaderProtocol:
    def test_disk_backend_conforms(self, tmp_path):
        backend = DiskMetricsBackend(DiskMetricsConfig(root_dir=tmp_path))
        assert isinstance(backend, MetricsHistoryReader)

    def test_redis_backend_conforms(self):
        backend = RedisMetricsBackend(RedisMetricsConfig())
        assert isinstance(backend, MetricsHistoryReader)


class TestRedisPostCloseRead:
    def test_get_history_uses_ephemeral_client_when_closed(self):
        # run.py renders the final frontier PNG *after* writer.close() — a
        # closed backend must still serve reads via a fresh connection.
        backend = RedisMetricsBackend(RedisMetricsConfig(key_prefix="p:metrics"))
        assert backend._client is None
        fake = MagicMock()
        fake.lrange.return_value = [
            json.dumps({"s": 1, "t": 0.0, "v": 0.5, "k": "scalar"})
        ]
        with patch(
            "gigaevo.utils.trackers.backends.redis.redis.Redis.from_url",
            return_value=fake,
        ) as from_url:
            history = backend.get_history("a/b")
        from_url.assert_called_once()
        fake.lrange.assert_called_once_with("p:metrics:history:a:b", 0, -1)
        assert history == [{"s": 1, "t": 0.0, "v": 0.5, "k": "scalar"}]


class TestDefaultHistoryReader:
    def test_returns_none_when_no_writer_initialized(self, monkeypatch):
        import gigaevo.utils.trackers as trackers

        monkeypatch.setattr(trackers, "_redis_default", None)
        monkeypatch.setattr(trackers, "_disk_default", None)
        assert trackers.get_default_history_reader() is None

    def test_returns_disk_backend_when_initialized(self, tmp_path, monkeypatch):
        import gigaevo.utils.trackers as trackers
        from gigaevo.utils.trackers.core import GenericLogger

        backend = DiskMetricsBackend(DiskMetricsConfig(root_dir=tmp_path))
        disk_logger = GenericLogger(backend)
        monkeypatch.setattr(trackers, "_redis_default", None)
        monkeypatch.setattr(trackers, "_disk_default", disk_logger)
        assert trackers.get_default_history_reader() is backend
        disk_logger.close()

    def test_returns_redis_backend_when_initialized(self, monkeypatch):
        import gigaevo.utils.trackers as trackers

        backend = RedisMetricsBackend(RedisMetricsConfig())
        fake_logger = MagicMock()
        fake_logger.backend = backend
        monkeypatch.setattr(trackers, "_redis_default", fake_logger)
        monkeypatch.setattr(trackers, "_disk_default", None)
        assert trackers.get_default_history_reader() is backend
