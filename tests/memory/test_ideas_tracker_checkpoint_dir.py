"""Per-run backend factory and checkpoint_dir thread through the write pipeline.

Regression for the bug where memory cards (api_index.json, amem_exports/,
gam_shared/) landed in a static fallback path even when the engine was
started with a per-run Hydra output dir.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker, _run_write_pipeline


def _make_log_files(tmp_path: Path) -> tuple[Path, Path]:
    banks = tmp_path / "banks.json"
    best = tmp_path / "best_ideas.json"
    banks.write_text(json.dumps([{"active_bank": []}]), encoding="utf-8")
    best.write_text(json.dumps([{"best_ideas": []}]), encoding="utf-8")
    return banks, best


class _FakeMemory:
    def __init__(self) -> None:
        self.saved: list = []
        self.closed = False

    def get_card_write_stats(self) -> dict[str, int]:
        return {
            "processed": len(self.saved),
            "added": len(self.saved),
            "updated": 0,
            "rejected": 0,
            "updated_target_cards": 0,
        }

    def save_card(self, card) -> str:
        self.saved.append(card)
        return getattr(card, "id", "fake-id")

    def get_card(self, card_id):
        return None

    def rebuild(self) -> None:
        pass

    def sweep_harmful(self) -> list:
        return []

    def close(self) -> None:
        self.closed = True


class TestRunWritePipelineForwardsOverrides:
    """``_run_write_pipeline`` must forward backend/checkpoint_dir to ``main``."""

    def test_forwards_backend_and_checkpoint_dir(self, tmp_path, monkeypatch):
        banks, best = _make_log_files(tmp_path)

        captured: dict[str, object] = {}

        def fake_main(**kwargs):
            captured.update(kwargs)
            return {"stats": {"processed": 0, "added": 0, "updated": 0, "rejected": 0}}

        import gigaevo.memory.write_pipeline as wp

        monkeypatch.setattr(wp, "main", fake_main)

        factory = LocalMemoryBackendFactory()
        run_dir = tmp_path / "hydra_run" / "memory"
        _run_write_pipeline(
            enabled=True,
            banks_path=banks,
            best_ideas_path=best,
            programs_path=None,
            backend=factory,
            checkpoint_dir=run_dir,
            best_programs_percent=12.5,
        )

        assert captured["backend"] is factory
        assert captured["checkpoint_dir"] == run_dir
        assert captured["best_programs_percent"] == 12.5

    def test_enabled_without_backend_raises(self, tmp_path, monkeypatch):
        banks, best = _make_log_files(tmp_path)

        def fake_main(**kwargs):
            raise AssertionError("main must not be reached without a backend")

        import gigaevo.memory.write_pipeline as wp

        monkeypatch.setattr(wp, "main", fake_main)

        with pytest.raises(ValueError, match="memory/backend"):
            _run_write_pipeline(
                enabled=True,
                banks_path=banks,
                best_ideas_path=best,
                programs_path=None,
                backend=None,
            )

    def test_disabled_skips_main(self, tmp_path, monkeypatch):
        banks, best = _make_log_files(tmp_path)
        called = False

        def fake_main(**kwargs):
            nonlocal called
            called = True
            return None

        import gigaevo.memory.write_pipeline as wp

        monkeypatch.setattr(wp, "main", fake_main)
        _run_write_pipeline(
            enabled=False,
            banks_path=banks,
            best_ideas_path=best,
            programs_path=None,
            backend=None,
            checkpoint_dir=tmp_path / "ignored",
        )

        assert called is False


class TestMainBuildsBackendViaFactory:
    """``write_pipeline.main`` constructs the card bank through the factory."""

    def test_injected_factory_builds_backend(self, tmp_path):
        from gigaevo.memory import write_pipeline as wp

        banks, best = _make_log_files(tmp_path)
        fake = _FakeMemory()
        factory = MagicMock(spec=LocalMemoryBackendFactory)
        factory.build.return_value = fake

        run_dir = tmp_path / "hydra_run" / "memory"
        snapshot = wp.main(
            banks_path=banks,
            best_ideas_path=best,
            backend=factory,
            checkpoint_dir=run_dir,
        )

        factory.build.assert_called_once_with(
            checkpoint_dir=run_dir, evictor=None, deduplicator=None
        )
        assert isinstance(snapshot, dict)
        assert fake.closed is True

    def test_backend_is_required_with_no_default(self):
        from gigaevo.memory import write_pipeline as wp

        params = inspect.signature(wp.main).parameters
        assert params["backend"].default is inspect.Parameter.empty

    def test_no_legacy_config_params(self):
        from gigaevo.memory import write_pipeline as wp

        params = inspect.signature(wp.main).parameters
        assert "config_path" not in params
        assert "namespace" not in params
        assert "backend" in params
        assert "best_programs_percent" in params

    def test_write_stats_snapshot_written_next_to_banks(self, tmp_path):
        from gigaevo.memory import write_pipeline as wp

        banks, best = _make_log_files(tmp_path)
        factory = MagicMock(spec=LocalMemoryBackendFactory)
        factory.build.return_value = _FakeMemory()
        snapshot = wp.main(
            banks_path=banks,
            best_ideas_path=best,
            backend=factory,
            checkpoint_dir=tmp_path / "mem",
        )

        stats_path = tmp_path / "memory_write_stats.json"
        assert stats_path.exists()
        assert snapshot is not None
        assert "stats" in snapshot


class TestIdeaTrackerRequiresBackendForWrites:
    def test_write_enabled_without_backend_raises(self):
        with pytest.raises(ValueError, match=r"memory\.backend"):
            IdeaTracker(memory_write_enabled=True, backend=None)
