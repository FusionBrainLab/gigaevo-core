"""Per-run checkpoint_dir/namespace are threaded through the write pipeline.

Regression for the bug where memory cards (api_index.json, amem_exports/,
gam_shared/) landed in the static memory_backend.yaml path even when the
engine was started with a per-run Hydra output dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigaevo.memory.ideas_tracker.ideas_tracker import _run_write_pipeline


def _make_log_files(tmp_path: Path) -> tuple[Path, Path]:
    banks = tmp_path / "banks.json"
    best = tmp_path / "best_ideas.json"
    banks.write_text(json.dumps([{"active_bank": []}]), encoding="utf-8")
    best.write_text(json.dumps([{"best_ideas": []}]), encoding="utf-8")
    return banks, best


class TestRunWritePipelineForwardsOverrides:
    """``_run_write_pipeline`` must forward checkpoint_dir/namespace to ``main``."""

    def test_forwards_checkpoint_dir_and_namespace(self, tmp_path, monkeypatch):
        banks, best = _make_log_files(tmp_path)

        captured: dict[str, object] = {}

        def fake_main(**kwargs):
            captured.update(kwargs)
            return {"stats": {"processed": 0, "added": 0, "updated": 0, "rejected": 0}}

        import gigaevo.memory.write_pipeline as wp

        monkeypatch.setattr(wp, "main", fake_main)

        run_dir = tmp_path / "hydra_run" / "memory"
        _run_write_pipeline(
            enabled=True,
            banks_path=banks,
            best_ideas_path=best,
            programs_path=None,
            usage_updates_path=None,
            memory_usage_tracking_enabled=False,
            config_path=None,
            checkpoint_dir=run_dir,
            namespace="run_ns_42",
        )

        assert captured["checkpoint_dir"] == run_dir
        assert captured["namespace"] == "run_ns_42"

    def test_defaults_to_none_when_not_provided(self, tmp_path, monkeypatch):
        """Back-compat: callers that don't pass overrides must still work."""
        banks, best = _make_log_files(tmp_path)

        captured: dict[str, object] = {}

        def fake_main(**kwargs):
            captured.update(kwargs)
            return None

        import gigaevo.memory.write_pipeline as wp

        monkeypatch.setattr(wp, "main", fake_main)

        _run_write_pipeline(
            enabled=True,
            banks_path=banks,
            best_ideas_path=best,
            programs_path=None,
            usage_updates_path=None,
            memory_usage_tracking_enabled=False,
            config_path=None,
        )

        assert captured["checkpoint_dir"] is None
        assert captured["namespace"] is None

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
            usage_updates_path=None,
            memory_usage_tracking_enabled=False,
            checkpoint_dir=tmp_path / "ignored",
            namespace="ignored",
        )

        assert called is False


class TestMainAppliesOverrides:
    """``write_pipeline.main`` must mutate cfg.memory_dir / cfg.namespace
    before they get baked into MemoryConfig / ApiConfig."""

    def test_checkpoint_dir_overrides_cfg(self, tmp_path, monkeypatch):
        pytest.importorskip("gigaevo.memory.write_pipeline_config")

        from gigaevo.memory import write_pipeline as wp

        captured_cfg = {}

        def fake_load_config(_path):
            cfg = wp.PipelineConfig(
                settings_path=Path("/tmp/dummy.yaml"),
                banks_path=tmp_path / "banks.json",
                best_ideas_path=tmp_path / "best_ideas.json",
                programs_path=tmp_path / "programs.json",
                usage_updates_path=None,
                memory_dir=tmp_path / "stale_dir",
                enable_usage_tracking=False,
                memory_api_url="http://localhost:8000",
                namespace="stale_ns",
                use_api=False,
                channel="latest",
                author=None,
                enable_llm_synthesis=False,
                should_evolve=False,
                fill_missing_fields_with_llm=False,
                search_limit=5,
                rebuild_interval=999,
                sync_batch_size=100,
                sync_on_init=False,
                enable_bm25=False,
            )
            captured_cfg["cfg"] = cfg
            return cfg

        class _BoomMemory:
            def __init__(self, **_kw):
                raise RuntimeError("stop here — cfg mutation is what we test")

        monkeypatch.setattr(wp, "load_config", fake_load_config)
        monkeypatch.setattr(wp, "AmemGamMemory", _BoomMemory)

        run_dir = tmp_path / "hydra_run" / "memory"
        with pytest.raises(RuntimeError, match="stop here"):
            wp.main(
                banks_path=tmp_path / "banks.json",
                best_ideas_path=tmp_path / "best_ideas.json",
                checkpoint_dir=run_dir,
                namespace="run_ns_42",
            )

        cfg = captured_cfg["cfg"]
        assert cfg.memory_dir == run_dir
        assert cfg.namespace == "run_ns_42"

    def test_no_override_keeps_cfg_values(self, tmp_path, monkeypatch):
        from gigaevo.memory import write_pipeline as wp

        captured_cfg = {}

        def fake_load_config(_path):
            cfg = wp.PipelineConfig(
                settings_path=Path("/tmp/dummy.yaml"),
                banks_path=tmp_path / "banks.json",
                best_ideas_path=tmp_path / "best_ideas.json",
                programs_path=tmp_path / "programs.json",
                usage_updates_path=None,
                memory_dir=tmp_path / "yaml_default_dir",
                enable_usage_tracking=False,
                memory_api_url="http://localhost:8000",
                namespace="yaml_default_ns",
                use_api=False,
                channel="latest",
                author=None,
                enable_llm_synthesis=False,
                should_evolve=False,
                fill_missing_fields_with_llm=False,
                search_limit=5,
                rebuild_interval=999,
                sync_batch_size=100,
                sync_on_init=False,
                enable_bm25=False,
            )
            captured_cfg["cfg"] = cfg
            return cfg

        class _BoomMemory:
            def __init__(self, **_kw):
                raise RuntimeError("stop here")

        monkeypatch.setattr(wp, "load_config", fake_load_config)
        monkeypatch.setattr(wp, "AmemGamMemory", _BoomMemory)

        with pytest.raises(RuntimeError, match="stop here"):
            wp.main(
                banks_path=tmp_path / "banks.json",
                best_ideas_path=tmp_path / "best_ideas.json",
            )

        cfg = captured_cfg["cfg"]
        assert cfg.memory_dir == tmp_path / "yaml_default_dir"
        assert cfg.namespace == "yaml_default_ns"
