"""``ensure_writable_hf_cache`` diagnostics must flow through loguru.

The helper mutates process-wide HF env vars on shared clusters; its messages
are operational warnings, so they must reach the configured log sinks instead
of bare stdout.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from gigaevo.memory.ideas_tracker.hf_cache import ensure_writable_hf_cache


def test_unwritable_cache_fallback_warns_via_loguru(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HOME", "/proc/gigaevo-unwritable")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        ensure_writable_hf_cache()
    finally:
        logger.remove(handler_id)

    assert capsys.readouterr().out == ""
    assert any("Clearing unwritable HF_HOME" in m for m in messages)
    assert any("HF cache directory" in m for m in messages)
