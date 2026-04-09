"""Tests for gigaevo.memory.config module-level constants."""

from __future__ import annotations

import importlib
import sys


def _reload_config(monkeypatch, extra_env: dict | None = None):
    """Reload config module with clean env."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    if "gigaevo.memory.config" in sys.modules:
        del sys.modules["gigaevo.memory.config"]
    return importlib.import_module("gigaevo.memory.config")


class TestConfigConstants:
    def test_string_constants_are_strings(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        assert isinstance(cfg.OPENROUTER_MODEL_NAME, str)
        assert isinstance(cfg.AMEM_EMBEDDING_MODEL_NAME, str)
        assert isinstance(cfg.GAM_DENSE_RETRIEVER_MODEL_NAME, str)

    def test_openai_api_key_none_when_not_set(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        assert cfg.OPENAI_API_KEY is None

    def test_openai_api_key_from_env(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"OPENAI_API_KEY": "sk-test"})
        assert cfg.OPENAI_API_KEY == "sk-test"

    def test_openai_api_key_from_openrouter_env_fallback(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"OPENROUTER_API_KEY": "sk-or-test"})
        assert cfg.OPENAI_API_KEY == "sk-or-test"

    def test_openrouter_reasoning_is_dict(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        assert isinstance(cfg.OPENROUTER_REASONING, dict)

    def test_amem_embedding_model_name_has_default(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        assert cfg.AMEM_EMBEDDING_MODEL_NAME  # non-empty string

    def test_no_load_settings_imported(self):
        """config.py must NOT import from runtime_config except resolve_settings_path."""
        import ast
        import pathlib

        src = pathlib.Path("gigaevo/memory/config.py").read_text()
        tree = ast.parse(src)
        forbidden = {
            "deep_get",
            "load_settings",
            "to_str",
            "to_bool",
            "to_int",
            "to_list",
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "runtime_config" in node.module
            ):
                names = {alias.name for alias in node.names}
                assert not names & forbidden, f"Forbidden import: {names & forbidden}"
