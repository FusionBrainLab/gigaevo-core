"""Unit tests for the LLM preset builders."""

from __future__ import annotations

import pytest

from gigaevo.config.defaults import DEFAULT_LLM_MAX_TOKENS
from gigaevo.config.llm_presets import (
    OPENROUTER_BASE_URL,
    OPENROUTER_FOUR_MODELS,
    build_gemini_3_flash,
    build_gemini_25_pro,
    build_gemini_31_pro,
    build_google,
    build_heterogeneous_bandit,
    build_openai,
    build_openrouter_bandit,
    build_openrouter_ensemble,
    build_single,
)
from gigaevo.config.schemas import BanditRouterConfig, EnsembleRouterConfig


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestOpenRouterEnsemble:
    def test_default_yields_four_models_with_equal_probs(self) -> None:
        cfg = build_openrouter_ensemble()
        assert isinstance(cfg, EnsembleRouterConfig)
        assert len(cfg.models) == 4
        assert cfg.probabilities == [0.25, 0.25, 0.25, 0.25]

    def test_models_match_canonical_openrouter_set(self) -> None:
        cfg = build_openrouter_ensemble()
        assert [m.model for m in cfg.models] == list(OPENROUTER_FOUR_MODELS)

    def test_every_endpoint_points_at_openrouter(self) -> None:
        cfg = build_openrouter_ensemble()
        for endpoint in cfg.models:
            assert endpoint.base_url == OPENROUTER_BASE_URL

    def test_custom_probabilities_applied(self) -> None:
        cfg = build_openrouter_ensemble(probabilities=[0.5, 0.2, 0.2, 0.1])
        assert cfg.probabilities == [0.5, 0.2, 0.2, 0.1]

    def test_custom_temperature_propagates(self) -> None:
        cfg = build_openrouter_ensemble(temperature=0.3)
        for endpoint in cfg.models:
            assert endpoint.temperature == 0.3


class TestOpenRouterBandit:
    def test_default_yields_bandit_over_four_models(self) -> None:
        cfg = build_openrouter_bandit()
        assert isinstance(cfg, BanditRouterConfig)
        assert len(cfg.models) == 4
        assert cfg.exploration_constant == 1.41
        assert cfg.window_size == 100

    def test_models_match_canonical_openrouter_set(self) -> None:
        cfg = build_openrouter_bandit()
        assert [m.model for m in cfg.models] == list(OPENROUTER_FOUR_MODELS)

    def test_custom_exploration_constant(self) -> None:
        cfg = build_openrouter_bandit(exploration_constant=2.5)
        assert cfg.exploration_constant == 2.5


class TestSinglePreset:
    def test_minimal(self) -> None:
        cfg = build_single("gpt-4o-mini")
        assert isinstance(cfg, EnsembleRouterConfig)
        assert len(cfg.models) == 1
        assert cfg.models[0].model == "gpt-4o-mini"
        assert cfg.probabilities == [1.0]

    def test_with_base_url(self) -> None:
        cfg = build_single("custom-model", base_url="http://internal:8080/v1")
        assert cfg.models[0].base_url == "http://internal:8080/v1"


class TestHeterogeneousBandit:
    def test_base_url_required(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            build_heterogeneous_bandit()

    def test_env_var_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_MODEL_1", raising=False)
        monkeypatch.delenv("LLM_MODEL_2", raising=False)
        cfg = build_heterogeneous_bandit(base_url="http://server:8080/v1")
        assert cfg.models[0].model == "meta-llama/Llama-3.3-70B-Instruct"
        assert cfg.models[1].model == "Qwen/Qwen2.5-72B-Instruct"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL_1", "custom/model-1")
        monkeypatch.setenv("LLM_MODEL_2", "custom/model-2")
        cfg = build_heterogeneous_bandit(base_url="http://server:8080/v1")
        assert cfg.models[0].model == "custom/model-1"
        assert cfg.models[1].model == "custom/model-2"

    def test_explicit_models_take_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MODEL_1", "env/m1")
        cfg = build_heterogeneous_bandit(
            model_1="explicit/m1",
            model_2="explicit/m2",
            base_url="http://server:8080/v1",
        )
        assert cfg.models[0].model == "explicit/m1"
        assert cfg.models[1].model == "explicit/m2"

    def test_temperature_defaults_to_yaml_value(self) -> None:
        cfg = build_heterogeneous_bandit(base_url="http://server:8080/v1")
        for endpoint in cfg.models:
            assert endpoint.temperature == 0.8


class TestGeminiPresets:
    def test_gemini_3_flash(self) -> None:
        cfg = build_gemini_3_flash()
        assert cfg.models[0].model == "google/gemini-3-flash-preview"
        assert cfg.models[0].base_url == OPENROUTER_BASE_URL

    def test_gemini_31_pro_includes_preview_suffix(self) -> None:
        """The OpenRouter slug is gemini-3.1-pro-preview, not
        gemini-3.1-pro. Without the suffix the route 404s."""
        cfg = build_gemini_31_pro()
        assert cfg.models[0].model == "google/gemini-3.1-pro-preview"

    def test_gemini_25_pro(self) -> None:
        cfg = build_gemini_25_pro()
        assert cfg.models[0].model == "google/gemini-2.5-pro"


class TestOpenAIPreset:
    def test_default_two_model_split(self) -> None:
        cfg = build_openai()
        assert len(cfg.models) == 2
        assert cfg.models[0].model == "openai/gpt-5"
        assert cfg.models[1].model == "openai/gpt-5-mini"
        assert cfg.probabilities == [0.1, 0.9]

    def test_custom_probabilities(self) -> None:
        cfg = build_openai(probabilities=[0.5, 0.5])
        assert cfg.probabilities == [0.5, 0.5]


class TestGooglePreset:
    def test_default_two_model_split(self) -> None:
        cfg = build_google()
        assert len(cfg.models) == 2
        assert cfg.models[0].model == "google/gemini-2.5-flash"
        assert cfg.models[1].model == "google/gemini-2.5-pro"
        assert cfg.probabilities == [0.1, 0.9]


class TestMaxTokensDefault:
    """Audit catch: openrouter and gemini YAMLs reference ${max_tokens}
    which resolves to constants/llm.yaml's 81920. Earlier preset
    builders defaulted to 16_384, a 5x token drift between the
    typed and YAML paths."""

    def test_openrouter_bandit_uses_default_max_tokens(self) -> None:
        cfg = build_openrouter_bandit()
        for endpoint in cfg.models:
            assert endpoint.max_tokens == DEFAULT_LLM_MAX_TOKENS

    def test_openrouter_ensemble_uses_default_max_tokens(self) -> None:
        cfg = build_openrouter_ensemble()
        for endpoint in cfg.models:
            assert endpoint.max_tokens == DEFAULT_LLM_MAX_TOKENS

    def test_gemini_3_flash_uses_default_max_tokens(self) -> None:
        cfg = build_gemini_3_flash()
        assert cfg.models[0].max_tokens == DEFAULT_LLM_MAX_TOKENS

    def test_openai_uses_default_max_tokens(self) -> None:
        cfg = build_openai()
        for endpoint in cfg.models:
            assert endpoint.max_tokens == DEFAULT_LLM_MAX_TOKENS

    def test_google_uses_default_max_tokens(self) -> None:
        cfg = build_google()
        for endpoint in cfg.models:
            assert endpoint.max_tokens == DEFAULT_LLM_MAX_TOKENS


class TestYAMLParity:
    """The presets exist to replace deployment-specific YAMLs with
    typed Python. Each preset must match the model set + endpoint
    coordinates from the corresponding YAML byte-equal."""

    def test_openrouter_bandit_matches_yaml_model_set(self) -> None:
        cfg = build_openrouter_bandit()
        yaml_models = {
            "google/gemini-2.5-flash",
            "google/gemini-3-flash-preview",
            "deepseek/deepseek-v3.2",
            "openai/gpt-4.1-mini",
        }
        assert {m.model for m in cfg.models} == yaml_models

    def test_openrouter_ensemble_matches_yaml_model_set(self) -> None:
        cfg = build_openrouter_ensemble()
        yaml_models = {
            "google/gemini-2.5-flash",
            "google/gemini-3-flash-preview",
            "deepseek/deepseek-v3.2",
            "openai/gpt-4.1-mini",
        }
        assert {m.model for m in cfg.models} == yaml_models

    def test_openrouter_bandit_exploration_matches_yaml(self) -> None:
        """openrouter_bandit.yaml pins exploration_constant: 1.41 and
        window_size: 100; both default values must match."""
        cfg = build_openrouter_bandit()
        assert cfg.exploration_constant == 1.41
        assert cfg.window_size == 100
