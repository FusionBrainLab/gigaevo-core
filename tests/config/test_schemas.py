"""Unit tests for the typed configuration schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    DataPlaneSettings,
    EnsembleRouterConfig,
    LLMConfig,
    RedisConfig,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestRedisConfig:
    def test_default_url(self) -> None:
        cfg = RedisConfig()
        assert cfg.url == "redis://localhost:6379/0"

    def test_custom_url(self) -> None:
        cfg = RedisConfig(host="redis.internal", port=6380, db=2)
        assert cfg.url == "redis://redis.internal:6380/2"

    def test_port_range(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(port=0)
        with pytest.raises(ValidationError):
            RedisConfig(port=70_000)

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(host="x", typo="oops")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = RedisConfig()
        with pytest.raises(ValidationError):
            cfg.host = "x"  # type: ignore[misc]

    def test_empty_host_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisConfig(host="")

    def test_to_storage_config_round_trip(self) -> None:
        redis = RedisConfig(host="r", port=6379, db=5)
        storage = redis.to_storage_config(key_prefix="my_problem")
        assert storage.redis_url == "redis://r:6379/5"
        assert storage.key_prefix == "my_problem"

    def test_json_round_trip_preserves_computed_url(self) -> None:
        original = RedisConfig(host="r", port=6380, db=2)
        as_json = original.model_dump_json()
        parsed = RedisConfig.model_validate_json(as_json)
        assert parsed == original
        assert parsed.url == "redis://r:6380/2"


class TestDataPlaneSettings:
    def test_compose_with_redis(self) -> None:
        cfg = DataPlaneSettings(
            redis=RedisConfig(),
            key_prefix="gigaevo:test",
        )
        assert cfg.redis.url == "redis://localhost:6379/0"
        assert cfg.startup_timeout_s > 0

    def test_startup_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DataPlaneSettings(
                redis=RedisConfig(),
                key_prefix="p",
                startup_timeout_s=0.0,
            )

    def test_empty_key_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataPlaneSettings(redis=RedisConfig(), key_prefix="")

    def test_nested_frozen_propagates(self) -> None:
        cfg = DataPlaneSettings(redis=RedisConfig(), key_prefix="gigaevo:test")
        with pytest.raises(ValidationError):
            cfg.redis.host = "x"  # type: ignore[misc]


class TestChatOpenAIConfig:
    def test_typo_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="temprature"):
            ChatOpenAIConfig(model="x", temprature=0.5)  # type: ignore[call-arg]

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ChatOpenAIConfig(model="x", temperature=-0.1)
        with pytest.raises(ValidationError):
            ChatOpenAIConfig(model="x", temperature=2.5)

    def test_missing_api_key_raises_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            ChatOpenAIConfig(model="x")

    def test_explicit_api_key_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = ChatOpenAIConfig(model="x", api_key="explicit")
        assert cfg.api_key == "explicit"

    def test_kind_literal_pinned(self) -> None:
        cfg = ChatOpenAIConfig(model="x")
        assert cfg.kind == "chat_openai"

    def test_empty_model_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatOpenAIConfig(model="")

    def test_api_key_excluded_from_dump_and_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resolved API key must not leak through serialization
        surfaces: model_dump, model_dump_json, and __repr__. The CLI
        writes the dumped JSON to disk as ``config.json`` and the repr
        is used in log lines; an api_key in either is a credential
        leak."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-DO-NOT-LEAK")
        cfg = ChatOpenAIConfig(model="x")
        assert cfg.api_key == "sk-secret-DO-NOT-LEAK"
        dumped = cfg.model_dump()
        assert "api_key" not in dumped
        dumped_json = cfg.model_dump_json()
        assert "api_key" not in dumped_json
        assert "sk-secret" not in dumped_json
        assert "sk-secret" not in repr(cfg)

    def test_api_key_round_trip_resolves_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A round trip through JSON drops the api_key on dump and
        re-resolves it from the environment on load. The reloaded
        config is functionally equivalent to the original even though
        the on-disk JSON carries no credential."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-round-trip-key")
        cfg = ChatOpenAIConfig(model="x")
        parsed = ChatOpenAIConfig.model_validate_json(cfg.model_dump_json())
        assert parsed.api_key == "sk-round-trip-key"
        assert parsed.model == cfg.model


class TestBanditRouterConfig:
    def test_minimal(self) -> None:
        cfg = BanditRouterConfig(models=[ChatOpenAIConfig(model="gpt-4o-mini")])
        assert cfg.kind == "bandit"
        assert cfg.exploration_constant > 0

    def test_requires_at_least_one_model(self) -> None:
        with pytest.raises(ValidationError):
            BanditRouterConfig(models=[])


class TestEnsembleRouterConfig:
    def test_probabilities_length_must_match(self) -> None:
        m = ChatOpenAIConfig(model="x")
        with pytest.raises(ValidationError, match="length"):
            EnsembleRouterConfig(models=[m, m], probabilities=[1.0])

    def test_negative_probability_rejected(self) -> None:
        m = ChatOpenAIConfig(model="x")
        with pytest.raises(ValidationError, match="positive"):
            EnsembleRouterConfig(models=[m, m], probabilities=[1.0, -0.1])

    def test_no_probabilities_means_uniform_at_runtime(self) -> None:
        m = ChatOpenAIConfig(model="x")
        cfg = EnsembleRouterConfig(models=[m, m])
        assert cfg.probabilities is None


class TestLLMDiscriminatedUnion:
    def test_round_trip_bandit(self) -> None:
        m = ChatOpenAIConfig(model="x")
        cfg = BanditRouterConfig(models=[m])
        ta: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, BanditRouterConfig)

    def test_round_trip_ensemble(self) -> None:
        m = ChatOpenAIConfig(model="x")
        cfg = EnsembleRouterConfig(models=[m])
        ta: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
        parsed = ta.validate_python(cfg.model_dump())
        assert isinstance(parsed, EnsembleRouterConfig)

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "unknown", "models": []})
