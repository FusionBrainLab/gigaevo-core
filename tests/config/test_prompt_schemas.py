"""Unit tests for the prompt-fetcher typed schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    FixedDirPromptFetcherConfig,
    GigaEvoArchivePromptFetcherConfig,
    PromptFetcherConfig,
)


class TestFixedDirPromptFetcherConfig:
    def test_default_constructs(self) -> None:
        cfg = FixedDirPromptFetcherConfig()
        assert cfg.kind == "fixed"
        assert cfg.prompts_dir is None

    def test_with_prompts_dir(self) -> None:
        cfg = FixedDirPromptFetcherConfig(
            prompts_dir=Path("/srv/gigaevo/prompts")
        )
        assert cfg.prompts_dir == Path("/srv/gigaevo/prompts")

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            FixedDirPromptFetcherConfig(prompt_dir=Path("/x"))  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = FixedDirPromptFetcherConfig()
        with pytest.raises(ValidationError):
            cfg.prompts_dir = Path("/x")  # type: ignore[misc]

    def test_builds_runtime_fetcher(self) -> None:
        from gigaevo.prompts.fetcher import FixedDirPromptFetcher

        fetcher = FixedDirPromptFetcherConfig().build()
        assert isinstance(fetcher, FixedDirPromptFetcher)

    def test_empty_prompts_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path must be real"):
            FixedDirPromptFetcherConfig(prompts_dir=Path(""))

    def test_cwd_dot_prompts_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path must be real"):
            FixedDirPromptFetcherConfig(prompts_dir=Path("."))


class TestGigaEvoArchivePromptFetcherConfig:
    def _minimal_kwargs(self) -> dict:  # type: ignore[no-untyped-def]
        return {
            "prompt_redis_db": 3,
            "main_redis_prefix": "gigaevo:test",
        }

    def test_minimal_constructs(self) -> None:
        cfg = GigaEvoArchivePromptFetcherConfig(**self._minimal_kwargs())
        assert cfg.kind == "coevolved"
        assert cfg.prompt_redis_db == 3
        assert cfg.main_redis_prefix == "gigaevo:test"
        assert cfg.prompt_prefix == "prompt_evolution"
        assert cfg.cache_ttl_seconds == 30.0
        assert cfg.fitness_key == "fitness"

    def test_main_redis_prefix_required(self) -> None:
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(prompt_redis_db=3)

    def test_prompt_redis_db_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=-1, main_redis_prefix="p"
            )

    def test_main_redis_db_optional_and_validated(self) -> None:
        cfg = GigaEvoArchivePromptFetcherConfig(
            prompt_redis_db=3, main_redis_prefix="p", main_redis_db=0
        )
        assert cfg.main_redis_db == 0

        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix="p", main_redis_db=-1
            )

    def test_port_range(self) -> None:
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix="p", port=0
            )
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix="p", port=70_000
            )

    def test_cache_ttl_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix="p", cache_ttl_seconds=0.0
            )

    def test_empty_main_redis_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix=""
            )

    def test_empty_fitness_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix="p", fitness_key=""
            )

    def test_empty_fallback_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path must be real"):
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3,
                main_redis_prefix="p",
                fallback_prompts_dir=Path(""),
            )

    def test_build_propagates_parameters_to_runtime(self) -> None:
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        cfg = GigaEvoArchivePromptFetcherConfig(
            prompt_redis_db=9,
            main_redis_prefix="gigaevo:exp",
            main_redis_db=0,
            prompt_prefix="custom_prefix",
            archive_prefix="island_main",
            host="redis.internal",
            port=6380,
            cache_ttl_seconds=15.0,
            fitness_key="custom_fit",
        )
        fetcher = cfg.build()
        assert isinstance(fetcher, GigaEvoArchivePromptFetcher)
        assert fetcher._prompt_redis_db == 9
        assert fetcher._main_redis_prefix == "gigaevo:exp"
        assert fetcher._prompt_prefix == "custom_prefix"
        assert fetcher._archive_prefix == "island_main"
        assert fetcher._host == "redis.internal"
        assert fetcher._port == 6380
        assert fetcher._cache_ttl == 15.0
        assert fetcher._fitness_key == "custom_fit"


class TestPromptFetcherUnion:
    def test_round_trip_each_variant(self) -> None:
        ta: TypeAdapter[PromptFetcherConfig] = TypeAdapter(PromptFetcherConfig)
        for cfg in (
            FixedDirPromptFetcherConfig(),
            GigaEvoArchivePromptFetcherConfig(
                prompt_redis_db=3, main_redis_prefix="p"
            ),
        ):
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)

    def test_json_round_trip_preserves_coevolved_parameters(self) -> None:
        ta: TypeAdapter[PromptFetcherConfig] = TypeAdapter(PromptFetcherConfig)
        cfg = GigaEvoArchivePromptFetcherConfig(
            prompt_redis_db=7,
            main_redis_prefix="gigaevo:round_trip",
            main_redis_db=0,
            prompt_prefix="custom_evolution",
            archive_prefix="island_alpha",
            host="redis.internal",
            port=6380,
            cache_ttl_seconds=45.0,
            fallback_prompts_dir=Path("/srv/fallback"),
            fitness_key="custom_fitness",
        )
        parsed = ta.validate_json(ta.dump_json(cfg))
        assert isinstance(parsed, GigaEvoArchivePromptFetcherConfig)
        assert parsed.prompt_redis_db == 7
        assert parsed.prompt_prefix == "custom_evolution"
        assert parsed.archive_prefix == "island_alpha"
        assert parsed.host == "redis.internal"
        assert parsed.port == 6380
        assert parsed.cache_ttl_seconds == 45.0
        assert parsed.fallback_prompts_dir == Path("/srv/fallback")
        assert parsed.fitness_key == "custom_fitness"

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[PromptFetcherConfig] = TypeAdapter(PromptFetcherConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_fetcher"})

    def test_each_variant_has_unique_kind(self) -> None:
        kinds = [
            FixedDirPromptFetcherConfig.model_fields["kind"].default,
            GigaEvoArchivePromptFetcherConfig.model_fields["kind"].default,
        ]
        assert len(set(kinds)) == len(kinds)
        assert set(kinds) == {"fixed", "coevolved"}
