from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, field_validator

from gigaevo.config.schemas._base import FrozenStrictModel, reject_empty_or_cwd_path

if TYPE_CHECKING:
    from gigaevo.prompts.fetcher import PromptFetcher


class FixedDirPromptFetcherConfig(FrozenStrictModel):
    """Reads prompts from a directory or the package defaults.

    The runtime ``FixedDirPromptFetcher`` caches loaded templates in
    memory; this is the default for problems whose prompts are static
    files. ``prompts_dir=None`` falls back to the package defaults baked
    into ``gigaevo/prompts/``.
    """

    kind: Literal["fixed"] = "fixed"
    prompts_dir: Path | None = None

    @field_validator("prompts_dir")
    @classmethod
    def _prompts_dir_not_empty(cls, value: Path | None) -> Path | None:
        return reject_empty_or_cwd_path("prompts_dir", value)

    def build(self) -> PromptFetcher:
        from gigaevo.prompts.fetcher import FixedDirPromptFetcher

        return FixedDirPromptFetcher(prompts_dir=self.prompts_dir)


class GigaEvoArchivePromptFetcherConfig(FrozenStrictModel):
    """Pulls the current MAP-Elites champion from a co-running prompt
    GigaEvo instance and feeds its rendered prompt text to agents.

    Used when prompt evolution runs alongside the main evolution: the
    prompt run optimises prompts as programs, and this fetcher reads
    the current champion every ``cache_ttl_seconds``. Falls back to
    :class:`FixedDirPromptFetcherConfig` until the first champion is
    available.

    ``prompt_redis_db`` is the Redis DB of the prompt evolution run;
    ``main_redis_db`` is the Redis DB of THIS run (so stats can be
    written back so the prompt run can compute fitness). Both are
    integers because Redis databases are integer-indexed.
    """

    kind: Literal["coevolved"] = "coevolved"
    prompt_redis_db: int = Field(ge=0)
    main_redis_prefix: str = Field(min_length=1)
    main_redis_db: int | None = Field(default=None, ge=0)
    prompt_prefix: str = Field(default="prompt_evolution", min_length=1)
    archive_prefix: str = Field(default="island_fitness_island", min_length=1)
    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=6379, ge=1, le=65535)
    cache_ttl_seconds: float = Field(default=30.0, gt=0.0)
    fallback_prompts_dir: Path | None = None
    fitness_key: str = Field(default="fitness", min_length=1)

    @field_validator("fallback_prompts_dir")
    @classmethod
    def _fallback_dir_not_empty(cls, value: Path | None) -> Path | None:
        return reject_empty_or_cwd_path("fallback_prompts_dir", value)

    def build(self) -> PromptFetcher:
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        return GigaEvoArchivePromptFetcher(
            prompt_redis_db=self.prompt_redis_db,
            main_redis_prefix=self.main_redis_prefix,
            main_redis_db=self.main_redis_db,
            prompt_prefix=self.prompt_prefix,
            archive_prefix=self.archive_prefix,
            host=self.host,
            port=self.port,
            cache_ttl_seconds=self.cache_ttl_seconds,
            fallback_prompts_dir=self.fallback_prompts_dir,
            fitness_key=self.fitness_key,
        )


PromptFetcherConfig = Annotated[
    FixedDirPromptFetcherConfig | GigaEvoArchivePromptFetcherConfig,
    Field(discriminator="kind"),
]
