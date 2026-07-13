"""Storage configuration — what to embed, how to retrieve, where to persist.

Embedding is config, not code: ``embed_scopes`` maps a scope name to the card
text fields whose concatenation gets embedded into that scope's vector
collection. Adding a retrieval channel is a YAML edit, never a code change.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gigaevo.memory.cards import Card

CARD_TEXT_FIELDS = frozenset(
    name for name, field in Card.model_fields.items() if field.annotation is str
)

DEFAULT_EMBED_SCOPES: dict[str, tuple[str, ...]] = {
    "description": ("description",),
    "desc_expl": ("description", "explanation_summary"),
    "desc_task": ("description", "task_description_summary"),
}


class EmbedConfig(BaseModel):
    """Which model embeds which card fields into which collections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_model: str = Field(
        default="Snowflake/snowflake-arctic-embed-m-v1.5",
        description="SentenceTransformer model powering every collection.",
    )
    embed_scopes: dict[str, tuple[str, ...]] = Field(
        default_factory=lambda: dict(DEFAULT_EMBED_SCOPES),
        description="Scope name -> card text fields concatenated and embedded "
        "into that scope's collection.",
    )
    nearest_scope: str = Field(
        default="desc_expl",
        description="Scope backing nearest(); desc_expl won the embedder sweep. "
        "task_description is deliberately not part of it.",
    )
    query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="Instruction prepended to every retrieval QUERY before "
        "embedding (never to the indexed card documents) — the asymmetric "
        "query prompt arctic-embed-m-v1.5 was trained with; the embedder sweep "
        "left it as free headroom. Set '' for a symmetric embedder that takes "
        "no query instruction.",
    )

    @model_validator(mode="after")
    def _validate_scopes(self) -> EmbedConfig:
        if not self.embed_scopes:
            raise ValueError("embed_scopes must define at least one scope")
        for scope, fields in self.embed_scopes.items():
            if not fields:
                raise ValueError(f"embed scope {scope!r} lists no card fields")
            unknown = [f for f in fields if f not in CARD_TEXT_FIELDS]
            if unknown:
                raise ValueError(
                    f"embed scope {scope!r} references non-text card fields "
                    f"{unknown}; valid fields: {sorted(CARD_TEXT_FIELDS)}"
                )
        if self.nearest_scope not in self.embed_scopes:
            raise ValueError(
                f"nearest_scope {self.nearest_scope!r} is not a configured "
                f"embed scope {sorted(self.embed_scopes)}"
            )
        return self


class ResearchConfig(BaseModel):
    """Budgets and search surface of the agentic research loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_scopes: tuple[str, ...] = Field(
        default=(),
        description="Embed scopes the planner may query; empty = all configured.",
    )
    top_k_by_scope: dict[str, int] = Field(default_factory=dict)
    default_top_k: int = Field(default=3, gt=0)
    max_iters: int = Field(default=3, gt=0)
    max_cards: int = Field(default=3, gt=0)
    reflect_payload_chars: int = Field(
        default=36000,
        gt=0,
        description="Char budget for the reflector's rendered candidate briefs; "
        "over-budget briefs are dropped whole (tail-first) behind an "
        "omitted-ids marker, never clipped mid-JSON.",
    )
    mmr_lambda: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="MMR relevance-diversity trade-off for candidate ordering "
        "(VectorIndex.mmr_order); 1.0 = pure relevance (off). Not yet consumed "
        "by the research loop.",
    )

    def top_k(self, scope: str) -> int:
        return self.top_k_by_scope.get(scope, self.default_top_k)


class StoreConfig(BaseModel):
    """Bank persistence and retrieval knobs of a local store.

    The card bank is persisted under ``path``; the vector index is in-memory
    and derived from the bank by each process.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path = Field(description="Checkpoint directory owning the card bank.")
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)

    @property
    def bank_file(self) -> Path:
        return self.path / "cards.json"

    @model_validator(mode="after")
    def _validate_research_scopes(self) -> StoreConfig:
        unknown = [
            s for s in self.research.query_scopes if s not in self.embed.embed_scopes
        ]
        if unknown:
            raise ValueError(
                f"research.query_scopes {unknown} are not configured embed "
                f"scopes {sorted(self.embed.embed_scopes)}"
            )
        return self

    @property
    def resolved_query_scopes(self) -> tuple[str, ...]:
        return self.research.query_scopes or tuple(self.embed.embed_scopes)
