"""StoreConfig/EmbedConfig/ResearchConfig validation and StoreState transitions."""

from __future__ import annotations

import pytest

from gigaevo.memory.storage.config import EmbedConfig, ResearchConfig, StoreConfig
from gigaevo.memory.storage.state import (
    StoreState,
    is_valid_transition,
    validate_transition,
)


def test_default_embed_config_is_consistent():
    embed = EmbedConfig()
    assert embed.nearest_scope in embed.embed_scopes
    assert "task_description" not in embed.embed_scopes[embed.nearest_scope]


def test_default_embed_scopes_use_only_semantic_card_content():
    embed = EmbedConfig()

    fields = {field for scope in embed.embed_scopes.values() for field in scope}

    assert fields <= {
        "description",
        "explanation_summary",
        "task_description_summary",
    }
    assert "description" in fields
    assert not {"keywords", "category", "fitness", "task_key"}.intersection(fields)


def test_embed_scopes_must_exist():
    with pytest.raises(ValueError, match="at least one scope"):
        EmbedConfig(embed_scopes={})


def test_embed_scope_rejects_empty_field_list():
    with pytest.raises(ValueError, match="no card fields"):
        EmbedConfig(embed_scopes={"s": ()}, nearest_scope="s")


def test_embed_scope_rejects_non_text_card_fields():
    with pytest.raises(ValueError, match="non-text card fields"):
        EmbedConfig(embed_scopes={"s": ("fitness",)}, nearest_scope="s")


def test_nearest_scope_must_be_configured():
    with pytest.raises(ValueError, match="nearest_scope"):
        EmbedConfig(embed_scopes={"s": ("description",)}, nearest_scope="other")


def test_mmr_lambda_defaults_to_pure_relevance():
    assert ResearchConfig().mmr_lambda == 1.0


def test_mmr_lambda_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="mmr_lambda"):
        ResearchConfig(mmr_lambda=1.5)
    with pytest.raises(ValueError, match="mmr_lambda"):
        ResearchConfig(mmr_lambda=-0.1)


def test_top_k_by_scope_overrides_default():
    research = ResearchConfig(top_k_by_scope={"a": 7}, default_top_k=2)
    assert research.top_k("a") == 7
    assert research.top_k("b") == 2


def test_store_config_bank_layout(tmp_path):
    config = StoreConfig(path=tmp_path)
    assert config.bank_file == tmp_path / "cards.json"


def test_query_scopes_must_be_embed_scopes(tmp_path):
    with pytest.raises(ValueError, match="query_scopes"):
        StoreConfig(path=tmp_path, research=ResearchConfig(query_scopes=("nope",)))


def test_resolved_query_scopes_default_to_all(tmp_path):
    config = StoreConfig(path=tmp_path)
    assert config.resolved_query_scopes == tuple(config.embed.embed_scopes)


def test_resolved_query_scopes_respect_explicit_subset(tmp_path):
    config = StoreConfig(
        path=tmp_path, research=ResearchConfig(query_scopes=("description",))
    )
    assert config.resolved_query_scopes == ("description",)


def test_store_state_transitions():
    assert is_valid_transition(StoreState.INITIALIZING, StoreState.READY)
    assert is_valid_transition(StoreState.READY, StoreState.BUILDING)
    assert is_valid_transition(StoreState.BUILDING, StoreState.READY)
    assert is_valid_transition(StoreState.BUILDING, StoreState.ERROR)
    assert is_valid_transition(StoreState.ERROR, StoreState.INITIALIZING)
    assert is_valid_transition(StoreState.READY, StoreState.READY)
    assert not is_valid_transition(StoreState.INITIALIZING, StoreState.BUILDING)
    assert not is_valid_transition(StoreState.ERROR, StoreState.READY)
    with pytest.raises(ValueError, match="Invalid state transition"):
        validate_transition(StoreState.ERROR, StoreState.READY)
