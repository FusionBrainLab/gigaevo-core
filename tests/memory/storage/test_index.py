"""In-memory VectorIndex scope documents, filters, and diff-sync."""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import CardKind
from gigaevo.memory.storage.config import EmbedConfig
from gigaevo.memory.storage.index import VectorIndex, render_scope_document

SCOPES = {
    "description": ("description",),
    "desc_expl": ("description", "explanation_summary"),
}


@pytest.fixture
def index():
    # Symmetric embedder for the retrieval-mechanics tests below; the asymmetric
    # query_prefix is exercised separately in test_query_prefix_*.
    embed = EmbedConfig(
        embed_scopes=dict(SCOPES), nearest_scope="description", query_prefix=""
    )
    return VectorIndex(embed)


def test_render_single_field_is_raw_text(make_card):
    card = make_card(description="alpha beta")
    assert render_scope_document(card, ("description",)) == "alpha beta"


def test_render_multi_field_labels_lines(make_card):
    card = make_card(description="alpha", explanation_summary="beta")
    assert (
        render_scope_document(card, ("description", "explanation_summary"))
        == "DESCRIPTION: alpha\nEXPLANATION_SUMMARY: beta"
    )


def test_render_skips_empty_fields(make_card):
    card = make_card(description="alpha", explanation_summary="")
    assert (
        render_scope_document(card, ("description", "explanation_summary"))
        == "DESCRIPTION: alpha"
    )


def test_render_all_empty_is_blank(make_card):
    card = make_card(description="   ")
    assert render_scope_document(card, ("description",)) == ""


def test_scopes_property(index):
    assert index.scopes == ("description", "desc_expl")


def test_query_orders_by_ascending_distance(index, make_card):
    exact = make_card(description="alpha alpha alpha")
    partial = make_card(description="alpha beta gamma")
    unrelated = make_card(description="delta epsilon zeta")
    index.upsert([exact, partial, unrelated])
    hits = index.query("description", "alpha", 3)
    assert [hit.card_id for hit in hits] == [exact.id, partial.id, unrelated.id]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[0].distance <= hits[1].distance <= hits[2].distance


def test_kind_and_exclude_filters(index, make_card):
    insight = make_card(description="shared topic")
    exemplar = make_card(
        kind=CardKind.PROGRAM, program_id="prog-1", description="shared topic"
    )
    index.upsert([insight, exemplar])

    by_kind = index.query("description", "shared topic", 5, kind=CardKind.PROGRAM)
    assert [hit.card_id for hit in by_kind] == [exemplar.id]

    by_exclusion = index.query(
        "description", "shared topic", 5, exclude_ids=frozenset({insight.id})
    )
    assert [hit.card_id for hit in by_exclusion] == [exemplar.id]

    combined = index.query(
        "description",
        "shared topic",
        5,
        kind=CardKind.INSIGHT,
        exclude_ids=frozenset({insight.id}),
    )
    assert combined == []


def test_query_prefix_applies_to_queries_not_documents(make_card, fake_embedder):
    embed = EmbedConfig(
        embed_scopes={"description": ("description",)},
        nearest_scope="description",
        query_prefix="INSTRUCT: ",
    )
    index = VectorIndex(embed)
    index.upsert([make_card(description="alpha beta")])
    assert fake_embedder.embedded == ["alpha beta"]
    fake_embedder.embedded.clear()
    index.query("description", "alpha", 3)
    assert fake_embedder.embedded == ["INSTRUCT: alpha"]


def test_empty_query_prefix_is_noop(make_card, fake_embedder):
    embed = EmbedConfig(
        embed_scopes={"description": ("description",)},
        nearest_scope="description",
        query_prefix="",
    )
    index = VectorIndex(embed)
    index.upsert([make_card(description="alpha")])
    fake_embedder.embedded.clear()
    index.query("description", "alpha", 3)
    assert fake_embedder.embedded == ["alpha"]


def test_unknown_scope_raises(index):
    with pytest.raises(KeyError, match="unknown embed scope"):
        index.query("nope", "text", 3)


def test_query_guards_return_empty(index, make_card):
    assert index.query("description", "alpha", 3) == []
    index.upsert([make_card(description="alpha")])
    assert index.query("description", "   ", 3) == []
    assert index.query("description", "alpha", 0) == []


def test_rebuild_over_unchanged_cards_embeds_nothing(index, make_card, fake_embedder):
    cards = [make_card(description="alpha"), make_card(description="beta")]
    index.rebuild(cards)
    fake_embedder.embedded.clear()
    index.rebuild(cards)
    assert fake_embedder.embedded == []


def test_rebuild_diff_syncs_stale_and_changed(index, make_card, fake_embedder):
    keep = make_card(description="alpha")
    drop = make_card(description="beta")
    index.rebuild([keep, drop])

    changed = keep.model_copy(update={"description": "alpha prime"})
    fake_embedder.embedded.clear()
    index.rebuild([changed])
    assert len(fake_embedder.embedded) == len(SCOPES)
    assert all("alpha prime" in text for text in fake_embedder.embedded)

    ids = {hit.card_id for hit in index.query("description", "beta alpha prime", 5)}
    assert ids == {keep.id}


def test_rebuild_refreshes_filter_metadata_without_text_change(index, make_card):
    original = make_card(task_key="task-a", description="same semantic text")
    index.rebuild([original])

    changed_task = original.model_copy(update={"task_key": "task-b"})
    index.rebuild([changed_task])

    assert index.query("description", "same semantic text", 1, task_key="task-a") == []
    assert [
        hit.card_id
        for hit in index.query(
            "description", "same semantic text", 1, task_key="task-b"
        )
    ] == [original.id]


def test_upsert_drops_card_from_emptied_scope(index, make_card):
    card = make_card(description="alpha", explanation_summary="beta")
    index.upsert([card])
    index.upsert([card.model_copy(update={"description": ""})])
    assert index.query("description", "alpha", 5) == []
    hits = index.query("desc_expl", "beta", 5)
    assert [hit.card_id for hit in hits] == [card.id]


def test_remove_is_idempotent(index, make_card):
    card = make_card(description="alpha")
    index.upsert([card])
    index.remove([card.id])
    assert index.query("description", "alpha", 5) == []
    index.remove([card.id])


def _mmr_cards(index, make_card):
    exact = make_card(description="alpha beta gamma")
    near_dup = make_card(description="alpha beta delta")
    diverse = make_card(description="zeta eta theta")
    index.upsert([exact, near_dup, diverse])
    return exact, near_dup, diverse


def test_mmr_lambda_one_is_pure_relevance(index, make_card):
    exact, near_dup, diverse = _mmr_cards(index, make_card)
    ordered = index.mmr_order(
        "description",
        "alpha beta gamma",
        [diverse.id, near_dup.id, exact.id],
        lambda_=1.0,
    )
    assert ordered == [exact.id, near_dup.id, diverse.id]


def test_mmr_low_lambda_promotes_diversity(index, make_card):
    exact, near_dup, diverse = _mmr_cards(index, make_card)
    ordered = index.mmr_order(
        "description",
        "alpha beta gamma",
        [exact.id, near_dup.id, diverse.id],
        lambda_=0.3,
    )
    assert ordered == [exact.id, diverse.id, near_dup.id]


def test_mmr_relevance_override_replaces_query_similarity(index, make_card):
    exact, near_dup, diverse = _mmr_cards(index, make_card)
    ordered = index.mmr_order(
        "description",
        "alpha beta gamma",
        [exact.id, near_dup.id, diverse.id],
        lambda_=1.0,
        relevance={diverse.id: 1.0, near_dup.id: 0.5, exact.id: 0.0},
    )
    assert ordered == [diverse.id, near_dup.id, exact.id]


def test_mmr_missing_ids_keep_input_order_at_tail(index, make_card):
    exact, near_dup, diverse = _mmr_cards(index, make_card)
    ordered = index.mmr_order(
        "description",
        "alpha beta gamma",
        ["ghost-2", diverse.id, "ghost-1", exact.id],
        lambda_=1.0,
    )
    assert ordered == [exact.id, diverse.id, "ghost-2", "ghost-1"]


def test_mmr_empty_ids_is_empty(index):
    assert index.mmr_order("description", "alpha", [], lambda_=0.5) == []


def test_mmr_unknown_scope_raises(index):
    with pytest.raises(KeyError, match="unknown embed scope"):
        index.mmr_order("nope", "alpha", ["x"], lambda_=0.5)


def _embed(model: str, *, query_prefix: str = "") -> EmbedConfig:
    return EmbedConfig(
        embedding_model=model,
        embed_scopes=dict(SCOPES),
        nearest_scope="description",
        query_prefix=query_prefix,
    )


def test_instances_with_different_embed_configs_are_independent(make_card):
    first = VectorIndex(_embed("model-a", query_prefix="A: "))
    second = VectorIndex(_embed("model-b", query_prefix="B: "))
    card = make_card(description="alpha")
    first.upsert([card])

    assert [hit.card_id for hit in first.query("description", "alpha", 1)] == [card.id]
    assert second.query("description", "alpha", 1) == []
