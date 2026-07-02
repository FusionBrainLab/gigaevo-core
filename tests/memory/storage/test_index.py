"""VectorIndex over tmpdir Chroma: scope documents, filters, and diff-sync."""

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
def index(tmp_path):
    embed = EmbedConfig(embed_scopes=dict(SCOPES), nearest_scope="description")
    return VectorIndex(tmp_path / "chroma", embed)


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
