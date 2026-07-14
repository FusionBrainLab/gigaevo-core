from gigaevo.evolution.engine.mutation import (
    base_parent_index,
    freeze_base_parent_snapshot,
)
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.programs.metrics.paired import PER_SAMPLE_SCORES_KEY


class _FakeParent:
    def __init__(self, selected, metrics, pid="p", no_card_control=False):
        self._selected = selected
        self.metrics = metrics
        self.id = pid
        self._no_card_control = no_card_control

    def get_metadata(self, key):
        if key == MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY:
            return self._selected
        if key == MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY:
            return self._no_card_control
        return None


def test_snapshot_picks_base_parent_by_one_based_index():
    p1 = _FakeParent(["card-x"], {"r2": 0.5}, pid="p1")
    p2 = _FakeParent(["card-y"], {"r2": 0.8}, pid="p2")
    snap = freeze_base_parent_snapshot([p1, p2], base_parent=2)
    assert snap[MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY] == ["card-y"]
    assert snap[MUTATION_MEMORY_BASE_METRICS_METADATA_KEY] == {"r2": 0.8}
    assert snap[MUTATION_MEMORY_BASE_ID_METADATA_KEY] == "p2"
    assert snap[MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY] is False


def test_snapshot_carries_no_card_control_flag():
    p1 = _FakeParent([], {"r2": 0.5}, no_card_control=True)
    snap = freeze_base_parent_snapshot([p1], base_parent=1)
    assert snap[MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY] is True


def test_snapshot_clamps_out_of_range_index_to_first_parent():
    p1 = _FakeParent(["card-x"], {"r2": 0.5})
    snap = freeze_base_parent_snapshot([p1], base_parent=7)
    assert snap[MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY] == ["card-x"]


def test_snapshot_drops_falsy_card_ids():
    p1 = _FakeParent(["card-x", "", None], {"r2": 0.5})
    snap = freeze_base_parent_snapshot([p1], base_parent=1)
    assert snap[MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY] == ["card-x"]


def test_snapshot_empty_when_no_parents():
    assert freeze_base_parent_snapshot([], base_parent=1) == {}


def test_base_parent_index_accepts_wire_json_integers():
    assert base_parent_index(1) == 1
    assert base_parent_index("2") == 2


def test_base_parent_index_accepts_diff_namespace_letters():
    assert base_parent_index("A") == 1
    assert base_parent_index("B") == 2


def test_base_parent_index_defaults_to_first_parent_on_garbage():
    assert base_parent_index(None) == 1
    assert base_parent_index("parent A") == 1


class _FakeParentWithScores(_FakeParent):
    def __init__(self, *args, scores=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._scores = scores

    def get_metadata(self, key):
        if key == PER_SAMPLE_SCORES_KEY:
            return self._scores
        return super().get_metadata(key)


def test_snapshot_freezes_base_per_sample_scores():
    parent = _FakeParentWithScores(["card-x"], {"r2": 0.5}, scores=[0.4, 0.6])
    snap = freeze_base_parent_snapshot([parent], base_parent=1)
    assert snap[MUTATION_MEMORY_BASE_SCORES_METADATA_KEY] == [0.4, 0.6]


def test_snapshot_omits_scores_when_parent_has_none():
    for scores in (None, [], (0.4, 0.6)):
        parent = _FakeParentWithScores(["card-x"], {"r2": 0.5}, scores=scores)
        snap = freeze_base_parent_snapshot([parent], base_parent=1)
        assert MUTATION_MEMORY_BASE_SCORES_METADATA_KEY not in snap


def test_shared_card_provenance_credits_base_parent_not_listing_order():
    # Both parents cite the SAME card; base_parent=2 (the second-listed parent).
    # A card credited against an arbitrary parent's baseline would use the wrong
    # counterfactual — the retained provenance must anchor the shared card to the
    # base parent, whose baseline the child's overall gain is measured against.
    donor = _FakeParent(["shared-card"], {"r2": 0.4}, pid="donor")
    base = _FakeParent(["shared-card"], {"r2": 0.9}, pid="base")
    snap = freeze_base_parent_snapshot([donor, base], base_parent=2)
    sources = snap[MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY]
    assert sources["shared-card"]["parent_id"] == "base"
    assert sources["shared-card"]["parent_metrics"] == {"r2": 0.9}
