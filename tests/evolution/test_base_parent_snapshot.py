from gigaevo.evolution.engine.mutation import (
    base_parent_index,
    freeze_base_parent_snapshot,
)
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)


class _FakeParent:
    def __init__(self, selected, metrics, pid="p"):
        self._selected = selected
        self.metrics = metrics
        self.id = pid

    def get_metadata(self, key):
        if key == MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY:
            return self._selected
        return None


def test_snapshot_picks_base_parent_by_one_based_index():
    p1 = _FakeParent(["card-x"], {"r2": 0.5}, pid="p1")
    p2 = _FakeParent(["card-y"], {"r2": 0.8}, pid="p2")
    snap = freeze_base_parent_snapshot([p1, p2], base_parent=2)
    assert snap[MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY] == ["card-y"]
    assert snap[MUTATION_MEMORY_BASE_METRICS_METADATA_KEY] == {"r2": 0.8}
    assert snap[MUTATION_MEMORY_BASE_ID_METADATA_KEY] == "p2"


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
