from gigaevo.evolution.engine.mutation import freeze_base_parent_snapshot
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)


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
