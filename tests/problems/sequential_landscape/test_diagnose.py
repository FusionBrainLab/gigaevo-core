import pytest

from problems.sequential_landscape.diagnose import recover, verify
from problems.sequential_landscape.specs import get_ladder


@pytest.mark.parametrize("inst", get_ladder(), ids=lambda i: i.name)
def test_every_ladder_instance_realizes_its_tree(inst):
    ls = inst.landscape()
    assert verify(ls), f"{inst.name} does not realize its prescribed tree"


@pytest.mark.parametrize("inst", get_ladder(), ids=lambda i: i.name)
def test_recover_counts_match(inst):
    ls = inst.landscape()
    rec = recover(ls)
    assert len(rec["minima"]) == ls.num_minima
    assert len(rec["barriers"]) == ls.num_minima - 1
