"""Tests for tools.lineage — lineage walk should tolerate corrupted parent chains."""

import uuid

from gigaevo.programs.program import Lineage, Program
from tools.lineage import _walk_lineage


def _uid(label: str) -> str:
    """Deterministic UUID per label so tests can reference programs by name."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, label))


def _prog(label: str, parent: str | None) -> Program:
    parents = [_uid(parent)] if parent else []
    return Program(
        id=_uid(label),
        code="x = 1",
        lineage=Lineage(parents=parents, generation=1),
    )


def test_walk_lineage_returns_chain_to_root() -> None:
    root = _prog("root", None)
    mid = _prog("mid", "root")
    tip = _prog("tip", "mid")
    id_map = {p.id: p for p in (root, mid, tip)}

    chain = _walk_lineage(tip, id_map, depth=None)
    assert [p.id for p in chain] == [_uid("tip"), _uid("mid"), _uid("root")]


def test_walk_lineage_stops_when_parent_missing() -> None:
    # external seed not present in id_map
    tip = _prog("tip", "ghost")
    id_map = {tip.id: tip}

    chain = _walk_lineage(tip, id_map, depth=None)
    assert [p.id for p in chain] == [_uid("tip")]


def test_walk_lineage_respects_depth_limit() -> None:
    chain_programs = [
        _prog(f"p{i}", f"p{i - 1}" if i > 0 else None) for i in range(5)
    ]
    id_map = {p.id: p for p in chain_programs}

    chain = _walk_lineage(chain_programs[-1], id_map, depth=2)
    assert [p.id for p in chain] == [_uid("p4"), _uid("p3"), _uid("p2")]


def test_walk_lineage_breaks_on_cycle_without_depth() -> None:
    # Corrupted parent chain: a -> b -> a. Without a cycle guard the walk
    # would loop forever (depth=None).
    a = _prog("a", "b")
    b = _prog("b", "a")
    id_map = {a.id: a, b.id: b}

    chain = _walk_lineage(a, id_map, depth=None)
    # Walk must terminate; visited IDs must stay unique.
    ids = [p.id for p in chain]
    assert ids[0] == _uid("a")
    assert len(ids) == len(set(ids))
    assert len(ids) <= 2


def test_walk_lineage_breaks_on_self_loop() -> None:
    # Pathological self-parent reference: a -> a.
    self_id = _uid("a")
    a = Program(
        id=self_id,
        code="x = 1",
        lineage=Lineage(parents=[self_id], generation=1),
    )
    id_map = {a.id: a}

    chain = _walk_lineage(a, id_map, depth=None)
    assert [p.id for p in chain] == [self_id]
