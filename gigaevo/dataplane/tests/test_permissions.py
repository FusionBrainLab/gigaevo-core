"""Tests for the move-only :class:`Token` permission tokens."""

from __future__ import annotations

import copy
import pickle

import pytest

from gigaevo.dataplane.errors import TokenAlreadyConsumed, TokenNotPickleable
from gigaevo.dataplane.permissions import (
    Token,
    mint_combine,
    mint_root,
    mint_split,
    mint_split_n,
)


class TestTokenConsume:
    def test_consume_returns_tag(self) -> None:
        t = mint_root("subspace-a")
        assert t.consume() == "subspace-a"

    def test_consume_flips_flag(self) -> None:
        t = mint_root("x")
        assert not t.consumed
        t.consume()
        assert t.consumed

    def test_double_consume_raises(self) -> None:
        t = mint_root("x")
        t.consume()
        with pytest.raises(TokenAlreadyConsumed):
            t.consume()

    def test_tag_readable_after_consume(self) -> None:
        t = mint_root("readable")
        t.consume()
        assert t.tag == "readable"


class TestLinearityEnforcement:
    def test_copy_raises(self) -> None:
        t = mint_root("x")
        with pytest.raises(TypeError, match="move-only"):
            copy.copy(t)

    def test_deepcopy_raises(self) -> None:
        t = mint_root("x")
        with pytest.raises(TypeError, match="move-only"):
            copy.deepcopy(t)

    def test_pickle_raises(self) -> None:
        t = mint_root("x")
        with pytest.raises(TokenNotPickleable):
            pickle.dumps(t)

    def test_pickle_via_each_protocol(self) -> None:
        t = mint_root("x")
        for protocol in (0, 1, 2, 3, 4, 5):
            with pytest.raises(TokenNotPickleable):
                pickle.dumps(t, protocol=protocol)


class TestMintRoot:
    def test_returns_live_token(self) -> None:
        t = mint_root("root")
        assert isinstance(t, Token)
        assert not t.consumed
        assert t.tag == "root"


class TestMintSplit:
    def test_consumes_parent(self) -> None:
        parent = mint_root("parent")
        mint_split(parent, "left", "right")
        assert parent.consumed

    def test_produces_two_orthogonal_tokens(self) -> None:
        parent = mint_root("p")
        left, right = mint_split(parent, "L", "R")
        assert left.tag == "L"
        assert right.tag == "R"
        assert not left.consumed
        assert not right.consumed

    def test_cannot_split_consumed_parent(self) -> None:
        parent = mint_root("p")
        parent.consume()
        with pytest.raises(TokenAlreadyConsumed):
            mint_split(parent, "L", "R")


class TestMintSplitN:
    def test_zero_children_consumes_parent(self) -> None:
        parent = mint_root("p")
        result = mint_split_n(parent, [])
        assert result == []
        assert parent.consumed

    def test_n_children(self) -> None:
        parent = mint_root("p")
        children = mint_split_n(parent, ["a", "b", "c", "d"])
        assert len(children) == 4
        assert [c.tag for c in children] == ["a", "b", "c", "d"]
        assert all(not c.consumed for c in children)


class TestMintCombine:
    def test_consumes_both_inputs(self) -> None:
        a = mint_root("a")
        b = mint_root("b")
        c = mint_combine(a, b, "combined")
        assert a.consumed and b.consumed
        assert c.tag == "combined"
        assert not c.consumed

    def test_cannot_combine_consumed_input(self) -> None:
        a = mint_root("a")
        a.consume()
        b = mint_root("b")
        with pytest.raises(TokenAlreadyConsumed):
            mint_combine(a, b, "c")


class TestSplitCombineRoundTrip:
    def test_split_then_combine_works(self) -> None:
        root = mint_root("root")
        left, right = mint_split(root, "L", "R")
        recombined = mint_combine(left, right, "root-again")
        assert recombined.tag == "root-again"
        assert not recombined.consumed
