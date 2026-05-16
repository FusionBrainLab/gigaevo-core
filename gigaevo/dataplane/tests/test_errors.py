"""Tests for the DataPlaneError hierarchy.

The big invariant: structured subclasses inherit from
:class:`_StructuredError` so their ``str()`` shows the failure detail
(not an empty string, as would happen if the dataclass-generated
``__init__`` didn't call ``Exception.__init__``).
"""

from __future__ import annotations

import dataclasses

import pytest

from gigaevo.dataplane.errors import (
    DataPlaneError,
    DeadlineExceeded,
    LockHeld,
    LockLost,
    NotStartedError,
    ScriptLostError,
    ScriptNotRegisteredError,
    StaleReadError,
    StartupError,
    TokenAlreadyConsumed,
    TransitionError,
    all_error_types,
)


class TestStructuredErrors:
    def test_startup_error_str_contains_reason(self) -> None:
        err = StartupError(reason="bad url")
        assert "reason=" in str(err)
        assert "bad url" in str(err)

    def test_deadline_exceeded_str_contains_fields(self) -> None:
        err = DeadlineExceeded(elapsed_s=1.5, budget_s=1.0)
        s = str(err)
        assert "elapsed_s=1.5" in s
        assert "budget_s=1.0" in s

    def test_stale_read_error_str_contains_all_four_fields(self) -> None:
        err = StaleReadError(
            observed_epoch=3,
            observed_generation=1,
            min_epoch=5,
            min_generation=0,
        )
        s = str(err)
        assert "observed_epoch=3" in s
        assert "min_epoch=5" in s

    def test_lock_held_optional_holder(self) -> None:
        a = LockHeld(key="k", holder=None)
        b = LockHeld(key="k", holder="some-token")
        assert "holder=None" in str(a)
        assert "some-token" in str(b)


class TestTransitionError:
    def test_classmethod_constructors_kind_field(self) -> None:
        assert TransitionError.stale("x").kind == "stale"
        assert TransitionError.illegal("x").kind == "illegal"
        assert TransitionError.duplicate("x").kind == "duplicate"
        assert TransitionError.unknown("X", "p").kind == "unknown"

    def test_classmethod_constructors_detail_field(self) -> None:
        assert TransitionError.stale("expected RUNNING").detail == "expected RUNNING"

    def test_str_includes_kind_and_detail(self) -> None:
        s = str(TransitionError.illegal("QUEUED -> DONE"))
        assert "illegal" in s
        assert "QUEUED -> DONE" in s


class TestRaiseAndCatch:
    def test_raise_structured_error(self) -> None:
        with pytest.raises(StartupError, match="bad"):
            raise StartupError(reason="bad")

    def test_catch_via_base_class(self) -> None:
        with pytest.raises(DataPlaneError):
            raise LockLost(key="some-key")

    def test_catch_specific(self) -> None:
        with pytest.raises(LockLost) as exc_info:
            raise LockLost(key="prefix:lock")
        assert exc_info.value.key == "prefix:lock"


class TestFrozenSemantics:
    def test_error_fields_are_immutable(self) -> None:
        err = StartupError(reason="initial")
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            err.reason = "mutated"  # type: ignore[misc]

    def test_errors_are_hashable(self) -> None:
        a = ScriptLostError(script_name="x")
        b = ScriptLostError(script_name="x")
        # Both frozen + slots -> hashable, dataclass eq -> equal
        assert hash(a) == hash(b)
        assert {a, b} == {a}


class TestRegistry:
    def test_all_error_types_lists_every_concrete_class(self) -> None:
        types = all_error_types()
        # All members are DataPlaneError subclasses
        for t in types:
            assert issubclass(t, DataPlaneError)

    def test_all_error_types_unique(self) -> None:
        types = all_error_types()
        assert len(types) == len(set(types))

    def test_all_error_types_includes_known_classes(self) -> None:
        types = set(all_error_types())
        assert StartupError in types
        assert LockHeld in types
        assert TokenAlreadyConsumed in types
        assert ScriptNotRegisteredError in types
        assert NotStartedError in types
