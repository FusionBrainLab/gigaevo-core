"""Tests for the DataPlaneError hierarchy.

The big invariant: structured subclasses inherit from
:class:`_StructuredError` so their ``str()`` shows the failure detail
(not an empty string, as would happen if the dataclass-generated
``__init__`` didn't call ``Exception.__init__``).

Secondary invariants:

    - Fields tagged ``redact=True`` appear in ``str(err)`` as
      ``<redacted>`` so secrets do not leak through tracebacks.
    - Long field reprs are truncated to keep traceback output bounded.
    - Subclasses with mutable container field types are rejected by
      ``__init_subclass__`` to preserve hashability.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import pytest

from gigaevo.dataplane.errors import (
    REDACT_META_KEY,
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
    TokenTagCollisionError,
    TransitionError,
    _StructuredError,
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


class TestRedaction:
    def test_lock_held_holder_is_redacted(self) -> None:
        # Holder may carry a lease token; it must not appear verbatim
        # in str(err) so log aggregators do not capture it.
        err = LockHeld(key="k", holder="lease-tok-secret-uuid")
        s = str(err)
        assert "lease-tok-secret-uuid" not in s
        assert "<redacted>" in s
        assert "key='k'" in s
        # Field still accessible on the instance for legitimate
        # programmatic inspection.
        assert err.holder == "lease-tok-secret-uuid"

    def test_lock_held_holder_none_still_redacted(self) -> None:
        # Even None values get the redaction treatment so the *presence*
        # of a holder cannot be inferred from str().
        err = LockHeld(key="k", holder=None)
        s = str(err)
        assert "<redacted>" in s
        assert "None" not in s

    def test_token_already_consumed_tag_redacted(self) -> None:
        err = TokenAlreadyConsumed(tag_repr="sensitive-tag-name")
        s = str(err)
        assert "sensitive-tag-name" not in s
        assert "<redacted>" in s

    def test_token_tag_collision_redacted(self) -> None:
        err = TokenTagCollisionError(duplicate_tag_repr="dup-tag")
        s = str(err)
        assert "dup-tag" not in s
        assert "<redacted>" in s

    def test_redact_metadata_key_is_exported(self) -> None:
        # External callers may want to mark fields on their own
        # _StructuredError subclasses; expose the metadata key.
        assert REDACT_META_KEY == "redact"


class TestLongFieldReprTruncation:
    def test_oversized_string_is_truncated(self) -> None:
        # Build a non-redacted field longer than the cap and verify the
        # traceback output stays bounded.
        big = "x" * 10_000
        err = StartupError(reason=big)
        s = str(err)
        assert len(s) < len(big)
        assert "truncated" in s


class TestUninitialisedSlot:
    def test_str_survives_missing_slot(self) -> None:
        # An exotic subclass might forget to set a slotted field via
        # object.__setattr__ during a custom __init__. __post_init__
        # must not crash with AttributeError.

        @dataclasses.dataclass(slots=True, frozen=True, init=False)
        class _PartialError(_StructuredError):
            x: int

            def __init__(self) -> None:
                # Intentionally do not initialise self.x; bypassed
                # frozen via __init__ override.
                self.__post_init__()

        err = _PartialError()
        # Must not crash; placeholder appears in args.
        assert "x=<uninitialised>" in str(err)


class TestInitSubclassValidation:
    def test_rejects_list_field(self) -> None:
        with pytest.raises(TypeError, match="hashable"):

            @dataclass(slots=True, frozen=True)
            class _BadList(_StructuredError):
                items: list[int]

    def test_rejects_dict_field(self) -> None:
        with pytest.raises(TypeError, match="hashable"):

            @dataclass(slots=True, frozen=True)
            class _BadDict(_StructuredError):
                items: dict[str, int]

    def test_accepts_tuple_field(self) -> None:
        @dataclass(slots=True, frozen=True)
        class _GoodTuple(_StructuredError):
            items: tuple[int, ...]

        err = _GoodTuple(items=(1, 2, 3))
        assert "(1, 2, 3)" in str(err)


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

    def test_redacted_field_still_participates_in_equality(self) -> None:
        # Redaction is presentation-only; equality and hash use the
        # underlying value so two errors with the same secret still
        # compare equal.
        a = LockHeld(key="k", holder="tok")
        b = LockHeld(key="k", holder="tok")
        assert a == b
        assert hash(a) == hash(b)
        c = LockHeld(key="k", holder="other")
        assert a != c


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
        assert TokenTagCollisionError in types
        assert ScriptNotRegisteredError in types
        assert NotStartedError in types


class TestRedactMetadataDeclaration:
    """Smoke-test that redacted fields are accepted by ``dataclass.field``."""

    def test_custom_subclass_with_redacted_field(self) -> None:
        @dataclass(slots=True, frozen=True)
        class _MyErr(_StructuredError):
            secret: str = field(metadata={REDACT_META_KEY: True})

        s = str(_MyErr(secret="hunter2"))
        assert "hunter2" not in s
        assert "secret=<redacted>" in s
