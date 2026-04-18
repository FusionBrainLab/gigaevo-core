"""Tests for the general canonical-events registry.

The registry lives in `gigaevo.monitoring.events`. Events are Pydantic subclasses
of `BaseEvent`. Subclassing a concrete event auto-registers it in the
module-level `CANONICAL_EVENTS` dict via `__init_subclass__`.

Role-specific fields (G/D labels, adversarial-specific invariants) MUST NOT
appear in the general registry — those belong in `gigaevo.adversarial.events`.
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import ValidationError
import pytest

from gigaevo.monitoring.events import (
    CANONICAL_EVENTS,
    BaseEvent,
)


class TestBaseEvent:
    def test_base_event_has_required_classvars(self) -> None:
        assert hasattr(BaseEvent, "__init_subclass__")
        annotations = BaseEvent.__annotations__
        for expected in (
            "event",
            "description",
            "health_question",
            "expected_after_gen",
            "schema_version",
        ):
            assert expected in annotations, f"BaseEvent missing ClassVar {expected!r}"

    def test_base_event_has_optional_run_label_field(self) -> None:
        fields = BaseEvent.model_fields
        assert "run_label" in fields, (
            "BaseEvent must expose run_label for regex grouping"
        )
        # optional: default None
        assert fields["run_label"].default is None

    def test_base_event_has_no_role_field(self) -> None:
        # Role labels (G/D) are experiment-specific convention, not a framework primitive.
        assert "role" not in BaseEvent.model_fields

    def test_base_event_cannot_be_instantiated_without_subclass(self) -> None:
        # Abstract-by-convention: BaseEvent has no `event` ClassVar value,
        # so instantiating it directly must not auto-register.
        # (Nothing to assert on construction itself; the invariant is that
        # only subclasses with a concrete `event` land in CANONICAL_EVENTS.)
        assert "" not in CANONICAL_EVENTS
        assert None not in CANONICAL_EVENTS


class TestAutoRegistration:
    def test_defining_subclass_registers_it(self) -> None:
        # Local subclass — must auto-register via __init_subclass__
        class _TestEventAlpha(BaseEvent):
            event: ClassVar[str] = "__TEST_ALPHA__"
            description: ClassVar[str] = "unit-test event"
            health_question: ClassVar[str] = "does registration work?"

        try:
            assert "__TEST_ALPHA__" in CANONICAL_EVENTS
            assert CANONICAL_EVENTS["__TEST_ALPHA__"] is _TestEventAlpha
        finally:
            CANONICAL_EVENTS.pop("__TEST_ALPHA__", None)

    def test_duplicate_event_name_raises(self) -> None:
        class _TestEventBeta(BaseEvent):
            event: ClassVar[str] = "__TEST_BETA__"
            description: ClassVar[str] = "first"
            health_question: ClassVar[str] = "?"

        try:
            with pytest.raises(ValueError, match="__TEST_BETA__"):

                class _TestEventBetaDup(BaseEvent):
                    event: ClassVar[str] = "__TEST_BETA__"
                    description: ClassVar[str] = "duplicate"
                    health_question: ClassVar[str] = "?"
        finally:
            CANONICAL_EVENTS.pop("__TEST_BETA__", None)

    def test_subclass_without_event_is_not_registered(self) -> None:
        # Intermediate abstract subclass (no `event`) must not be registered.
        before = set(CANONICAL_EVENTS)

        class _AbstractIntermediate(BaseEvent):
            pass

        assert set(CANONICAL_EVENTS) == before


class TestPydanticRoundtrip:
    def test_concrete_event_roundtrips(self) -> None:
        class _TestEventGamma(BaseEvent):
            event: ClassVar[str] = "__TEST_GAMMA__"
            description: ClassVar[str] = "roundtrip test"
            health_question: ClassVar[str] = "?"
            foo: int
            bar: str | None = None

        try:
            instance = _TestEventGamma(foo=42, bar="hi", run_label="K5_1_G")
            dumped = instance.model_dump()
            assert dumped["foo"] == 42
            assert dumped["bar"] == "hi"
            assert dumped["run_label"] == "K5_1_G"
            restored = _TestEventGamma.model_validate(dumped)
            assert restored.foo == 42
            assert restored.run_label == "K5_1_G"
        finally:
            CANONICAL_EVENTS.pop("__TEST_GAMMA__", None)

    def test_validation_rejects_bad_type(self) -> None:
        class _TestEventDelta(BaseEvent):
            event: ClassVar[str] = "__TEST_DELTA__"
            description: ClassVar[str] = "validation test"
            health_question: ClassVar[str] = "?"
            latency_ms: int

        try:
            with pytest.raises(ValidationError):
                _TestEventDelta(latency_ms="not-an-int")  # type: ignore[arg-type]
        finally:
            CANONICAL_EVENTS.pop("__TEST_DELTA__", None)


class TestGeneralRegistryIsRoleAgnostic:
    def test_no_role_references_in_general_events_module(self) -> None:
        # The general registry file must not reference G/D role strings.
        # Adversarial-specific role logic belongs in gigaevo/adversarial/events.py.
        import inspect

        import gigaevo.monitoring.events as events_mod

        src = inspect.getsource(events_mod)
        # Forbidden substrings that would indicate role hardcoding.
        forbidden_patterns = [
            r"\brole\s*=",  # role= keyword arg
            r"\bconstructor\b.*\bimprover\b",  # role-pair strings
            r'"[GD]"\s*[:,]',  # 'G': or 'D':
        ]
        for pattern in forbidden_patterns:
            assert not re.search(pattern, src), (
                f"gigaevo/monitoring/events.py contains role-specific pattern {pattern!r} — "
                "move to gigaevo/adversarial/events.py"
            )
