"""Tests for the ID taxonomy.

NewType aliases are compile-time only — indistinguishable from their
base types at runtime. The meaningful runtime tests are for the
:class:`ActorIdentity` round-trip and the :func:`make_script_name`
validator.
"""

from __future__ import annotations

import pytest

from gigaevo.dataplane.ids import (
    ActorId,
    ActorIdentity,
    RunId,
    ScriptName,
    WorkerId,
    make_actor_id,
    make_script_name,
)


class TestMakeActorId:
    def test_combines_run_and_worker(self) -> None:
        actor = make_actor_id(RunId("run-1"), WorkerId("worker-2"))
        assert actor == "run-1:worker-2"
        assert isinstance(actor, str)

    def test_round_trip_through_str(self) -> None:
        run = RunId("r")
        worker = WorkerId("w")
        actor = make_actor_id(run, worker)
        parts = actor.split(":", 1)
        assert parts == ["r", "w"]

    def test_actor_id_is_str_at_runtime(self) -> None:
        # NewType is erased; ActorId('x') == str at runtime.
        a: ActorId = ActorId("x")
        assert isinstance(a, str)


class TestActorIdentity:
    def test_pack_renders_joined_form(self) -> None:
        ident = ActorIdentity(run_id=RunId("run-1"), worker_id=WorkerId("w-2"))
        assert ident.pack() == "run-1:w-2"

    def test_str_equals_pack(self) -> None:
        ident = ActorIdentity(run_id=RunId("a"), worker_id=WorkerId("b"))
        assert str(ident) == ident.pack() == "a:b"

    def test_parse_roundtrip(self) -> None:
        original = ActorIdentity(run_id=RunId("r"), worker_id=WorkerId("w"))
        recovered = ActorIdentity.parse(original.pack())
        assert recovered == original

    def test_parse_handles_strings_with_internal_dashes(self) -> None:
        recovered = ActorIdentity.parse(ActorId("run-7:worker-9"))
        assert recovered.run_id == "run-7"
        assert recovered.worker_id == "worker-9"

    def test_parse_missing_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="missing ':'"):
            ActorIdentity.parse("noseparatorhere")

    def test_construct_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id is empty"):
            ActorIdentity(run_id=RunId(""), worker_id=WorkerId("w"))

    def test_construct_empty_worker_id_raises(self) -> None:
        with pytest.raises(ValueError, match="worker_id is empty"):
            ActorIdentity(run_id=RunId("r"), worker_id=WorkerId(""))

    def test_construct_run_id_with_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id contains ':'"):
            ActorIdentity(run_id=RunId("a:b"), worker_id=WorkerId("w"))

    def test_construct_worker_id_with_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="worker_id contains ':'"):
            ActorIdentity(run_id=RunId("r"), worker_id=WorkerId("a:b"))

    @pytest.mark.parametrize("nul", ["\x00", "head\x00tail"])
    def test_rejects_nul_byte(self, nul: str) -> None:
        # NUL is a string terminator in many client libraries (Postgres
        # via asyncpg in particular); accepting it would break wire
        # round-trips downstream.
        with pytest.raises(ValueError, match="control character"):
            ActorIdentity(run_id=RunId(nul), worker_id=WorkerId("w"))

    @pytest.mark.parametrize("ctrl", ["\n", "\r", "\x1b", "\x07", "\x7f"])
    def test_rejects_control_characters(self, ctrl: str) -> None:
        # CR / LF / ANSI ESC enable log-line injection on terminals and
        # in aggregators that ingest unstructured strings.
        with pytest.raises(ValueError, match="control character"):
            ActorIdentity(run_id=RunId(ctrl), worker_id=WorkerId("w"))
        with pytest.raises(ValueError, match="control character"):
            ActorIdentity(run_id=RunId("r"), worker_id=WorkerId(ctrl))

    @pytest.mark.parametrize("bidi", ["‮", "‭", "⁨", " "])
    def test_rejects_bidirectional_overrides(self, bidi: str) -> None:
        # RLO / LRO / line separators reverse or split the visible order
        # in renders; they must not pass type-check.
        with pytest.raises(
            ValueError, match="(bidirectional override|control character)"
        ):
            ActorIdentity(run_id=RunId(bidi), worker_id=WorkerId("w"))

    def test_frozen(self) -> None:
        ident = ActorIdentity(run_id=RunId("r"), worker_id=WorkerId("w"))
        with pytest.raises(AttributeError):
            ident.run_id = RunId("x")  # type: ignore[misc]  # frozen dataclass refuses

    def test_hashable_for_use_in_sets(self) -> None:
        a = ActorIdentity(run_id=RunId("r"), worker_id=WorkerId("w"))
        b = ActorIdentity(run_id=RunId("r"), worker_id=WorkerId("w"))
        c = ActorIdentity(run_id=RunId("r"), worker_id=WorkerId("x"))
        assert {a, b, c} == {a, c}


class TestMakeScriptName:
    def test_valid_snake_case(self) -> None:
        s = make_script_name("transition_program_state")
        assert s == "transition_program_state"
        assert isinstance(s, str)

    def test_valid_short(self) -> None:
        assert make_script_name("a") == "a"

    def test_valid_with_digits(self) -> None:
        assert make_script_name("rev2_swap") == "rev2_swap"

    def test_rejects_leading_digit(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            make_script_name("1foo")

    def test_rejects_uppercase(self) -> None:
        with pytest.raises(ValueError):
            make_script_name("FooBar")

    def test_rejects_dash(self) -> None:
        with pytest.raises(ValueError):
            make_script_name("foo-bar")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            make_script_name("")

    def test_rejects_space(self) -> None:
        with pytest.raises(ValueError):
            make_script_name("foo bar")

    def test_returns_typed_script_name(self) -> None:
        s = make_script_name("ok_name")
        # NewType-erased at runtime, but the typed return makes the
        # caller's downstream code accept it where ``ScriptName`` is
        # expected.
        accepted: ScriptName = s
        assert accepted == "ok_name"
