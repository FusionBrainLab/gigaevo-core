"""Tests for the FSM tables and encoding helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

from gigaevo.dataplane.transitions import (
    CLAIM_STATE_TRANSITIONS,
    LOCK_STATE_TRANSITIONS,
    PROGRAM_STATE_TRANSITIONS,
    ClaimState,
    LockState,
    ProgramState,
    encode_for_lua,
    fsm_key,
    is_valid_transition,
    load_fsm_table,
)


class TestProgramStateFSM:
    def test_queued_to_running_legal(self) -> None:
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,
            ProgramState.QUEUED,
            ProgramState.RUNNING,
        )

    def test_queued_to_done_illegal(self) -> None:
        # QUEUED cannot jump to DONE — must go through RUNNING.
        assert not is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,
            ProgramState.QUEUED,
            ProgramState.DONE,
        )

    def test_running_to_done_legal(self) -> None:
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,
            ProgramState.RUNNING,
            ProgramState.DONE,
        )

    def test_done_to_queued_legal_reevaluation(self) -> None:
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,
            ProgramState.DONE,
            ProgramState.QUEUED,
        )

    def test_discarded_terminal(self) -> None:
        # DISCARDED can only stay DISCARDED.
        for target in (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE):
            assert not is_valid_transition(
                PROGRAM_STATE_TRANSITIONS,
                ProgramState.DISCARDED,
                target,
            )
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,
            ProgramState.DISCARDED,
            ProgramState.DISCARDED,
        )

    def test_self_transitions_allowed(self) -> None:
        # Every non-terminal state allows a self-transition (idempotent re-emit).
        for s in (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE):
            assert is_valid_transition(
                PROGRAM_STATE_TRANSITIONS,
                s,
                s,
            )


class TestClaimStateFSM:
    def test_unclaimed_to_claimed(self) -> None:
        assert is_valid_transition(
            CLAIM_STATE_TRANSITIONS,
            ClaimState.UNCLAIMED,
            ClaimState.CLAIMED,
        )

    def test_claimed_to_acked_or_expired(self) -> None:
        for target in (ClaimState.ACKED, ClaimState.EXPIRED):
            assert is_valid_transition(
                CLAIM_STATE_TRANSITIONS,
                ClaimState.CLAIMED,
                target,
            )

    def test_acked_terminal(self) -> None:
        for target in ClaimState:
            assert not is_valid_transition(
                CLAIM_STATE_TRANSITIONS,
                ClaimState.ACKED,
                target,
            )

    def test_expired_can_reenter_unclaimed(self) -> None:
        assert is_valid_transition(
            CLAIM_STATE_TRANSITIONS,
            ClaimState.EXPIRED,
            ClaimState.UNCLAIMED,
        )


class TestLockStateFSM:
    def test_held_to_renewed_released_lost(self) -> None:
        for target in (LockState.RENEWED, LockState.RELEASED, LockState.LOST):
            assert is_valid_transition(
                LOCK_STATE_TRANSITIONS,
                LockState.HELD,
                target,
            )

    def test_renewed_can_repeat(self) -> None:
        # Successive renewals stay in RENEWED.
        assert is_valid_transition(
            LOCK_STATE_TRANSITIONS,
            LockState.RENEWED,
            LockState.RENEWED,
        )

    def test_released_and_lost_are_terminal(self) -> None:
        for terminal in (LockState.RELEASED, LockState.LOST):
            for target in LockState:
                assert not is_valid_transition(
                    LOCK_STATE_TRANSITIONS,
                    terminal,
                    target,
                )


class TestEncodeForLua:
    def test_rejects_comma_in_state_value(self) -> None:
        # Comma is the on-wire separator the Lua side uses to detect
        # legal targets; a comma inside a state value would let a forged
        # ``"X,Y"`` value satisfy a check for either ``"X"`` or ``"Y"``
        # alone. The encoder MUST refuse such inputs at the boundary.
        from enum import StrEnum

        class _BadEnum(StrEnum):
            OK = "OK"
            EVIL = "EVIL,STATE"

        table = {_BadEnum.OK: {_BadEnum.EVIL}, _BadEnum.EVIL: {_BadEnum.OK}}
        with pytest.raises(ValueError, match="contains ','"):
            encode_for_lua(table)

    def test_encodes_each_from_state(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        # Every from-state with at least one allowed target appears under
        # both its declared form and its lowercase alias so callers in
        # either vocabulary can resolve the row.
        for from_state in PROGRAM_STATE_TRANSITIONS:
            assert from_state.value in encoded
            assert from_state.value.lower() in encoded

    def test_targets_comma_joined_and_sorted(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        queued_targets = encoded["QUEUED"]
        parts = queued_targets.split(",")
        assert parts == sorted(parts)
        # Each target appears in both declared (uppercase) and lowercase
        # alias form so the membership walk succeeds for either case.
        assert set(parts) == {
            "DISCARDED",
            "QUEUED",
            "RUNNING",
            "discarded",
            "queued",
            "running",
        }
        # The lowercase row carries an identical comma-list.
        assert encoded["queued"] == queued_targets

    def test_discarded_self_only(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        # Self-only terminal state — the comma list still carries both
        # case variants of the single legal target.
        assert encoded["DISCARDED"] == "DISCARDED,discarded"
        assert encoded["discarded"] == "DISCARDED,discarded"

    def test_lowercase_alias_resolves_to_same_targets(self) -> None:
        """A blob persisted with a lowercase state value resolves the same
        row a coordinator-native (uppercase) caller does.

        Drives the case-tolerance invariant: callers from either
        vocabulary share the same FSM hash, so the legacy lowercase
        persisted blob format and the dp-native uppercase enum can
        co-exist without a separate bridging helper.
        """
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        for src in PROGRAM_STATE_TRANSITIONS:
            assert encoded[src.value] == encoded[src.value.lower()]


class TestFsmKey:
    def test_prefixed(self) -> None:
        assert fsm_key("run-7", "program_state") == "run-7:fsm:program_state"

    def test_different_prefixes_isolate(self) -> None:
        a = fsm_key("run-a", "x")
        b = fsm_key("run-b", "x")
        assert a != b


# ── load_fsm_table (fakeredis-backed) ─────────────────────────────────


@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """Independent fakeredis instance per test."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


class TestLoadFsmTable:
    async def test_writes_encoded_mapping(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        await load_fsm_table(
            fake_redis,
            key_prefix="run-x",
            name="program_state",
            table=PROGRAM_STATE_TRANSITIONS,
        )
        key = fsm_key("run-x", "program_state")
        stored = await fake_redis.hgetall(key)  # type: ignore[misc]
        assert stored == encode_for_lua(PROGRAM_STATE_TRANSITIONS)

    async def test_idempotent_under_same_content(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        for _ in range(3):
            await load_fsm_table(
                fake_redis,
                key_prefix="run-x",
                name="lock_state",
                table=LOCK_STATE_TRANSITIONS,
            )
        key = fsm_key("run-x", "lock_state")
        stored = await fake_redis.hgetall(key)  # type: ignore[misc]
        assert stored == encode_for_lua(LOCK_STATE_TRANSITIONS)

    async def test_removes_stale_entries_from_prior_schema(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        """A state present in a prior load but absent in the new table
        must not survive — the function clears the key before writing.
        """
        key = fsm_key("run-x", "program_state")
        await fake_redis.hset(  # type: ignore[misc]
            key, mapping={"LEGACY_STATE": "QUEUED,RUNNING", "QUEUED": "RUNNING"}
        )
        await load_fsm_table(
            fake_redis,
            key_prefix="run-x",
            name="program_state",
            table=PROGRAM_STATE_TRANSITIONS,
        )
        stored = await fake_redis.hgetall(key)  # type: ignore[misc]
        assert "LEGACY_STATE" not in stored
        assert stored == encode_for_lua(PROGRAM_STATE_TRANSITIONS)

    async def test_empty_table_clears_the_key(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        """An empty FSM table deletes any pre-existing entry so callers
        cannot observe a stale half-loaded state masquerading as the
        current spec.
        """
        key = fsm_key("run-x", "ephemeral")
        await fake_redis.hset(key, mapping={"OLD": "X,Y"})  # type: ignore[misc]
        await load_fsm_table(
            fake_redis,
            key_prefix="run-x",
            name="ephemeral",
            table={},
        )
        assert not await fake_redis.exists(key)  # type: ignore[misc]

    async def test_isolated_by_prefix(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        await load_fsm_table(
            fake_redis,
            key_prefix="run-a",
            name="program_state",
            table=PROGRAM_STATE_TRANSITIONS,
        )
        await load_fsm_table(
            fake_redis,
            key_prefix="run-b",
            name="program_state",
            table=CLAIM_STATE_TRANSITIONS,
        )
        a = await fake_redis.hgetall(fsm_key("run-a", "program_state"))  # type: ignore[misc]
        b = await fake_redis.hgetall(fsm_key("run-b", "program_state"))  # type: ignore[misc]
        assert a == encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        assert b == encode_for_lua(CLAIM_STATE_TRANSITIONS)
        assert a != b


# ── structural invariants ────────────────────────────────────────────


class TestRetiredHelpersAreAbsent:
    """Pin that the case-bridge helpers and dead emitters have been
    fully retired from the public surfaces.

    These three negative assertions are the structural guard against
    re-introduction by a future change: a contributor importing the
    name observes the failing test and either re-justifies the bridge
    or rewrites their patch to live without it.
    """

    def test_load_legacy_program_fsm_not_in_engine_startup(self) -> None:
        from gigaevo.dataplane import engine_startup

        assert not hasattr(engine_startup, "load_legacy_program_fsm"), (
            "the case-bridge reloader was retired in favour of the "
            "case-tolerant FSM hash emitted by encode_for_lua"
        )

    def test_load_legacy_program_fsm_not_exported_from_storage(self) -> None:
        from gigaevo.database import redis_program_storage

        assert not hasattr(redis_program_storage, "load_legacy_program_fsm"), (
            "load_legacy_program_fsm was the sticky-tape between an "
            "uppercase coordinator enum and a lowercase persisted blob; "
            "the case-tolerant FSM hash removed the case mismatch at "
            "the source so the helper must not return"
        )

    def test_publish_status_event_not_on_storage(self) -> None:
        from gigaevo.database.program_storage import ProgramStorage
        from gigaevo.database.redis_program_storage import RedisProgramStorage

        assert not hasattr(ProgramStorage, "publish_status_event"), (
            "publish_status_event had no production callers — the "
            "dataplane Lua emits status events server-side"
        )
        assert not hasattr(RedisProgramStorage, "publish_status_event"), (
            "publish_status_event had no production callers — the "
            "dataplane Lua emits status events server-side"
        )
