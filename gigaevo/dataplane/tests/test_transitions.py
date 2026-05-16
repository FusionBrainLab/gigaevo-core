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
    def test_encodes_each_from_state(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        # Every from-state with at least one allowed target appears.
        for from_state in PROGRAM_STATE_TRANSITIONS:
            assert from_state.value in encoded

    def test_targets_comma_joined_and_sorted(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        queued_targets = encoded["QUEUED"]
        parts = queued_targets.split(",")
        assert parts == sorted(parts)
        assert set(parts) == {"DISCARDED", "QUEUED", "RUNNING"}

    def test_discarded_self_only(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)
        assert encoded["DISCARDED"] == "DISCARDED"


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
