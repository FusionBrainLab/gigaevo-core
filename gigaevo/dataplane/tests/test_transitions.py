"""Tests for the FSM tables and encoding helpers."""

from __future__ import annotations

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
)


class TestProgramStateFSM:
    def test_queued_to_running_legal(self) -> None:
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ProgramState.QUEUED,
            ProgramState.RUNNING,
        )

    def test_queued_to_done_illegal(self) -> None:
        # QUEUED cannot jump to DONE — must go through RUNNING.
        assert not is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ProgramState.QUEUED,
            ProgramState.DONE,
        )

    def test_running_to_done_legal(self) -> None:
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ProgramState.RUNNING,
            ProgramState.DONE,
        )

    def test_done_to_queued_legal_reevaluation(self) -> None:
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ProgramState.DONE,
            ProgramState.QUEUED,
        )

    def test_discarded_terminal(self) -> None:
        # DISCARDED can only stay DISCARDED.
        for target in (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE):
            assert not is_valid_transition(
                PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
                ProgramState.DISCARDED,
                target,
            )
        assert is_valid_transition(
            PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ProgramState.DISCARDED,
            ProgramState.DISCARDED,
        )

    def test_self_transitions_allowed(self) -> None:
        # Every non-terminal state allows a self-transition (idempotent re-emit).
        for s in (ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE):
            assert is_valid_transition(
                PROGRAM_STATE_TRANSITIONS,  # type: ignore[arg-type]
                s,
                s,
            )


class TestClaimStateFSM:
    def test_unclaimed_to_claimed(self) -> None:
        assert is_valid_transition(
            CLAIM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ClaimState.UNCLAIMED,
            ClaimState.CLAIMED,
        )

    def test_claimed_to_acked_or_expired(self) -> None:
        for target in (ClaimState.ACKED, ClaimState.EXPIRED):
            assert is_valid_transition(
                CLAIM_STATE_TRANSITIONS,  # type: ignore[arg-type]
                ClaimState.CLAIMED,
                target,
            )

    def test_acked_terminal(self) -> None:
        for target in ClaimState:
            assert not is_valid_transition(
                CLAIM_STATE_TRANSITIONS,  # type: ignore[arg-type]
                ClaimState.ACKED,
                target,
            )

    def test_expired_can_reenter_unclaimed(self) -> None:
        assert is_valid_transition(
            CLAIM_STATE_TRANSITIONS,  # type: ignore[arg-type]
            ClaimState.EXPIRED,
            ClaimState.UNCLAIMED,
        )


class TestLockStateFSM:
    def test_held_to_renewed_released_lost(self) -> None:
        for target in (LockState.RENEWED, LockState.RELEASED, LockState.LOST):
            assert is_valid_transition(
                LOCK_STATE_TRANSITIONS,  # type: ignore[arg-type]
                LockState.HELD,
                target,
            )

    def test_renewed_can_repeat(self) -> None:
        # Successive renewals stay in RENEWED.
        assert is_valid_transition(
            LOCK_STATE_TRANSITIONS,  # type: ignore[arg-type]
            LockState.RENEWED,
            LockState.RENEWED,
        )

    def test_released_and_lost_are_terminal(self) -> None:
        for terminal in (LockState.RELEASED, LockState.LOST):
            for target in LockState:
                assert not is_valid_transition(
                    LOCK_STATE_TRANSITIONS,  # type: ignore[arg-type]
                    terminal,
                    target,
                )


class TestEncodeForLua:
    def test_encodes_each_from_state(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)  # type: ignore[arg-type]
        # Every from-state with at least one allowed target appears.
        for from_state in PROGRAM_STATE_TRANSITIONS:
            assert from_state.value in encoded

    def test_targets_comma_joined_and_sorted(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)  # type: ignore[arg-type]
        queued_targets = encoded["QUEUED"]
        parts = queued_targets.split(",")
        assert parts == sorted(parts)
        assert set(parts) == {"DISCARDED", "QUEUED", "RUNNING"}

    def test_discarded_self_only(self) -> None:
        encoded = encode_for_lua(PROGRAM_STATE_TRANSITIONS)  # type: ignore[arg-type]
        assert encoded["DISCARDED"] == "DISCARDED"


class TestFsmKey:
    def test_prefixed(self) -> None:
        assert fsm_key("run-7", "program_state") == "run-7:fsm:program_state"

    def test_different_prefixes_isolate(self) -> None:
        a = fsm_key("run-a", "x")
        b = fsm_key("run-b", "x")
        assert a != b
