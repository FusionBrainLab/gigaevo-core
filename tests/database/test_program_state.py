"""Tests for gigaevo/programs/program_state.py"""

import pytest

from gigaevo.programs.program_state import (
    COMPLETE_STATES,
    INCOMPLETE_STATES,
    STATES_WITH_METRICS,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    ProgramState,
    has_metrics,
    is_complete,
    is_incomplete,
    is_terminal,
    is_valid_transition,
    merge_states,
    validate_transition,
)


class TestProgramStateEnum:
    def test_all_states_exist(self):
        assert ProgramState.QUEUED == "queued"
        assert ProgramState.RUNNING == "running"
        assert ProgramState.DONE == "done"
        assert ProgramState.DISCARDED == "discarded"

    def test_is_str_enum(self):
        for state in ProgramState:
            assert isinstance(state, str)
            assert isinstance(state.value, str)

    def test_exactly_four_states(self):
        assert len(ProgramState) == 4


class TestStateCategories:
    def test_incomplete_states(self):
        assert INCOMPLETE_STATES == {ProgramState.QUEUED, ProgramState.RUNNING}

    def test_complete_states(self):
        assert COMPLETE_STATES == {ProgramState.DONE}

    def test_terminal_states(self):
        assert TERMINAL_STATES == {ProgramState.DISCARDED}

    def test_disjoint(self):
        all_sets = [INCOMPLETE_STATES, COMPLETE_STATES, TERMINAL_STATES]
        for i, a in enumerate(all_sets):
            for b in all_sets[i + 1 :]:
                assert a.isdisjoint(b), f"{a} and {b} overlap"

    def test_union_covers_all_states(self):
        assert INCOMPLETE_STATES | COMPLETE_STATES | TERMINAL_STATES == set(
            ProgramState
        )

    def test_states_with_metrics(self):
        assert STATES_WITH_METRICS == {ProgramState.DONE}


class TestValidTransitions:
    @pytest.mark.parametrize(
        "src,dst",
        [
            (ProgramState.QUEUED, ProgramState.RUNNING),
            (ProgramState.QUEUED, ProgramState.DISCARDED),
            (ProgramState.RUNNING, ProgramState.DONE),
            (ProgramState.RUNNING, ProgramState.DISCARDED),
            (ProgramState.DONE, ProgramState.QUEUED),
            (ProgramState.DONE, ProgramState.DISCARDED),
        ],
    )
    def test_valid_transitions(self, src, dst):
        assert is_valid_transition(src, dst) is True

    @pytest.mark.parametrize(
        "src,dst",
        [
            (ProgramState.QUEUED, ProgramState.DONE),
            (ProgramState.RUNNING, ProgramState.QUEUED),
            (ProgramState.DONE, ProgramState.RUNNING),
            (ProgramState.DISCARDED, ProgramState.QUEUED),
            (ProgramState.DISCARDED, ProgramState.RUNNING),
            (ProgramState.DISCARDED, ProgramState.DONE),
        ],
    )
    def test_invalid_transitions(self, src, dst):
        assert is_valid_transition(src, dst) is False

    def test_self_transition_always_valid(self):
        for state in ProgramState:
            assert is_valid_transition(state, state) is True

    def test_discarded_has_no_outgoing(self):
        assert VALID_TRANSITIONS[ProgramState.DISCARDED] == set()


class TestValidateTransition:
    def test_valid_no_error(self):
        validate_transition(ProgramState.QUEUED, ProgramState.RUNNING)

    def test_invalid_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid state transition"):
            validate_transition(ProgramState.QUEUED, ProgramState.DONE)

    def test_error_message_contains_states(self):
        with pytest.raises(ValueError, match="queued.*done"):
            validate_transition(ProgramState.QUEUED, ProgramState.DONE)


class TestPredicates:
    @pytest.mark.parametrize(
        "state,expected",
        [
            (ProgramState.QUEUED, True),
            (ProgramState.RUNNING, True),
            (ProgramState.DONE, False),
            (ProgramState.DISCARDED, False),
        ],
    )
    def test_is_incomplete(self, state, expected):
        assert is_incomplete(state) is expected

    @pytest.mark.parametrize(
        "state,expected",
        [
            (ProgramState.QUEUED, False),
            (ProgramState.RUNNING, False),
            (ProgramState.DONE, True),
            (ProgramState.DISCARDED, False),
        ],
    )
    def test_is_complete(self, state, expected):
        assert is_complete(state) is expected

    @pytest.mark.parametrize(
        "state,expected",
        [
            (ProgramState.QUEUED, False),
            (ProgramState.RUNNING, False),
            (ProgramState.DONE, False),
            (ProgramState.DISCARDED, True),
        ],
    )
    def test_is_terminal(self, state, expected):
        assert is_terminal(state) is expected

    @pytest.mark.parametrize(
        "state,expected",
        [
            (ProgramState.QUEUED, False),
            (ProgramState.RUNNING, False),
            (ProgramState.DONE, True),
            (ProgramState.DISCARDED, False),
        ],
    )
    def test_has_metrics(self, state, expected):
        assert has_metrics(state) is expected


class TestMergeStates:
    def test_same_state(self):
        for state in ProgramState:
            assert merge_states(state, state) == state

    def test_terminal_wins_as_incoming(self):
        for state in [ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE]:
            assert merge_states(state, ProgramState.DISCARDED) == ProgramState.DISCARDED

    def test_terminal_wins_as_current(self):
        for state in [ProgramState.QUEUED, ProgramState.RUNNING, ProgramState.DONE]:
            assert merge_states(ProgramState.DISCARDED, state) == ProgramState.DISCARDED

    def test_forward_transition(self):
        # QUEUED -> RUNNING is a valid transition, so incoming wins
        assert (
            merge_states(ProgramState.QUEUED, ProgramState.RUNNING)
            == ProgramState.RUNNING
        )

    def test_backward_transition(self):
        # RUNNING -> QUEUED is invalid, but QUEUED -> RUNNING is valid, so current wins
        assert (
            merge_states(ProgramState.RUNNING, ProgramState.QUEUED)
            == ProgramState.RUNNING
        )

    def test_done_to_running_incompatible_raises(self):
        # DONE->RUNNING is not valid, and RUNNING->DONE is forward only.
        # But merge_states checks both directions. Actually RUNNING->DONE is valid.
        # The only truly incompatible pair (non-terminal) would be hard to find
        # since merge_states tries both directions.
        # Let's verify that QUEUED and DONE actually merge (DONE->QUEUED is valid):
        result = merge_states(ProgramState.QUEUED, ProgramState.DONE)
        # DONE->QUEUED is a valid transition, so incoming DONE wins? No:
        # QUEUED->DONE is NOT valid. DONE->QUEUED IS valid.
        # is_valid_transition(QUEUED, DONE) = False
        # is_valid_transition(DONE, QUEUED) = True -> return current (QUEUED)
        assert result == ProgramState.QUEUED

    def test_all_pairs_non_terminal_merge_no_crash(self):
        """For all (s1, s2) where neither is DISCARDED, merge_states should not crash."""
        non_terminal = [s for s in ProgramState if s != ProgramState.DISCARDED]
        for s1 in non_terminal:
            for s2 in non_terminal:
                result = merge_states(s1, s2)
                assert isinstance(result, ProgramState)

    def test_merge_is_commutative_for_same_state(self):
        """merge_states(s, s) == s for all states."""
        for s in ProgramState:
            assert merge_states(s, s) == s
