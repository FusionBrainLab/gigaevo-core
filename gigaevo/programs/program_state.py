from enum import Enum


class ProgramState(str, Enum):
    FRESH = "fresh"
    DAG_PROCESSING_STARTED = "dag_processing_started"
    DAG_PROCESSING_COMPLETED = "dag_processing_completed"
    EVOLVING = "evolving"
    DISCARDED = "discarded"


INCOMPLETE_STATES = {
    ProgramState.FRESH,
    ProgramState.DAG_PROCESSING_STARTED,
}

COMPLETE_STATES = {
    ProgramState.DAG_PROCESSING_COMPLETED,
    ProgramState.EVOLVING,
}

TERMINAL_STATES = {
    ProgramState.DISCARDED,
}

STATES_WITH_METRICS = {
    ProgramState.DAG_PROCESSING_COMPLETED,
    ProgramState.EVOLVING,
}

VALID_TRANSITIONS: dict[ProgramState, set[ProgramState]] = {
    ProgramState.FRESH: {
        ProgramState.DAG_PROCESSING_STARTED,
        ProgramState.DISCARDED,
    },
    ProgramState.DAG_PROCESSING_STARTED: {
        ProgramState.DAG_PROCESSING_COMPLETED,
        ProgramState.DISCARDED,
    },
    ProgramState.DAG_PROCESSING_COMPLETED: {
        ProgramState.EVOLVING,
        ProgramState.DISCARDED,
    },
    ProgramState.EVOLVING: {
        ProgramState.FRESH,
        ProgramState.DISCARDED,
    },
    ProgramState.DISCARDED: set(),
}


def is_valid_transition(current: ProgramState, new: ProgramState) -> bool:
    if current == new:
        return True
    return new in VALID_TRANSITIONS.get(current, set())


def validate_transition(current: ProgramState, new: ProgramState) -> None:
    if not is_valid_transition(current, new):
        valid_next = VALID_TRANSITIONS.get(current, set())
        raise ValueError(
            f"Invalid state transition: {current.value} -> {new.value}. "
            f"Valid transitions from {current.value}: {[s.value for s in valid_next]}"
        )


def is_incomplete(state: ProgramState) -> bool:
    return state in INCOMPLETE_STATES


def is_complete(state: ProgramState) -> bool:
    return state in COMPLETE_STATES


def is_terminal(state: ProgramState) -> bool:
    return state in TERMINAL_STATES


def has_metrics(state: ProgramState) -> bool:
    return state in STATES_WITH_METRICS


def merge_states(
    current_state: ProgramState, incoming_state: ProgramState
) -> ProgramState:
    if current_state == incoming_state:
        return current_state

    if incoming_state in TERMINAL_STATES:
        return incoming_state
    if current_state in TERMINAL_STATES:
        return current_state

    if is_valid_transition(current_state, incoming_state):
        return incoming_state
    if is_valid_transition(incoming_state, current_state):
        return current_state

    raise ValueError(
        f"Cannot merge incompatible states: {current_state.value} vs {incoming_state.value}"
    )
