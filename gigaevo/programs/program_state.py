from enum import Enum


class ProgramState(str, Enum):
    """Lifecycle state of a Program object."""

    # Newly created – not yet sent to DAG
    FRESH = "fresh"

    # DAG execution has started but not yet finished
    DAG_PROCESSING_STARTED = "dag_processing_started"

    # DAG execution finished successfully
    DAG_PROCESSING_COMPLETED = "dag_processing_completed"

    # Program participates in evolution (selected/elites etc.)
    EVOLVING = "evolving"

    # Program explicitly discarded – excluded from any further processing
    DISCARDED = "discarded"


NEW_STATES_PROGRAM_LIFECYCLE = {
    ProgramState.FRESH,
    ProgramState.DAG_PROCESSING_STARTED,
}

FINAL_STATES_PROGRAM_LIFECYCLE = {
    ProgramState.DAG_PROCESSING_COMPLETED,
    ProgramState.EVOLVING,
    ProgramState.DISCARDED,
}

STATE_HIERARCHY: dict[ProgramState, int] = {
    ProgramState.FRESH: 0,
    ProgramState.DAG_PROCESSING_STARTED: 1,
    ProgramState.DAG_PROCESSING_COMPLETED: 2,
    ProgramState.EVOLVING: 3,
    ProgramState.DISCARDED: 99,  # Terminal state - highest priority
}


def get_state_priority(state: ProgramState) -> int:
    if state not in STATE_HIERARCHY:
        raise ValueError(f"Unknown program state: {state}")
    return STATE_HIERARCHY[state]


def should_advance_state(current_state: ProgramState, new_state: ProgramState) -> bool:
    return get_state_priority(new_state) > get_state_priority(current_state)


def merge_states(
    current_state: ProgramState, incoming_state: ProgramState
) -> ProgramState:
    """Merge two program states, taking the more advanced one.

    Special case: Allow bidirectional transitions between EVOLVING and FRESH
    for refresh purposes (EVOLVING -> FRESH for refresh, FRESH -> EVOLVING after refresh).

    Args:
        current_state: The current program state
        incoming_state: The incoming program state

    Returns:
        The more advanced state according to the hierarchy, or incoming state
        for EVOLVING <-> FRESH transitions
    """
    # Special case: Allow bidirectional EVOLVING <-> FRESH transitions for refresh
    if (
        current_state == ProgramState.EVOLVING and incoming_state == ProgramState.FRESH
    ) or (
        current_state == ProgramState.FRESH and incoming_state == ProgramState.EVOLVING
    ):
        return incoming_state

    # Normal hierarchy-based merging for all other cases
    if get_state_priority(incoming_state) > get_state_priority(current_state):
        return incoming_state
    else:
        return current_state
