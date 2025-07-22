from enum import Enum
from typing import Dict


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


# State hierarchy for merging logic (higher values = more advanced states)
# This defines the natural progression order of program states
STATE_HIERARCHY: Dict[ProgramState, int] = {
    ProgramState.FRESH: 0,
    ProgramState.DAG_PROCESSING_STARTED: 1,
    ProgramState.DAG_PROCESSING_COMPLETED: 2,
    ProgramState.EVOLVING: 3,
    ProgramState.DISCARDED: 4,  # Terminal state - highest priority
}


def get_state_priority(state: ProgramState) -> int:
    """Get the priority/hierarchy level of a program state.

    Higher values indicate more advanced states in the program lifecycle.
    This is used for merge operations to ensure states only advance, never regress.

    Args:
        state: The program state to get priority for

    Returns:
        Integer priority (higher = more advanced)

    Raises:
        ValueError: If the state is not in the hierarchy
    """
    if state not in STATE_HIERARCHY:
        raise ValueError(f"Unknown program state: {state}")
    return STATE_HIERARCHY[state]


def should_advance_state(
    current_state: ProgramState, new_state: ProgramState
) -> bool:
    """Determine if a state should be advanced from current to new.

    States can only advance forward in the hierarchy, never regress.

    Args:
        current_state: The current program state
        new_state: The proposed new program state

    Returns:
        True if the state should be advanced, False otherwise
    """
    return get_state_priority(new_state) > get_state_priority(current_state)


def merge_states(
    current_state: ProgramState, incoming_state: ProgramState
) -> ProgramState:
    """Merge two program states, taking the more advanced one.

    Args:
        current_state: The current program state
        incoming_state: The incoming program state

    Returns:
        The more advanced state according to the hierarchy
    """
    if get_state_priority(incoming_state) > get_state_priority(current_state):
        return incoming_state
    else:
        return current_state
