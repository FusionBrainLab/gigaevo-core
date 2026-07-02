from enum import StrEnum

from gigaevo.utils import state_machine


class StoreState(StrEnum):
    INITIALIZING = "initializing"
    READY = "ready"
    BUILDING = "building"
    ERROR = "error"


VALID_TRANSITIONS: dict[StoreState, set[StoreState]] = {
    StoreState.INITIALIZING: {StoreState.READY, StoreState.ERROR},
    StoreState.READY: {StoreState.BUILDING, StoreState.ERROR},
    StoreState.BUILDING: {StoreState.READY, StoreState.ERROR},
    StoreState.ERROR: {StoreState.INITIALIZING},
}


def is_valid_transition(current: StoreState, new: StoreState) -> bool:
    return state_machine.is_valid_transition(current, new, VALID_TRANSITIONS)


def validate_transition(current: StoreState, new: StoreState) -> None:
    state_machine.validate_transition(current, new, VALID_TRANSITIONS)
