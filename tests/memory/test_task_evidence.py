from __future__ import annotations

from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.context.evidence import sign_help_counts, split_events_by_task


def _event(task_key: str) -> ContextualGain:
    return ContextualGain(context=DecisionContext(task_key=task_key), gain=0.1)


def test_split_events_by_task_uses_exact_task_key_equality():
    legacy = _event("")
    native = _event("task-a")
    foreign = _event("task-b")

    assert split_events_by_task((legacy, native, foreign), "task-a") == (
        (native,),
        (legacy, foreign),
    )
    assert split_events_by_task((legacy, native, foreign), "") == (
        (legacy,),
        (native, foreign),
    )


def test_sign_help_counts_is_invariant_to_same_sign_magnitude_and_se():
    tiny_noisy = ContextualGain(
        context=DecisionContext(task_key="task-b"), gain=1e-300, gain_se=1e300
    )
    huge_exact = ContextualGain(
        context=DecisionContext(task_key="task-b"), gain=1e300, gain_se=0.0
    )
    tiny_loss = ContextualGain(
        context=DecisionContext(task_key="task-b"), gain=-1e-300, gain_se=1e300
    )
    huge_loss = ContextualGain(
        context=DecisionContext(task_key="task-b"), gain=-1e300, gain_se=0.0
    )

    assert sign_help_counts((tiny_noisy, tiny_loss)) == (1.0, 2.0)
    assert sign_help_counts((huge_exact, huge_loss)) == (1.0, 2.0)
