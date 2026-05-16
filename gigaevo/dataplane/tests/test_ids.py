"""Smoke tests for the ID taxonomy.

NewType aliases are compile-time only — they're indistinguishable from
their base types at runtime. The only meaningful runtime test is the
:func:`make_actor_id` helper.
"""

from __future__ import annotations

from gigaevo.dataplane.ids import ActorId, RunId, WorkerId, make_actor_id


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
