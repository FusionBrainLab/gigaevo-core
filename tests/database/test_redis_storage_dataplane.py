"""Storage tests covering the optional dataplane-routed transition path.

When :class:`RedisProgramStorage` is constructed with a ``dataplane``
parameter, :meth:`atomic_state_transition` and
:meth:`fast_state_transition` route through
:meth:`gigaevo.dataplane.DataPlane.transition_program_state` instead of
the legacy WATCH / MULTI / EXEC pipeline. The coordinator's
``transition_state.lua`` script validates the (from, to) pair against
the FSM hash, merges the patch into the persisted blob, updates the
status set, and emits an audit-stream event in one atomic round-trip.

The coordinator ships an uppercase ``ProgramState`` enum whose values
key the FSM hash; the legacy blob format predates the coordinator and
uses lowercase. :func:`load_legacy_program_fsm` reloads the FSM hash
with lowercase keys; these tests exercise the routed path against the
reloaded table.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import uuid

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    load_legacy_program_fsm,
)
from gigaevo.database.state_manager import ProgramStateManager
import gigaevo.dataplane as dp
from gigaevo.exceptions import StorageError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


@pytest.fixture
async def dp_storage() -> AsyncIterator[tuple[RedisProgramStorage, dp.DataPlane]]:
    """Storage backed by fakeredis with a dataplane wired against the same fake.

    Both the storage's ``RedisConnection`` and the dataplane share the
    same ``fakeredis.FakeServer`` so the dp-routed transition writes the
    blob, status set, and event stream onto the keys the storage's read
    helpers consult.
    """
    server = fakeredis.FakeServer()

    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="testdp",
    )
    storage = RedisProgramStorage(config)
    storage_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = storage_redis  # type: ignore[attr-defined]
    storage._conn._closing = False  # type: ignore[attr-defined]

    coord = dp.DataPlane("redis://fake:6379/0", key_prefix="testdp")
    dp_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = dp_redis  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry
    from gigaevo.dataplane.transitions import (
        PROGRAM_STATE_TRANSITIONS,
        load_fsm_table,
    )

    lua = LuaRegistry(dp_redis)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    await load_fsm_table(
        dp_redis,
        key_prefix="testdp",
        name="program_state",
        table=PROGRAM_STATE_TRANSITIONS,
    )
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    # Reload the FSM with lowercase keys so the Lua's HGET succeeds
    # against the legacy blob format.
    await load_legacy_program_fsm(coord)

    storage._dataplane = coord  # type: ignore[attr-defined]
    try:
        yield storage, coord
    finally:
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await dp_redis.aclose()  # type: ignore[attr-defined]
        await storage.close()


def _program(state: ProgramState = ProgramState.QUEUED) -> Program:
    return Program(
        id=str(uuid.uuid4()),
        code="def solve(): return 0",
        state=state,
    )


class TestDpRoutedTransition:
    async def test_fast_transition_routes_through_dataplane(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """A QUEUED → RUNNING transition under dp routing updates state,
        status set membership, and the post-call in-memory counter."""
        storage, coord = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)

        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        # Persisted blob has the new state.
        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING
        # Status sets updated (status:queued lost the id, status:running gained it).
        running_ids = await storage.get_ids_by_status(ProgramState.RUNNING.value)
        queued_ids = await storage.get_ids_by_status(ProgramState.QUEUED.value)
        assert prog.id in running_ids
        assert prog.id not in queued_ids
        # Counter monotonically advanced.
        assert fetched.atomic_counter > 0

    async def test_atomic_transition_routes_through_dataplane(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Atomic transition path also routes when dp is present."""
        storage, _ = dp_storage
        prog = _program(ProgramState.RUNNING)
        await storage.add(prog)

        await storage.atomic_state_transition(
            prog, ProgramState.RUNNING.value, ProgramState.DONE.value
        )

        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.DONE

    async def test_illegal_transition_raises_storage_error(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The FSM rejects QUEUED → DONE; the dp wrapper surfaces it as
        a typed :class:`StorageError` so the legacy bypass that previously
        let illegal pairs slip past now fails loudly (bug class #14)."""
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)

        with pytest.raises(StorageError) as excinfo:
            await storage.fast_state_transition(
                prog, ProgramState.QUEUED.value, ProgramState.DONE.value
            )
        assert "illegal" in str(excinfo.value)

    async def test_stale_expected_from_raises_storage_error(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """When ``old_state`` does not match the persisted state, the dp
        returns the ``stale`` variant which surfaces as :class:`StorageError`.

        The legal-but-stale pair ``(DONE, QUEUED)`` is used so the FSM
        membership check passes — the dp's lua compares ``expected_from``
        to the observed blob state and fires the stale branch when the
        caller's pre-image is older than the persisted state.
        """
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        # The blob is QUEUED; the caller asserts ``old=DONE``. The dp
        # rejects the call as stale before applying the patch.
        with pytest.raises(StorageError) as excinfo:
            await storage.fast_state_transition(
                prog, ProgramState.DONE.value, ProgramState.QUEUED.value
            )
        assert "stale" in str(excinfo.value)

    async def test_audit_stream_emits_transition_event(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The dp-routed path emits the same status_events stream the
        legacy path appends to, so existing readers continue to observe
        every transition."""
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        r = await storage._conn.get()
        entries = await r.xrange(storage._keys.status_stream(), count=10)
        # ``add`` emits one ``created`` event; the dp transition appends
        # another. Both arrive on the same status_events stream key.
        assert len(entries) >= 2

    async def test_transition_event_payload_carries_legacy_and_dp_fields(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The Lua-emitted XADD payload covers both wire shapes.

        Legacy readers key on ``id`` / ``status`` / ``event``; dp-aware
        readers consume the (``pid``, ``from``, ``to``, ``epoch``)
        triple. A single XADD inside the Lua carries both so neither
        reader side needs a migration.
        """
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        r = await storage._conn.get()
        entries = await r.xrange(storage._keys.status_stream(), count=10)
        transition_entries = [
            fields
            for _, fields in entries
            if fields.get("event") == "transition" and fields.get("id") == prog.id
        ]
        assert len(transition_entries) == 1, (
            f"expected exactly one transition event for {prog.id}, "
            f"observed {len(transition_entries)} (stream: {entries!r})"
        )
        payload = transition_entries[0]
        # Legacy schema.
        assert payload["id"] == prog.id
        assert payload["status"] == ProgramState.RUNNING.value
        assert payload["event"] == "transition"
        # dp-native schema.
        assert payload["pid"] == prog.id
        assert payload["from"] == ProgramState.QUEUED.value
        assert payload["to"] == ProgramState.RUNNING.value
        assert int(payload["epoch"]) >= 1

    async def test_dp_path_emits_exactly_one_event_per_transition(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Pin the single-emit invariant on the dp-routed path.

        ``transition_state.lua`` is the sole event source; the wrapper
        :meth:`RedisProgramStorage._transition_via_dataplane` does not
        call :meth:`publish_status_event` after the script returns, so
        the per-transition event count on the status stream is exactly
        one. The test exercises three transitions and counts events
        keyed on ``event == "transition"`` matching the program id.
        """
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)

        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        await storage.atomic_state_transition(
            prog, ProgramState.RUNNING.value, ProgramState.DONE.value
        )
        await storage.atomic_state_transition(
            prog, ProgramState.DONE.value, ProgramState.QUEUED.value
        )

        r = await storage._conn.get()
        entries = await r.xrange(storage._keys.status_stream(), count=100)
        transition_count = sum(
            1
            for _, fields in entries
            if fields.get("event") == "transition" and fields.get("id") == prog.id
        )
        assert transition_count == 3, (
            f"expected one event per transition (3 total), observed "
            f"{transition_count} (stream: {entries!r})"
        )

    async def test_dp_path_preserves_empty_dict_fields(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Empty dict fields survive a dp-routed transition as ``{}``.

        The Lua re-encodes the merged blob with ``cjson.encode``; on
        Redis builds shipping lua-cjson 2.1+, the directive
        ``cjson.encode_empty_table_as_object(true)`` at the top of
        ``transition_state.lua`` keeps empty objects as ``{}``. On
        environments without the directive (e.g. fakeredis's embedded
        Lua VM), :meth:`RedisProgramStorage._safe_deserialize`'s coercion
        recovers the dict shape. Either way, the round-tripped Program
        instance has dict-typed empty fields, not list-typed.
        """
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        # All three dict-typed fields start empty on a freshly-built
        # Program; explicit assignment pins the invariant against future
        # default-factory changes.
        prog.metrics = {}
        prog.stage_results = {}
        prog.metadata = {}
        await storage.add(prog)

        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.metrics == {} and isinstance(fetched.metrics, dict)
        assert fetched.stage_results == {} and isinstance(fetched.stage_results, dict)
        assert fetched.metadata == {} and isinstance(fetched.metadata, dict)

    async def test_dp_path_stamps_atomic_counter_in_blob(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The Lua stamps both ``atomic_counter`` and ``epoch`` on the blob.

        The persisted blob carries the legacy field name so dp-routed
        writes are pass-through for any reader that keys on
        ``atomic_counter`` (the gigaevo Program merge tiebreaker). The
        read path then strips ``epoch`` to satisfy the Pydantic model's
        ``extra="forbid"`` constraint.
        """
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        r = await storage._conn.get()
        raw = await r.get(storage._keys.program(prog.id))
        assert raw is not None
        import json as _json

        blob = _json.loads(raw)
        assert "atomic_counter" in blob, (
            f"Lua must stamp atomic_counter on the post-transition blob "
            f"so legacy readers see the post-INCR counter without a "
            f"read-side rename; got keys={sorted(blob)!r}"
        )
        assert "epoch" in blob, (
            f"Lua must also stamp epoch for dp-aware consumers; "
            f"got keys={sorted(blob)!r}"
        )
        assert blob["atomic_counter"] == blob["epoch"]
        assert isinstance(blob["atomic_counter"], int)
        assert blob["atomic_counter"] >= 1


class TestLegacyPathUnchanged:
    async def test_default_storage_has_no_dataplane(self) -> None:
        """Storage constructed without ``dataplane`` keeps the legacy
        WATCH/MULTI/EXEC path, leaving fakeredis-only tests and any
        read-only / Hydra-instantiated storage operational unchanged."""
        server = fakeredis.FakeServer()
        config = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0", key_prefix="legacy"
        )
        storage = RedisProgramStorage(config)
        storage._conn._redis = fakeredis.aioredis.FakeRedis(  # type: ignore[attr-defined]
            server=server, decode_responses=True
        )
        storage._conn._closing = False  # type: ignore[attr-defined]
        try:
            assert storage._dataplane is None
            prog = _program(ProgramState.QUEUED)
            await storage.add(prog)
            # Legacy fast_state_transition reads ``program.state`` from
            # the caller-supplied object; the caller is responsible for
            # updating the in-memory state before persisting (see
            # :meth:`ProgramStateManager.set_program_state`). The
            # dataplane-routed path mirrors this update internally, but
            # the legacy path does not.
            prog.state = ProgramState.RUNNING
            await storage.fast_state_transition(
                prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
            )
            fetched = await storage.get(prog.id)
            assert fetched is not None
            assert fetched.state == ProgramState.RUNNING
        finally:
            await storage.close()


class TestStateManagerInMemoryHelper:
    async def test_set_in_memory_state_validates_and_assigns(
        self, state_manager: ProgramStateManager, make_program
    ) -> None:
        """The mirror helper validates the FSM transition and updates
        ``program.state`` without writing to storage; call sites that
        already persisted via a batch op rely on this to keep the
        in-memory Program object in sync."""
        prog = make_program(state=ProgramState.QUEUED)
        await state_manager.set_in_memory_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING

    async def test_set_in_memory_state_rejects_illegal_transition(
        self, state_manager: ProgramStateManager, make_program
    ) -> None:
        """An illegal mirror write (e.g. QUEUED → DONE) raises
        :class:`ValueError` instead of silently desyncing the in-memory
        state — bug class #14 caught at the source of the bypass."""
        prog = make_program(state=ProgramState.QUEUED)
        with pytest.raises(ValueError):
            await state_manager.set_in_memory_state(prog, ProgramState.DONE)
        # State must remain at the pre-call value on rejection.
        assert prog.state == ProgramState.QUEUED

    async def test_set_in_memory_state_self_loop_is_noop(
        self, state_manager: ProgramStateManager, make_program
    ) -> None:
        """Same-state assignment short-circuits without raising; this
        matches :meth:`set_program_state`'s idempotent semantics so the
        helper is safe to call inside a re-entrant ingestion loop."""
        prog = make_program(state=ProgramState.RUNNING)
        await state_manager.set_in_memory_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING
