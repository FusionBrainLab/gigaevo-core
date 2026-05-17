"""Storage tests covering the optional dataplane-routed transition path.

With ``dataplane`` wired in, :meth:`atomic_state_transition` and
:meth:`fast_state_transition` route through
:meth:`DataPlane.transition_program_state` instead of the WATCH / MULTI
/ EXEC pipeline. The ``transition_state.lua`` script validates the
(from, to) pair against the FSM hash, merges the patch into the
persisted blob, updates the status set, and emits an audit-stream event
in one atomic round-trip.

The FSM hash is case-tolerant: every row is written under both
uppercase and lowercase keys so callers serializing either form resolve
the same transition.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import uuid

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
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
        """FSM rejects QUEUED → DONE; the dp wrapper surfaces it as a
        typed :class:`StorageError`."""
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
        returns the ``stale`` variant which surfaces as :class:`StorageError`."""
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        with pytest.raises(StorageError) as excinfo:
            await storage.fast_state_transition(
                prog, ProgramState.DONE.value, ProgramState.QUEUED.value
            )
        assert "stale" in str(excinfo.value)

    async def test_audit_stream_emits_transition_event(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The dp-routed path appends to the same status_events stream
        as the non-dp path."""
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
        """The XADD payload carries both wire shapes: the
        ``id``/``status``/``event`` triple and the
        ``pid``/``from``/``to``/``epoch`` triple."""
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
        :meth:`RedisProgramStorage._transition_via_dataplane` issues no
        additional XADD after the script returns, so the per-transition
        event count on the status stream is exactly one. The test
        exercises three transitions and counts events keyed on
        ``event == "transition"`` matching the program id.
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
        """The Lua stamps both ``atomic_counter`` and ``epoch`` on the
        blob so readers keying on either field see a consistent value.
        The read path strips ``epoch`` to satisfy the Pydantic model's
        ``extra="forbid"`` constraint."""
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
            f"so readers see the post-INCR counter; got keys={sorted(blob)!r}"
        )
        assert "epoch" in blob, (
            f"Lua must also stamp epoch for dp-aware consumers; "
            f"got keys={sorted(blob)!r}"
        )
        assert blob["atomic_counter"] == blob["epoch"]
        assert isinstance(blob["atomic_counter"], int)
        assert blob["atomic_counter"] >= 1


class TestCaseTolerantFsmVocabulary:
    """The FSM hash resolves either case: a dp-routed transition
    succeeds whether the persisted blob carries a lowercase ``state``
    or the uppercase dataplane value."""

    async def test_lowercase_blob_resolves_through_dp(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """A lowercase persisted state value advances via the dp path
        because case tolerance lives in the FSM hash itself."""
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING

    async def test_persisted_state_value_stays_lowercase(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The dp-routed write preserves the application-layer
        lowercase vocabulary in the on-disk blob."""
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
        assert blob["state"] == "running"


class TestSafeDeserializeNoRename:
    """``_safe_deserialize`` does the schema-boundary ``epoch`` strip
    plus the fakeredis-only empty-dict fallback."""

    def test_no_epoch_promotion_to_atomic_counter(self) -> None:
        """A blob carrying only ``epoch`` does not synthesise
        ``atomic_counter`` from it; the model default applies."""
        from gigaevo.utils.json import dumps as _dumps

        blob = {
            "id": str(uuid.uuid4()),
            "code": "def f(): pass",
            "state": "queued",
            "epoch": 99,
        }
        raw = _dumps(blob)
        prog = RedisProgramStorage._safe_deserialize(raw, ctx="no-rename")
        assert prog is not None
        assert prog.atomic_counter != 99

    def test_empty_dict_round_trip_without_lua_directive(self) -> None:
        """The fakeredis-only fallback recovers list-typed empty fields
        as ``{}`` so the Pydantic model loads cleanly."""
        from gigaevo.utils.json import dumps as _dumps

        blob = {
            "id": str(uuid.uuid4()),
            "code": "def f(): pass",
            "state": "queued",
            "metrics": [],
            "stage_results": [],
            "metadata": [],
            "atomic_counter": 1,
        }
        raw = _dumps(blob)
        prog = RedisProgramStorage._safe_deserialize(raw, ctx="empty-dict")
        assert prog is not None
        assert prog.metrics == {} and isinstance(prog.metrics, dict)
        assert prog.stage_results == {} and isinstance(prog.stage_results, dict)
        assert prog.metadata == {} and isinstance(prog.metadata, dict)


class TestLegacyPathUnchanged:
    async def test_default_storage_has_no_dataplane(self) -> None:
        """Storage without a wired ``dataplane`` uses the
        WATCH/MULTI/EXEC path."""
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
            # Without the dataplane mirror, the caller updates
            # ``program.state`` before persisting.
            prog.state = ProgramState.RUNNING
            await storage.fast_state_transition(
                prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
            )
            fetched = await storage.get(prog.id)
            assert fetched is not None
            assert fetched.state == ProgramState.RUNNING
        finally:
            await storage.close()


class TestEngineRootStorageWiring:
    """Engine-root threading: per-call tokens derive by linear split."""

    @pytest.fixture
    async def dp_storage_with_root(
        self,
    ) -> AsyncIterator[tuple[RedisProgramStorage, dp.DataPlane, dp.EngineRoot]]:
        """Same wiring as :func:`dp_storage` but with an
        :class:`EngineRoot` threaded into storage via the constructor."""
        server = fakeredis.FakeServer()
        config = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0",
            key_prefix="testroot",
        )
        engine_root = dp.build_engine_root()
        storage = RedisProgramStorage(config, engine_root=engine_root)
        storage_redis = fakeredis.aioredis.FakeRedis(
            server=server, decode_responses=True
        )
        storage._conn._redis = storage_redis  # type: ignore[attr-defined]
        storage._conn._closing = False  # type: ignore[attr-defined]

        coord = dp.DataPlane("redis://fake:6379/0", key_prefix="testroot")
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
            key_prefix="testroot",
            name="program_state",
            table=PROGRAM_STATE_TRANSITIONS,
        )
        coord._lua = lua  # type: ignore[attr-defined]
        coord._started = True  # type: ignore[attr-defined]

        storage._dataplane = coord  # type: ignore[attr-defined]
        try:
            yield storage, coord, engine_root
        finally:
            coord._started = False  # type: ignore[attr-defined]
            coord._lua = None  # type: ignore[attr-defined]
            coord._connection._pool = None  # type: ignore[attr-defined]
            await dp_redis.aclose()  # type: ignore[attr-defined]
            await storage.close()

    async def test_engine_root_rotates_across_consecutive_transitions(
        self,
        dp_storage_with_root: tuple[RedisProgramStorage, dp.DataPlane, dp.EngineRoot],
    ) -> None:
        """Two transitions both succeed because the engine root rotates
        its long-lived witness on each split."""
        storage, _, engine_root = dp_storage_with_root
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)

        initial_root = engine_root._program_root  # type: ignore[attr-defined]
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        assert initial_root.consumed

        rotated_root = engine_root._program_root  # type: ignore[attr-defined]
        assert rotated_root is not initial_root
        assert not rotated_root.consumed

        await storage.atomic_state_transition(
            prog, ProgramState.RUNNING.value, ProgramState.DONE.value
        )
        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.DONE

    async def test_storage_without_engine_root_keeps_per_call_mint(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Storage built without ``engine_root`` mints a per-call root
        inside ``_transition_via_dataplane`` and transitions still succeed."""
        storage, _ = dp_storage
        assert storage._engine_root is None  # type: ignore[attr-defined]
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING


class TestReadProgramFreshness:
    """The :class:`Freshness` admission contract on :meth:`read_program`:
    every reader supplies an explicit floor, and a stale read returns
    :class:`StaleReadError` instead of an old blob."""

    async def test_eventual_returns_value_at_any_epoch(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        storage, coord = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        # Trigger one transition to bump the global epoch counter.
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        from gigaevo.dataplane import FreshnessEventual, Ok
        from gigaevo.dataplane.ids import ProgramId

        result = await coord.read_program(
            ProgramId(prog.id), freshness=FreshnessEventual()
        )
        assert isinstance(result, Ok)
        assert result.value is not None
        # ``LocalValue`` wraps the freshness-checked ``Versioned``.
        assert result.value.value.value["state"] == ProgramState.RUNNING.value

    async def test_at_least_below_floor_returns_stale_read_error(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """A stale read at :class:`FreshnessAtLeast` returns
        :class:`StaleReadError` where :class:`FreshnessEventual` would
        have returned the value."""
        storage, coord = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        from gigaevo.dataplane import (
            Err,
            FreshnessAtLeast,
            FreshnessEventual,
            Ok,
            StaleReadError,
        )
        from gigaevo.dataplane.ids import ProgramId

        # Read the persisted epoch to construct a tighter floor.
        eventual = await coord.read_program(
            ProgramId(prog.id), freshness=FreshnessEventual()
        )
        assert isinstance(eventual, Ok)
        assert eventual.value is not None
        observed_epoch = eventual.value.value.epoch

        # Asking for one beyond the observed epoch fires the stale
        # branch. The same call at FreshnessEventual succeeded above —
        # the only thing that changed is the admission contract.
        stale = await coord.read_program(
            ProgramId(prog.id),
            freshness=FreshnessAtLeast(epoch=observed_epoch + 1, generation=0),
        )
        assert isinstance(stale, Err)
        assert isinstance(stale.error, StaleReadError)
        assert stale.error.observed_epoch == observed_epoch
        assert stale.error.min_epoch == observed_epoch + 1

    async def test_strict_freshness_passes_when_blob_matches_counter(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """A single-writer engine's persisted blob is always >= the live
        counter, so :class:`FreshnessStrict` succeeds."""
        storage, coord = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        from gigaevo.dataplane import FreshnessStrict, Ok
        from gigaevo.dataplane.ids import ProgramId

        result = await coord.read_program(
            ProgramId(prog.id), freshness=FreshnessStrict()
        )
        assert isinstance(result, Ok)
        assert result.value is not None

    async def test_legacy_min_epoch_kwarg_still_works(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The bare ``min_epoch=`` kwarg constructs a
        :class:`FreshnessAtLeast` internally."""
        storage, coord = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        from gigaevo.dataplane import Err, Ok, StaleReadError
        from gigaevo.dataplane.ids import ProgramId

        # First read picks up the current epoch.
        result = await coord.read_program(ProgramId(prog.id))
        assert isinstance(result, Ok)
        observed_epoch = result.value.value.epoch  # type: ignore[union-attr]
        stale = await coord.read_program(
            ProgramId(prog.id), min_epoch=observed_epoch + 1
        )
        assert isinstance(stale, Err)
        assert isinstance(stale.error, StaleReadError)

    async def test_both_freshness_and_legacy_kwargs_rejects(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Mixing ``freshness=`` with a non-zero ``min_*`` is ambiguous;
        the resolver returns ``Err``."""
        storage, coord = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)

        from gigaevo.dataplane import (
            DataPlaneError,
            Err,
            FreshnessAtLeast,
        )
        from gigaevo.dataplane.ids import ProgramId

        result = await coord.read_program(
            ProgramId(prog.id),
            freshness=FreshnessAtLeast(epoch=3, generation=0),
            min_epoch=2,
        )
        assert isinstance(result, Err)
        assert isinstance(result.error, DataPlaneError)
        assert "freshness=" in str(result.error)


class TestStateManagerInMemoryHelper:
    async def test_set_in_memory_state_validates_and_assigns(
        self, state_manager: ProgramStateManager, make_program
    ) -> None:
        """Validates the FSM transition and updates ``program.state``
        without writing to storage."""
        prog = make_program(state=ProgramState.QUEUED)
        await state_manager.set_in_memory_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING

    async def test_set_in_memory_state_rejects_illegal_transition(
        self, state_manager: ProgramStateManager, make_program
    ) -> None:
        """An illegal mirror write (e.g. QUEUED → DONE) raises
        :class:`ValueError`; the in-memory state is unchanged."""
        prog = make_program(state=ProgramState.QUEUED)
        with pytest.raises(ValueError):
            await state_manager.set_in_memory_state(prog, ProgramState.DONE)
        # State must remain at the pre-call value on rejection.
        assert prog.state == ProgramState.QUEUED

    async def test_set_in_memory_state_self_loop_is_noop(
        self, state_manager: ProgramStateManager, make_program
    ) -> None:
        """Same-state assignment short-circuits without raising,
        matching :meth:`set_program_state`'s idempotent semantics."""
        prog = make_program(state=ProgramState.RUNNING)
        await state_manager.set_in_memory_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING


class TestBatchTransitionViaDataplane:
    """Batch transitions route per-item through the FSM Lua when the
    dataplane is wired; without one, the raw-pipeline path is used."""

    async def test_batch_transition_state_routes_through_dp(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        storage, _ = dp_storage
        progs = [_program(ProgramState.QUEUED) for _ in range(3)]
        for p in progs:
            await storage.add(p)

        count = await storage.batch_transition_state(
            progs, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        assert count == 3

        running_ids = set(await storage.get_ids_by_status(ProgramState.RUNNING.value))
        queued_ids = set(await storage.get_ids_by_status(ProgramState.QUEUED.value))
        for p in progs:
            assert p.id in running_ids
            assert p.id not in queued_ids
            # In-memory mirror follows the persisted state.
            assert p.state == ProgramState.RUNNING

    async def test_batch_transition_by_ids_routes_through_dp(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        storage, _ = dp_storage
        progs = [_program(ProgramState.QUEUED) for _ in range(3)]
        for p in progs:
            await storage.add(p)
        ids = [p.id for p in progs]

        count = await storage.batch_transition_by_ids(
            ids, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        assert count == 3

        running_ids = set(await storage.get_ids_by_status(ProgramState.RUNNING.value))
        for pid in ids:
            assert pid in running_ids

    async def test_batch_transition_by_ids_filters_mismatched_state(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Only programs whose current state matches ``old_state`` are
        transitioned; non-matching ids are silently skipped."""
        storage, _ = dp_storage
        queued = _program(ProgramState.QUEUED)
        running = _program(ProgramState.RUNNING)
        await storage.add(queued)
        await storage.add(running)

        count = await storage.batch_transition_by_ids(
            [queued.id, running.id],
            ProgramState.QUEUED.value,
            ProgramState.RUNNING.value,
        )
        assert count == 1
        running_ids = set(await storage.get_ids_by_status(ProgramState.RUNNING.value))
        assert queued.id in running_ids
        # The unrelated already-RUNNING program is untouched.
        assert running.id in running_ids

    async def test_batch_transition_illegal_pair_surfaces_as_storage_error(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """An (old, new) pair the FSM rejects raises before any dp call."""
        storage, _ = dp_storage
        prog = _program(ProgramState.DISCARDED)
        await storage.add(prog)
        # DISCARDED is terminal in the FSM; DISCARDED -> RUNNING is not
        # in the transition table and validate_transition rejects it
        # before the dp call.
        with pytest.raises((ValueError, StorageError)):
            await storage.batch_transition_state(
                [prog],
                ProgramState.DISCARDED.value,
                ProgramState.RUNNING.value,
            )

    async def test_batch_transition_legacy_path_when_dp_absent(self) -> None:
        """A storage without a wired dataplane uses the raw-pipeline
        path for both batch methods."""
        server = fakeredis.FakeServer()
        config = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0", key_prefix="legacy-batch"
        )
        storage = RedisProgramStorage(config)
        storage._conn._redis = fakeredis.aioredis.FakeRedis(  # type: ignore[attr-defined]
            server=server, decode_responses=True
        )
        storage._conn._closing = False  # type: ignore[attr-defined]
        try:
            assert storage._dataplane is None
            progs = [_program(ProgramState.QUEUED) for _ in range(2)]
            for p in progs:
                await storage.add(p)
            count = await storage.batch_transition_state(
                progs, ProgramState.QUEUED.value, ProgramState.RUNNING.value
            )
            assert count == 2
            running_ids = set(
                await storage.get_ids_by_status(ProgramState.RUNNING.value)
            )
            for p in progs:
                assert p.id in running_ids
        finally:
            await storage.close()
