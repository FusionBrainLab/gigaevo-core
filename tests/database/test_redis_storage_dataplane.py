"""Storage tests covering the optional dataplane-routed transition path.

When :class:`RedisProgramStorage` is constructed with a ``dataplane``
parameter, :meth:`atomic_state_transition` and
:meth:`fast_state_transition` route through
:meth:`gigaevo.dataplane.DataPlane.transition_program_state` instead of
the legacy WATCH / MULTI / EXEC pipeline. The coordinator's
``transition_state.lua`` script validates the (from, to) pair against
the FSM hash, merges the patch into the persisted blob, updates the
status set, and emits an audit-stream event in one atomic round-trip.

The coordinator emits the FSM hash with case-tolerant rows: every
entry is written under both its uppercase enum key and its lowercase
form so a program blob persisted by an application-layer caller whose
enum lowercases its ``.value`` resolves the same row as a dataplane
caller that keeps the uppercase form. These tests exercise the routed
path against the case-tolerant table.
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


class TestCaseTolerantFsmVocabulary:
    """The FSM hash resolves either case, removing the bridge helper.

    These tests pin the case-tolerance invariant from the storage's
    point of view: regardless of whether the persisted blob carries a
    lowercase ``state`` (the application-layer enum) or the dataplane's
    uppercase value, a dp-routed transition succeeds. Before the
    case-tolerant emit, an explicit reloader had to overwrite the FSM
    hash with the lowercase mirror so the membership check matched the
    persisted vocabulary; the test exercises the no-bridge path.
    """

    async def test_lowercase_blob_resolves_through_dp(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """A lowercase persisted state value advances via the dp path
        without any external FSM-table rewrite — case tolerance lives
        in the hash itself.
        """
        storage, _ = dp_storage
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)
        # The on-disk ``state`` is the lowercase ``queued``; the dp
        # path forwards it verbatim to the Lua script and the FSM hash
        # accepts the row even though the dp's own ``ProgramState``
        # enum prefers uppercase.
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
        lowercase vocabulary in the on-disk blob.

        The case-tolerant FSM hash means the dp does not need to
        normalise the wire vocabulary; the persisted blob format is
        unchanged so legacy / analytics readers keying on lowercase
        ``state`` strings keep working without a migration.
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
        # The on-disk vocabulary stays lowercase regardless of the
        # case-tolerant FSM hash.
        assert blob["state"] == "running"


class TestSafeDeserializeNoRename:
    """The read-side fixups are gone; ``_safe_deserialize`` is reduced
    to the schema-boundary ``epoch`` strip plus the fakeredis-only
    empty-dict fallback.
    """

    def test_no_epoch_promotion_to_atomic_counter(self) -> None:
        """A blob without ``atomic_counter`` no longer has it synthesised
        from ``epoch``.

        The Lua now stamps ``atomic_counter`` directly so the promote
        branch was dead code. This test pins the absence of the
        rename: a deserialised blob with only ``epoch`` falls back to
        the model's default counter rather than copying ``epoch`` over.
        """
        from gigaevo.utils.json import dumps as _dumps

        # A blob that only carries ``epoch`` — the legacy promote
        # branch would have copied it into ``atomic_counter``. With the
        # fixup removed the read-side must respect the schema-boundary
        # strip and leave ``atomic_counter`` at the model default.
        blob = {
            "id": str(uuid.uuid4()),
            "code": "def f(): pass",
            "state": "queued",
            "epoch": 99,
        }
        raw = _dumps(blob)
        prog = RedisProgramStorage._safe_deserialize(raw, ctx="no-rename")
        assert prog is not None
        # ``atomic_counter`` falls back to the Program model default
        # because the read path no longer renames ``epoch`` for us.
        assert prog.atomic_counter != 99

    def test_empty_dict_round_trip_without_lua_directive(self) -> None:
        """The fakeredis-only fallback keeps empty dict fields as ``{}``.

        Simulates the wire shape fakeredis emits — list-typed empty
        fields — and asserts the read path recovers the dict shape so
        the Pydantic model loads cleanly.
        """
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


class TestEngineRootStorageWiring:
    """Engine-root threading: per-call tokens derive by linear split."""

    @pytest.fixture
    async def dp_storage_with_root(
        self,
    ) -> AsyncIterator[tuple[RedisProgramStorage, dp.DataPlane, dp.EngineRoot]]:
        """Same wiring as :func:`dp_storage` but with an :class:`EngineRoot`.

        The root flows into storage via the constructor keyword; per-call
        FSM tokens are derived by :meth:`EngineRoot.split_program_token`
        inside :meth:`_transition_via_dataplane` rather than minted
        ad-hoc.
        """
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
        its long-lived witness on each split. If the engine consumed the
        root token directly (instead of via the rotating split helper),
        the second transition would raise :class:`TokenAlreadyConsumed`.
        """
        storage, _, engine_root = dp_storage_with_root
        prog = _program(ProgramState.QUEUED)
        await storage.add(prog)

        # First transition consumes the initial program root via split.
        initial_root = engine_root._program_root  # type: ignore[attr-defined]
        await storage.fast_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        assert initial_root.consumed

        # The engine retains a fresh root, unconsumed.
        rotated_root = engine_root._program_root  # type: ignore[attr-defined]
        assert rotated_root is not initial_root
        assert not rotated_root.consumed

        # Second transition succeeds — the rotation preserved the
        # single-live-witness invariant across two calls.
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
        """Backwards-compat invariant: storage built without an
        engine_root mints a per-call root inside ``_transition_via_dataplane``
        and the FSM transition still succeeds. The ``dp_storage`` fixture
        uses the no-engine-root construction; this test pins that
        regression."""
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
    """The :class:`Freshness` admission contract on :meth:`read_program`.

    Every reader passes one explicit value; a stale read at a tighter
    floor returns :class:`StaleReadError` instead of the silently-old
    blob the legacy unguarded read would have returned.
    """

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
        assert result.value.value["state"] == ProgramState.RUNNING.value

    async def test_at_least_below_floor_returns_stale_read_error(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """The mechanical demonstration: a stale read at
        :class:`FreshnessAtLeast` returns :class:`StaleReadError` even
        when the same value would have been returned at
        :class:`FreshnessEventual`."""
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
        observed_epoch = eventual.value.epoch

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
        """A single-writer engine's blob always >= the live counter, so
        :class:`FreshnessStrict` succeeds. The two-round-trip cost is
        the price of the cross-engine race guard."""
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
        """The bare ``min_epoch=`` kwarg routes through the legacy shim
        and constructs a :class:`FreshnessAtLeast` internally so older
        callers do not need a flag-day migration."""
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
        observed_epoch = result.value.epoch  # type: ignore[union-attr]
        stale = await coord.read_program(
            ProgramId(prog.id), min_epoch=observed_epoch + 1
        )
        assert isinstance(stale, Err)
        assert isinstance(stale.error, StaleReadError)

    async def test_both_freshness_and_legacy_kwargs_rejects(
        self,
        dp_storage: tuple[RedisProgramStorage, dp.DataPlane],
    ) -> None:
        """Mixing ``freshness=`` with a non-zero legacy ``min_*`` is a
        type-system ambiguity; the resolver surfaces it as ``Err`` so a
        typo cannot silently pick one over the other."""
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


class TestBatchTransitionViaDataplane:
    """Batch transitions route per-item through the FSM Lua when the
    dataplane is wired, preserving per-item atomicity. The legacy
    raw-pipeline path remains the default when ``dataplane=None``.
    """

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
        transitioned; non-matching ids are silently skipped, mirroring
        the legacy raw-JSON path's filter semantics."""
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
        """A storage without a wired dataplane keeps the legacy raw
        pipeline behaviour for both batch methods."""
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
