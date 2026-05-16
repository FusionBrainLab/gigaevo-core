"""Integration tests for the program-FSM transition API.

Covers ``transition_state.lua`` plus :meth:`DataPlane.transition_program_state`
and :meth:`DataPlane.read_program`. The Lua script does the FSM check,
patch merging, status-set update, and event-stream append in one atomic
round-trip; tests verify each invariant against fakeredis.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="test")
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry
    from gigaevo.dataplane.transitions import (
        PROGRAM_STATE_TRANSITIONS,
        load_fsm_table,
    )

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    await load_fsm_table(
        fake, key_prefix="test", name="program_state", table=PROGRAM_STATE_TRANSITIONS
    )
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    try:
        yield coord
    finally:
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await fake.aclose()  # type: ignore[attr-defined]


async def _put_program(
    coord: dp.DataPlane, pid: str, state: dp.ProgramState, **extra: object
) -> None:
    """Pre-populate a program blob via the underlying pool.

    The transition Lua expects an existing blob; there is no
    add_program method on DataPlane yet (that lands with the program
    storage rewrite), so tests inject the blob directly here.
    """
    blob: dict[str, object] = {"id": pid, "state": state.value, "epoch": 0, **extra}
    await coord._connection.pool.set(  # type: ignore[misc]
        f"test:program:{pid}", json.dumps(blob)
    )


def _token(pid: str) -> dp.Token[dp.ProgramId]:
    return dp.mint_root(dp.ProgramId(pid))


# ── happy-path transitions ───────────────────────────────────────────


class TestHappyPath:
    async def test_queued_to_running(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-1", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("p-1"),
            token=_token("p-1"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "RUNNING"
        assert result.value.epoch >= 1

    async def test_running_to_done(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-2", dp.ProgramState.RUNNING)
        result = await coord.transition_program_state(
            dp.ProgramId("p-2"),
            token=_token("p-2"),
            expected_from=dp.ProgramState.RUNNING,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "DONE"

    async def test_expected_from_none_matches_any(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-3", dp.ProgramState.RUNNING)
        result = await coord.transition_program_state(
            dp.ProgramId("p-3"),
            token=_token("p-3"),
            expected_from=None,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "DONE"

    async def test_patch_merged_into_blob(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-4", dp.ProgramState.QUEUED, score=0.0)
        result = await coord.transition_program_state(
            dp.ProgramId("p-4"),
            token=_token("p-4"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"score": 1.5, "worker": "w-1"}),
        )
        assert isinstance(result, dp.Ok)
        blob = result.value.value
        assert blob["score"] == 1.5
        assert blob["worker"] == "w-1"
        assert blob["state"] == "RUNNING"


# ── error paths ──────────────────────────────────────────────────────


class TestErrorPaths:
    async def test_missing_program_returns_stale(self, coord: dp.DataPlane) -> None:
        result = await coord.transition_program_state(
            dp.ProgramId("never-existed"),
            token=_token("never-existed"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "stale"
        assert "program not found" in result.error.detail

    async def test_expected_from_mismatch_returns_stale(
        self, coord: dp.DataPlane
    ) -> None:
        await _put_program(coord, "p-5", dp.ProgramState.RUNNING)
        result = await coord.transition_program_state(
            dp.ProgramId("p-5"),
            token=_token("p-5"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "stale"
        assert "expected_from=QUEUED" in result.error.detail

    async def test_illegal_transition_skips_states(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-6", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("p-6"),
            token=_token("p-6"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.DONE,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "illegal"
        assert "QUEUED -> DONE" in result.error.detail

    async def test_discarded_is_terminal(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-7", dp.ProgramState.DISCARDED)
        for target in (
            dp.ProgramState.QUEUED,
            dp.ProgramState.RUNNING,
            dp.ProgramState.DONE,
        ):
            result = await coord.transition_program_state(
                dp.ProgramId("p-7"),
                token=_token("p-7"),
                expected_from=None,
                to=target,
            )
            assert isinstance(result, dp.Err)
            assert result.error.kind == "illegal"

    async def test_token_pid_mismatch_returns_unknown(
        self, coord: dp.DataPlane
    ) -> None:
        await _put_program(coord, "p-8", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("p-8"),
            token=_token("a-different-pid"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "unknown"


# ── idempotency ──────────────────────────────────────────────────────


class TestIdempotency:
    async def test_replay_returns_duplicate_same_blob(
        self, coord: dp.DataPlane
    ) -> None:
        await _put_program(coord, "p-9", dp.ProgramState.QUEUED)
        first = await coord.transition_program_state(
            dp.ProgramId("p-9"),
            token=_token("p-9"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"trial": 1}),
        )
        assert isinstance(first, dp.Ok)
        second = await coord.transition_program_state(
            dp.ProgramId("p-9"),
            token=_token("p-9"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"trial": 1}),
        )
        assert isinstance(second, dp.Ok)
        assert first.value.value == second.value.value
        assert first.value.epoch == second.value.epoch


# ── side effects: status sets + events stream ────────────────────────


class TestSideEffects:
    async def test_status_sets_updated(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-11", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-11"),
            token=_token("p-11"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        queued = await pool.smembers("test:status:QUEUED")  # type: ignore[misc]
        running = await pool.smembers("test:status:RUNNING")  # type: ignore[misc]
        assert "p-11" not in queued
        assert "p-11" in running

    async def test_status_event_emitted(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-12", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-12"),
            token=_token("p-12"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        events = await pool.xrange("test:status_events")  # type: ignore[misc]
        assert len(events) >= 1
        _, last_fields = events[-1]
        # dp-native fields.
        assert last_fields["pid"] == "p-12"
        assert last_fields["from"] == "QUEUED"
        assert last_fields["to"] == "RUNNING"
        # Legacy fields, emitted alongside so pre-coordinator readers
        # observe the same wire shape they did on the WATCH/MULTI/EXEC
        # path.
        assert last_fields["id"] == "p-12"
        assert last_fields["status"] == "RUNNING"
        assert last_fields["event"] == "transition"

    async def test_lua_invokes_encode_empty_table_directive(
        self, coord: dp.DataPlane
    ) -> None:
        """Pin that the Lua source contains the cjson-empty-object guard.

        fakeredis's embedded Lua VM does not expose
        ``cjson.encode_empty_table_as_object``, so a runtime round-trip
        test would only check the directive's no-op path. The source-text
        pin is the strongest verification available without a real Redis
        instance: it asserts the directive is present at the top of the
        script and is invoked conditionally so deployments on Redis builds
        without the function continue to load the script.
        """
        from gigaevo.dataplane.scripts import load_lua_source

        source = load_lua_source(
            __import__(
                "gigaevo.dataplane.coordinator", fromlist=["_SCRIPT_TRANSITION_STATE"]
            )._SCRIPT_TRANSITION_STATE
        )
        assert "cjson.encode_empty_table_as_object" in source, (
            "transition_state.lua must call cjson.encode_empty_table_as_object "
            "so empty Lua tables encode as JSON objects rather than arrays; "
            "without this directive, persisted dict fields round-trip to []"
        )
        # The call must be guarded so older Redis builds / fakeredis pass
        # the script-load step.
        assert "type(cjson.encode_empty_table_as_object) == 'function'" in source, (
            "the directive must be guarded by a type check so script load "
            "succeeds against Redis builds that omit the function"
        )

    async def test_atomic_counter_mirrors_epoch_in_blob(
        self, coord: dp.DataPlane
    ) -> None:
        """The Lua stamps both ``epoch`` and ``atomic_counter`` on the blob.

        Application-layer callers that key on ``atomic_counter`` (the
        gigaevo Program merge tiebreaker) read the persisted blob and
        expect a monotonic integer there; the dp-native ``epoch`` is the
        same value under a different name. Both are present so the read
        path is a pass-through for either field name.
        """
        await _put_program(coord, "ac-1", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("ac-1"),
            token=_token("ac-1"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert isinstance(result, dp.Ok)
        blob = result.value.value
        assert isinstance(blob, dict)
        assert "epoch" in blob and "atomic_counter" in blob
        assert blob["epoch"] == blob["atomic_counter"]
        assert int(blob["epoch"]) == result.value.epoch


# ── read_program ─────────────────────────────────────────────────────


class TestReadProgram:
    async def test_missing_returns_ok_none(self, coord: dp.DataPlane) -> None:
        result = await coord.read_program(dp.ProgramId("does-not-exist"))
        assert isinstance(result, dp.Ok)
        assert result.value is None

    async def test_existing_returns_versioned(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-13", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-13"),
            token=_token("p-13"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        result = await coord.read_program(dp.ProgramId("p-13"))
        assert isinstance(result, dp.Ok)
        assert result.value is not None
        assert result.value.value["state"] == "RUNNING"
        assert result.value.epoch >= 1

    async def test_stale_floor_raises(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-14", dp.ProgramState.QUEUED)
        await coord.transition_program_state(
            dp.ProgramId("p-14"),
            token=_token("p-14"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        current = await coord.read_program(dp.ProgramId("p-14"))
        assert isinstance(current, dp.Ok) and current.value is not None
        future = await coord.read_program(
            dp.ProgramId("p-14"), min_epoch=current.value.epoch + 10
        )
        assert isinstance(future, dp.Err)
        assert isinstance(future.error, dp.StaleReadError)


# ── token discipline ─────────────────────────────────────────────────


class TestTokenDiscipline:
    async def test_token_consumed_on_call(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-15", dp.ProgramState.QUEUED)
        token = _token("p-15")
        await coord.transition_program_state(
            dp.ProgramId("p-15"),
            token=token,
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        assert token.consumed

    async def test_reused_token_raises(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "p-16", dp.ProgramState.QUEUED)
        token = _token("p-16")
        await coord.transition_program_state(
            dp.ProgramId("p-16"),
            token=token,
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
        )
        with pytest.raises(dp.TokenAlreadyConsumed):
            await coord.transition_program_state(
                dp.ProgramId("p-16"),
                token=token,
                expected_from=dp.ProgramState.RUNNING,
                to=dp.ProgramState.DONE,
            )


# ── reserved patch fields ────────────────────────────────────────────


class TestReservedPatchFields:
    """A caller's patch must not be able to corrupt the blob's identity.

    ``state``, ``epoch``, and ``id`` are server-owned; a malicious or
    buggy caller emitting them in the patch must see them silently
    dropped rather than overwritten. The script is the single source of
    truth for ``state`` (the target) and ``epoch`` (the next epoch);
    ``id`` is the program identifier that pins the blob.
    """

    async def test_patch_cannot_overwrite_id(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "rsvd-1", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("rsvd-1"),
            token=_token("rsvd-1"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"id": "spoofed"}),
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["id"] == "rsvd-1"

    async def test_patch_cannot_overwrite_state(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "rsvd-2", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("rsvd-2"),
            token=_token("rsvd-2"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"state": "DONE"}),
        )
        assert isinstance(result, dp.Ok)
        assert result.value.value["state"] == "RUNNING"

    async def test_patch_cannot_overwrite_epoch(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "rsvd-3", dp.ProgramState.QUEUED)
        result = await coord.transition_program_state(
            dp.ProgramId("rsvd-3"),
            token=_token("rsvd-3"),
            expected_from=dp.ProgramState.QUEUED,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(fields={"epoch": 999_999}),
        )
        assert isinstance(result, dp.Ok)
        assert int(result.value.value["epoch"]) < 999_999


# ── typed `invalid` error surfacing ──────────────────────────────────


class TestInvalidStatusTypedError:
    """Lua ``'invalid'`` returns surface as ``TransitionError(kind="invalid")``.

    Distinct from ``illegal`` (FSM table rejects the pair) and from
    ``stale`` (expected_from mismatch); ``invalid`` covers caller-side
    input that the script could not act on at all (malformed patch
    JSON, corrupt persisted blob).
    """

    async def test_corrupt_blob_surfaces_as_invalid(self, coord: dp.DataPlane) -> None:
        pool = coord._connection.pool  # type: ignore[attr-defined]
        # Inject a malformed JSON blob — the Lua decode pcall returns
        # ``invalid`` with a descriptive payload.
        await pool.set("test:program:corrupt", "not-a-json-object")  # type: ignore[misc]
        result = await coord.transition_program_state(
            dp.ProgramId("corrupt"),
            token=_token("corrupt"),
            expected_from=None,
            to=dp.ProgramState.RUNNING,
            patch=dp.ProgramPatch(),
        )
        assert isinstance(result, dp.Err)
        assert result.error.kind == "invalid"


# ── batch transitions ────────────────────────────────────────────────


class TestBatch:
    async def test_empty_batch_returns_ok_empty(self, coord: dp.DataPlane) -> None:
        result = await coord.transition_program_state_batch(items=())
        assert isinstance(result, dp.Ok)
        assert result.value.items == ()

    async def test_batch_applies_all_in_order(self, coord: dp.DataPlane) -> None:
        for i in range(1, 4):
            await _put_program(coord, f"b-{i}", dp.ProgramState.QUEUED)
        items = tuple(
            dp.BatchTransitionItem(
                program_id=dp.ProgramId(f"b-{i}"),
                token=_token(f"b-{i}"),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            )
            for i in range(1, 4)
        )
        result = await coord.transition_program_state_batch(items=items)
        assert isinstance(result, dp.Ok)
        outcomes = result.value.items
        assert len(outcomes) == 3
        for v, expected_pid in zip(outcomes, ("b-1", "b-2", "b-3")):
            assert v.value["id"] == expected_pid
            assert v.value["state"] == "RUNNING"

    async def test_batch_per_item_token_consumed(self, coord: dp.DataPlane) -> None:
        await _put_program(coord, "b-x", dp.ProgramState.QUEUED)
        await _put_program(coord, "b-y", dp.ProgramState.QUEUED)
        tx = _token("b-x")
        ty = _token("b-y")
        items = (
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-x"),
                token=tx,
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-y"),
                token=ty,
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
        )
        result = await coord.transition_program_state_batch(items=items)
        assert isinstance(result, dp.Ok)
        assert tx.consumed and ty.consumed

    async def test_batch_partial_failure_returns_err_after_partial_commit(
        self, coord: dp.DataPlane
    ) -> None:
        # b-good is QUEUED and will succeed; b-bad doesn't exist so the
        # second item fails. The first item is already committed when
        # the second fails — the batch is not atomic across items.
        await _put_program(coord, "b-good", dp.ProgramState.QUEUED)
        items = (
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-good"),
                token=_token("b-good"),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
            dp.BatchTransitionItem(
                program_id=dp.ProgramId("b-bad"),  # never written
                token=_token("b-bad"),
                expected_from=dp.ProgramState.QUEUED,
                to=dp.ProgramState.RUNNING,
            ),
        )
        result = await coord.transition_program_state_batch(items=items)
        assert isinstance(result, dp.Err)
        assert result.error.kind == "stale"
        # The first item's commit survived.
        read = await coord.read_program(dp.ProgramId("b-good"))
        assert isinstance(read, dp.Ok) and read.value is not None
        assert read.value.value["state"] == "RUNNING"
