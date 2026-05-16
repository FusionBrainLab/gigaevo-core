"""Import-surface smoke test.

Verifies the public ``gigaevo.dataplane`` API resolves every name in
``__all__``, that :class:`DataPlane` can be constructed (no I/O), and
that the public-contract dataclasses round-trip cleanly.
"""

from __future__ import annotations

from typing import cast

import pytest

import gigaevo.dataplane as dp
from gigaevo.dataplane.coordinator import (
    BatchTransitionItem,
    BatchTransitionOutcome,
    EliteInserted,
    EliteRejected,
    EliteSwapOutcome,
    EliteSwapped,
    InstanceLease,
    ProgramPatch,
)
from gigaevo.dataplane.crash import OneShotFlag
from gigaevo.dataplane.errors import NotStartedError
from gigaevo.dataplane.ids import LeaseToken, ProgramId
from gigaevo.dataplane.models import Versioned
from gigaevo.dataplane.permissions import mint_root
from gigaevo.dataplane.transitions import ProgramState


def test_version_present() -> None:
    assert isinstance(dp.__version__, str)
    assert dp.__version__.count(".") >= 1


def test_all_exports_resolve() -> None:
    """Every name in __all__ must be defined."""
    for name in dp.__all__:
        if name == "__version__":
            continue
        assert hasattr(dp, name), f"missing public export: {name}"


def test_dataplane_construction_no_io() -> None:
    """``DataPlane.__init__`` does no network I/O; verify it succeeds."""
    coord = dp.DataPlane("redis://invalid:0/0", key_prefix="smoke")
    assert not coord.started
    assert coord.key_prefix == "smoke"


@pytest.mark.asyncio
async def test_dataplane_startup_against_invalid_url_raises_typed() -> None:
    """Misconfigured URL surfaces as :class:`StartupError`, not a bare Exception."""
    coord = dp.DataPlane(
        "redis://10.255.255.1:1/0",  # RFC 5737 reserved; will fail to connect
        key_prefix="smoke",
        socket_connect_timeout_s=0.1,
    )
    with pytest.raises(dp.StartupError):
        await coord.startup()
    assert not coord.started


def test_method_stubs_raise_notimplemented() -> None:
    """Method bodies are stubs awaiting follow-up work; verify they fail loudly."""
    coord = dp.DataPlane("redis://localhost:6379/0", key_prefix="smoke")
    # Each method raises NotImplementedError when called; we don't await
    # the coroutines (they raise as soon as they're called because the
    # raise lives in the function body, not after an await).
    # Every public DataPlane method is now implemented. Verify the
    # surface is callable as a smoke check — bodies are exercised by
    # their dedicated test_*.py modules against fakeredis.
    public_methods = [
        "transition_program_state",
        "transition_program_state_batch",
        "read_program",
        "acquire_instance_lock",
        "renew_instance_lock",
        "release_instance_lock",
        "try_replace_elite",
        "crdt_inc",
        "crdt_read",
        "lwwr_set",
        "lwwr_get",
    ]
    for name in public_methods:
        method = getattr(coord, name)
        assert callable(method), name


def test_require_started_raises_when_not_started() -> None:
    """``_require_started`` is the gate every state-access body must pass.

    The guard surfaces a typed :class:`NotStartedError` whose ``method``
    field carries the caller-supplied label, so misuse is identified
    without a traceback walk.
    """
    coord = dp.DataPlane("redis://localhost:6379/0", key_prefix="smoke")
    with pytest.raises(NotStartedError) as exc:
        coord._require_started("test_label")
    assert exc.value.method == "test_label"


# ── public contract dataclasses ─────────────────────────────────────────


class TestProgramPatch:
    def test_default_empty(self) -> None:
        p = ProgramPatch()
        assert p.fields == {}

    def test_holds_fields(self) -> None:
        p = ProgramPatch(fields={"score": 0.5, "label": "x"})
        assert p.fields == {"score": 0.5, "label": "x"}

    def test_frozen(self) -> None:
        p = ProgramPatch(fields={"a": 1})
        with pytest.raises(AttributeError):
            p.fields = {"b": 2}  # type: ignore[misc]


class TestBatchTransition:
    def test_item_carries_per_program_token(self) -> None:
        token = mint_root(ProgramId("p-1"))
        item = BatchTransitionItem(
            program_id=ProgramId("p-1"),
            token=token,
            expected_from=ProgramState.QUEUED,
            to=ProgramState.RUNNING,
        )
        assert item.program_id == "p-1"
        assert item.expected_from is ProgramState.QUEUED
        assert item.to is ProgramState.RUNNING
        assert item.patch is None
        # Token is not yet consumed (the batch dispatcher consumes it).
        assert not token.consumed

    def test_outcome_preserves_order(self) -> None:
        snapshots = (
            Versioned(
                value=cast(dict[str, object], {"id": "p-1"}), epoch=1, generation=1
            ),
            Versioned(
                value=cast(dict[str, object], {"id": "p-2"}), epoch=1, generation=2
            ),
        )
        out = BatchTransitionOutcome(items=snapshots)
        assert out.items is snapshots
        assert out.items[0].value == {"id": "p-1"}
        assert out.items[1].generation == 2


class TestInstanceLease:
    def test_round_trip_fields(self) -> None:
        flag = OneShotFlag()
        lease = InstanceLease(
            token=LeaseToken("lease-abc"),
            key="gigaevo:lock:run-1",
            ttl_s=30.0,
            expires_at_monotonic=12345.6,
            flag=flag,
        )
        assert lease.token == "lease-abc"
        assert lease.ttl_s == 30.0
        assert lease.flag is flag

    def test_flag_does_not_participate_in_equality(self) -> None:
        # Two leases with the same token / key / ttl / expiry but
        # different ``OneShotFlag`` instances must compare equal; the
        # flag is an out-of-band signal channel, not part of identity.
        a = InstanceLease(
            token=LeaseToken("t"),
            key="k",
            ttl_s=1.0,
            expires_at_monotonic=2.0,
            flag=OneShotFlag(),
        )
        b = InstanceLease(
            token=LeaseToken("t"),
            key="k",
            ttl_s=1.0,
            expires_at_monotonic=2.0,
            flag=OneShotFlag(),
        )
        assert a == b

    def test_frozen(self) -> None:
        lease = InstanceLease(
            token=LeaseToken("t"),
            key="k",
            ttl_s=1.0,
            expires_at_monotonic=2.0,
            flag=OneShotFlag(),
        )
        with pytest.raises(AttributeError):
            lease.ttl_s = 5.0  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_wrap_lease_returns_crashwatchedhandle(self) -> None:
        """``DataPlane.wrap_lease`` returns a typed handle bound to the lease.

        The wrap is a pure-Python helper — no Redis I/O — so we can
        exercise the contract against an unstarted coordinator. The
        handle's flag identity matches the lease, and a pre-signalled
        flag short-circuits the next ``call`` into a ``CrashEvent``
        carrying the lock key as ``peer`` and the lease itself as
        ``resource``.
        """
        coord = dp.DataPlane("redis://x/0", key_prefix="smoke")
        flag = OneShotFlag()
        lease = InstanceLease(
            token=LeaseToken("tok-1"),
            key="smoke:lock:demo",
            ttl_s=1.0,
            expires_at_monotonic=2.0,
            flag=flag,
        )
        handle = coord.wrap_lease(lease)
        assert handle.inner is lease
        assert handle.flag is flag

        invoked = 0

        async def _op(current: InstanceLease) -> str:
            nonlocal invoked
            invoked += 1
            return current.token

        # Normal path: flag unset, the wrapped op runs and its value is returned.
        value, evt = await handle.call(_op)
        assert evt is None
        assert value == "tok-1"
        assert invoked == 1

        # Signalled path: pattern-match the recovery branch.
        flag.signal()
        match await handle.call(_op):
            case (None, recovery_evt):
                assert recovery_evt.peer == "smoke:lock:demo"
                assert isinstance(recovery_evt.resource, InstanceLease)
                assert recovery_evt.resource.token == "tok-1"
                assert recovery_evt.survivor_tokens == ()
            case (_value, None):
                raise AssertionError(
                    "wrap_lease did not short-circuit after flag.signal()"
                )
        # Wrapped op was NOT invoked again — the short-circuit precedes it.
        assert invoked == 1


class TestEliteSwapOutcome:
    def test_inserted_variant(self) -> None:
        outcome: EliteSwapOutcome = EliteInserted()
        assert outcome.kind == "inserted"

    def test_swapped_carries_displaced_id(self) -> None:
        outcome: EliteSwapOutcome = EliteSwapped(displaced_id=ProgramId("p-old"))
        assert outcome.kind == "swapped"
        assert outcome.displaced_id == "p-old"

    def test_rejected_carries_occupant_id(self) -> None:
        outcome: EliteSwapOutcome = EliteRejected(occupant_id=ProgramId("p-king"))
        assert outcome.kind == "rejected"
        assert outcome.occupant_id == "p-king"

    def test_exhaustive_match(self) -> None:
        """The three variants exhaustively cover :data:`EliteSwapOutcome`."""

        def describe(o: EliteSwapOutcome) -> str:
            match o:
                case EliteInserted():
                    return "inserted"
                case EliteSwapped(displaced_id=did):
                    return f"swapped:{did}"
                case EliteRejected(occupant_id=oid):
                    return f"rejected:{oid}"
            raise AssertionError("unreachable: EliteSwapOutcome is closed")

        assert describe(EliteInserted()) == "inserted"
        assert describe(EliteSwapped(displaced_id=ProgramId("p-1"))) == "swapped:p-1"
        assert describe(EliteRejected(occupant_id=ProgramId("p-2"))) == "rejected:p-2"

    def test_frozen_variants(self) -> None:
        swapped = EliteSwapped(displaced_id=ProgramId("p-old"))
        with pytest.raises(AttributeError):
            swapped.displaced_id = ProgramId("other")  # type: ignore[misc]


# ── input validation at the public boundary ─────────────────────────────


class TestInputValidationHelpers:
    """The validators reject hostile inputs at the coordinator boundary."""

    def test_validate_key_component_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            dp.DataPlane._validate_key_component(
                "", method="m", field_name="program_id"
            )

    def test_validate_key_component_rejects_nul(self) -> None:
        with pytest.raises(ValueError, match="forbidden control character"):
            dp.DataPlane._validate_key_component(
                "ok\x00inject", method="m", field_name="program_id"
            )

    def test_validate_key_component_rejects_newline(self) -> None:
        with pytest.raises(ValueError, match="forbidden control character"):
            dp.DataPlane._validate_key_component(
                "ok\ninject", method="m", field_name="program_id"
            )

    def test_validate_key_component_rejects_colon_atomic(self) -> None:
        with pytest.raises(ValueError, match="collides with the dataplane"):
            dp.DataPlane._validate_key_component(
                "ns:bleed", method="m", field_name="program_id"
            )

    def test_validate_key_component_allows_colon_when_opted_in(self) -> None:
        # Hierarchical keys (CounterKey, LWW register name) legitimately
        # carry colons as sub-namespace separators.
        dp.DataPlane._validate_key_component(
            "bandit:trials:arm-a",
            method="m",
            field_name="key",
            allow_colon=True,
        )

    def test_validate_ttl_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            dp.DataPlane._validate_ttl(float("nan"), method="m")

    def test_validate_ttl_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            dp.DataPlane._validate_ttl(float("inf"), method="m")

    def test_validate_ttl_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            dp.DataPlane._validate_ttl(0.0, method="m")

    def test_validate_ttl_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            dp.DataPlane._validate_ttl(-1.0, method="m")

    def test_validate_ttl_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="real number"):
            dp.DataPlane._validate_ttl(True, method="m")  # type: ignore[arg-type]

    def test_validate_floor_rejects_negative_epoch(self) -> None:
        with pytest.raises(ValueError, match="min_epoch"):
            dp.DataPlane._validate_floor(-1, 0, method="m")

    def test_validate_floor_rejects_negative_generation(self) -> None:
        with pytest.raises(ValueError, match="min_generation"):
            dp.DataPlane._validate_floor(0, -1, method="m")


@pytest.mark.asyncio
class TestPublicMethodValidation:
    """Public methods reject malformed inputs before any Redis I/O."""

    async def _coord(self) -> dp.DataPlane:
        # Construction never touches Redis; we don't need a started
        # coordinator to test input rejection — validators fire first.
        return dp.DataPlane("redis://x/0", key_prefix="smoke")

    async def test_transition_program_state_rejects_colon_pid(self) -> None:
        coord = await self._coord()
        from gigaevo.dataplane.permissions import mint_root

        token = mint_root(dp.ProgramId("a:b"))
        with pytest.raises(ValueError, match="collides"):
            await coord.transition_program_state(
                dp.ProgramId("a:b"),
                token=token,
                expected_from=None,
                to=dp.ProgramState.RUNNING,
            )
        # Token must NOT have been consumed — validation runs first.
        assert not token.consumed

    async def test_read_program_rejects_nul_pid(self) -> None:
        coord = await self._coord()
        with pytest.raises(ValueError, match="forbidden control"):
            await coord.read_program(dp.ProgramId("bad\x00pid"))

    async def test_acquire_instance_lock_rejects_nan_ttl(self) -> None:
        coord = await self._coord()
        with pytest.raises(ValueError, match="finite"):
            await coord.acquire_instance_lock(dp.KeyPrefix("ok"), ttl_s=float("nan"))

    async def test_acquire_instance_lock_rejects_negative_ttl(self) -> None:
        coord = await self._coord()
        with pytest.raises(ValueError, match="positive"):
            await coord.acquire_instance_lock(dp.KeyPrefix("ok"), ttl_s=-3.0)

    async def test_read_program_rejects_negative_floor(self) -> None:
        coord = await self._coord()
        with pytest.raises(ValueError, match="min_epoch"):
            await coord.read_program(dp.ProgramId("p"), min_epoch=-1)
