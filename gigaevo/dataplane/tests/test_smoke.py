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
    stub_names = [
        "transition_program_state",
        "transition_program_state_batch",
        "read_program",
        "try_replace_elite",
    ]
    for name in stub_names:
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


def test_require_lua_alias_still_works() -> None:
    """The legacy ``_require_lua`` name still raises ``NotStartedError``.

    Method bodies that have not yet landed call ``_require_lua``; keep
    the alias green until those bodies migrate to
    ``_require_started("<method>")``.
    """
    coord = dp.DataPlane("redis://localhost:6379/0", key_prefix="smoke")
    with pytest.raises(NotStartedError):
        coord._require_lua()


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
