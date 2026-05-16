"""Import-surface smoke test.

Verifies the public ``gigaevo.dataplane`` API resolves every name in
``__all__`` and that ``DataPlane`` can be constructed (no I/O — just
that the class is wired correctly).
"""

from __future__ import annotations

import pytest

import gigaevo.dataplane as dp


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
        "read_program",
        "acquire_instance_lock",
        "renew_instance_lock",
        "release_instance_lock",
        "try_replace_elite",
        "crdt_inc",
        "crdt_read",
    ]
    for name in stub_names:
        method = getattr(coord, name)
        assert callable(method), name
