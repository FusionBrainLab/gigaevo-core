"""Back-compat surface for `EngineConfig` after the two-sema refactor.

Several engine knobs were retired between v1 and v2:

* ``max_mutations_per_generation`` (epoch size — concept gone)
* ``generation_timeout`` (was already a deprecated no-op)
* ``refresh_passes``  (per-epoch archive refresh removed; JIT parent
  refresh handles this now)
* ``refresh_order``   (ordering gone with the per-epoch sweep)

External users still carry yaml configs that set these. To avoid breaking
them on upgrade we accept the old keys, drop them silently from the
model, and emit a one-shot ``DeprecationWarning`` per key so the user
sees they are no-ops.

Unknown keys that are NOT in the deprecated allow-list MUST still raise
(``extra='forbid'`` semantics) — that protects against typos like
``max_inflight: 5``.
"""

from __future__ import annotations

import warnings

from pydantic import ValidationError
import pytest

from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig

DEPRECATED_KEYS_WITH_DUMMY_VALUE: list[tuple[str, object]] = [
    ("max_mutations_per_generation", 50),
    ("generation_timeout", 60.0),
    ("refresh_passes", 2),
    ("refresh_order", "generation_bucketed"),
]


@pytest.mark.parametrize(
    "dead_key,dummy_value",
    DEPRECATED_KEYS_WITH_DUMMY_VALUE,
    ids=[k for k, _ in DEPRECATED_KEYS_WITH_DUMMY_VALUE],
)
def test_deprecated_key_warns_and_drops(dead_key: str, dummy_value: object) -> None:
    """Old yaml with retired knobs constructs cleanly + warns once + drops the field."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = EngineConfig(**{dead_key: dummy_value})

    matched = [w for w in captured if dead_key in str(w.message)]
    assert len(matched) >= 1, (
        f"expected DeprecationWarning naming {dead_key!r}, "
        f"got {[str(w.message) for w in captured]}"
    )
    assert issubclass(matched[0].category, DeprecationWarning), (
        f"wrong category: {matched[0].category}"
    )
    # The dead value MUST NOT survive on the model (the field is gone).
    assert not hasattr(cfg, dead_key), (
        f"{dead_key} leaked onto EngineConfig: {getattr(cfg, dead_key, None)!r}"
    )


def test_all_deprecated_keys_together_still_construct() -> None:
    """A legacy yaml setting every retired knob at once must still build."""
    payload = dict(DEPRECATED_KEYS_WITH_DUMMY_VALUE)
    payload["max_in_flight"] = 7  # real surviving field
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = EngineConfig(**payload)
    assert cfg.max_in_flight == 7
    dead_keys = {k for k, _ in DEPRECATED_KEYS_WITH_DUMMY_VALUE}
    seen = {k for k in dead_keys if any(k in str(w.message) for w in captured)}
    assert seen == dead_keys, f"missing warnings for: {dead_keys - seen}"


def test_unknown_key_still_raises() -> None:
    """A genuine typo (not in the deprecated allow-list) must still hard-fail."""
    with pytest.raises(ValidationError) as excinfo:
        EngineConfig(max_inflight=5)  # typo: max_in_flight has underscore
    assert "max_inflight" in str(excinfo.value).lower()


def test_steady_state_subclass_inherits_compat() -> None:
    """The compat shim must also work through the SteadyStateEngineConfig alias."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = SteadyStateEngineConfig(refresh_passes=2, max_in_flight=3)
    assert cfg.max_in_flight == 3
    assert any("refresh_passes" in str(w.message) for w in captured)


def test_no_warnings_when_only_valid_keys() -> None:
    """Clean yamls (no retired knobs) MUST not emit any DeprecationWarning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cfg = EngineConfig(max_in_flight=4, loop_interval=0.5)
    deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], (
        f"clean config raised spurious DeprecationWarning(s): "
        f"{[str(w.message) for w in deprecations]}"
    )
    assert cfg.max_in_flight == 4
    assert cfg.loop_interval == 0.5


__all__: list[str] = []
