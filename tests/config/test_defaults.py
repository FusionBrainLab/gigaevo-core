"""Regression tests for ``gigaevo.config.defaults``.

The module is the single source of truth for tunable scalars used
across the typed schema layer. The tests below assert the public
surface remains stable: every exported constant follows the
``DEFAULT_*`` naming convention, the namespace is free of accidental
leakage, and the validity-key constant agrees with the runtime
metrics module.
"""

from __future__ import annotations

from gigaevo.config import defaults as D


class TestValidityKey:
    def test_defaults_validity_key_matches_runtime(self) -> None:
        from gigaevo.programs.metrics.context import VALIDITY_KEY

        assert D.DEFAULT_VALIDITY_KEY == VALIDITY_KEY


class TestImmutabilitySurface:
    """The constants are ``Final``-typed but Python does not enforce
    Final at runtime. The convention is enforced by mypy + ruff. The
    test below confirms the constants are accessible as module
    attributes and that the canonical naming pattern (DEFAULT_*) is
    followed without drift."""

    def test_every_constant_uses_default_prefix(self) -> None:
        constants = [
            name
            for name in dir(D)
            if not name.startswith("_") and name.isupper()
        ]
        constants = [c for c in constants if c != "VALIDITY_KEY"]
        non_default = [c for c in constants if not c.startswith("DEFAULT_")]
        assert non_default == [], f"non-DEFAULT_ constants: {non_default}"

    def test_no_unexpected_modules_in_namespace(self) -> None:
        _ALLOWED_NON_CONSTANTS = {"Final", "annotations"}
        public_names = {n for n in dir(D) if not n.startswith("_")}
        non_constants = {
            n for n in public_names if not n.startswith("DEFAULT_")
        } - _ALLOWED_NON_CONSTANTS
        assert non_constants == set(), (
            f"unexpected public symbols leaked into defaults: "
            f"{sorted(non_constants)}"
        )
