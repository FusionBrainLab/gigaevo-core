"""Tests for the adversarial-specific canonical events registry.

Adversarial events live in `gigaevo.adversarial.events` and auto-register via
`BaseEvent.__init_subclass__`. Role invariants (constructor G / improver D
expectations) attach to specific events as Pydantic field validators — they
are intrinsic to the event, not held by an external auditor.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.monitoring.events import CANONICAL_EVENTS


class TestAdversarialEventsRegistered:
    def test_importing_module_registers_all_adversarial_events(self) -> None:
        # Force import — subclass definitions register on import.
        import gigaevo.adversarial.events  # noqa: F401

        for name in ("TRACKER_WRITE", "HOF_FETCH", "HOF_ROTATE", "CELL_PICK"):
            assert name in CANONICAL_EVENTS, (
                f"Adversarial event {name!r} should auto-register on module import"
            )


class TestTrackerWriteEvent:
    def test_fields_accepted(self) -> None:
        from gigaevo.adversarial.events import TrackerWrite

        ev = TrackerWrite(
            pairs_count=10,
            positive_count=3,
            d_wins_added=3,
            g_resisted_added=7,
            d_faced_added=10,
            gen=5,
        )
        assert ev.pairs_count == 10
        assert ev.d_wins_added == 3

    def test_rejects_negative_counts(self) -> None:
        """Role-invariant validator: counts cannot be negative."""
        from gigaevo.adversarial.events import TrackerWrite

        with pytest.raises(ValidationError):
            TrackerWrite(
                pairs_count=-1,
                positive_count=0,
                d_wins_added=0,
                g_resisted_added=0,
                d_faced_added=0,
            )

    def test_rejects_inconsistent_totals(self) -> None:
        """positive_count must not exceed pairs_count."""
        from gigaevo.adversarial.events import TrackerWrite

        with pytest.raises(ValidationError):
            TrackerWrite(
                pairs_count=3,
                positive_count=10,  # > pairs_count
                d_wins_added=10,
                g_resisted_added=0,
                d_faced_added=3,
            )


class TestHofFetchEvent:
    def test_fields_accepted(self) -> None:
        from gigaevo.adversarial.events import HofFetch

        ev = HofFetch(label="K5_1_G", n_elites=8, fitness_key="actual_fitness", gen=3)
        assert ev.label == "K5_1_G"
        assert ev.n_elites == 8

    def test_rejects_negative_n_elites(self) -> None:
        from gigaevo.adversarial.events import HofFetch

        with pytest.raises(ValidationError):
            HofFetch(label="X", n_elites=-1, fitness_key="fitness")


class TestHofRotateEvent:
    def test_fields_accepted(self) -> None:
        from gigaevo.adversarial.events import HofRotate

        ev = HofRotate(label="K5_1_D", old_hof_size=5, new_hof_size=6, gen=2)
        assert ev.old_hof_size == 5
        assert ev.new_hof_size == 6

    def test_rejects_negative_sizes(self) -> None:
        from gigaevo.adversarial.events import HofRotate

        with pytest.raises(ValidationError):
            HofRotate(label="X", old_hof_size=-1, new_hof_size=0)


class TestCellPickEvent:
    def test_fields_accepted(self) -> None:
        from gigaevo.adversarial.events import CellPick

        ev = CellPick(
            label="K5_1_G",
            cell_id="cell_3_2",
            program_id="abc123",
            fitness_key="actual_fitness",
            fitness_value=0.05,
            gen=4,
        )
        assert ev.cell_id == "cell_3_2"
        assert ev.fitness_value == 0.05


class TestFullRegistryAfterImport:
    def test_all_ten_events_present_after_package_import(self) -> None:
        """After importing `gigaevo`, all 10 canonical events must be registered."""
        # Clean import path — triggers both general and adversarial event registration
        import gigaevo  # noqa: F401

        expected = {
            # General (from gigaevo.monitoring.events)
            "GENERATION_BOUNDARY",
            "EXCEPTION",
            "STAGE_EXEC",
            "LLM_CALL",
            "METRIC_EMIT",
            # Adversarial (from gigaevo.adversarial.events)
            "TRACKER_WRITE",
            "HOF_FETCH",
            "HOF_ROTATE",
            "CELL_PICK",
        }
        missing = expected - set(CANONICAL_EVENTS)
        assert not missing, f"Missing canonical events: {missing}"
