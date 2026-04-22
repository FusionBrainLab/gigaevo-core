"""Tests for DGTrackerStage — DAG-driven recording of (D, G, delta) pairs.

Verifies:
  - Real program.id is used (not the old "<program>" placeholder).
  - Role-aware pair construction.
  - NaN filtering, alignment validation, malformed-payload guard.
  - Tracker.record_batch is awaited ONCE with an aligned pair list
    (single pipelined write that also populates d_wins / g_resisted /
    dg_metrics hashes — the metrics-dict schema lives in record_batch).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.adversarial.dg_tracker_stage import DGTrackerStage
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import Box


@pytest.fixture
def tracker():
    mock = AsyncMock()
    mock.record_batch.return_value = 0
    return mock


def _pairs_of(tracker) -> list[tuple[str, str, dict[str, float]]]:
    """Return the pair list from the (only expected) record_batch call."""
    assert tracker.record_batch.await_count == 1, (
        f"expected a single record_batch call, got {tracker.record_batch.await_count}"
    )
    (call,) = tracker.record_batch.await_args_list
    # record_batch(pairs) is positional-only in the stage; accept either shape.
    pairs = call.args[0] if call.args else call.kwargs["pairs"]
    return list(pairs)


def _make_stage(tracker, role: str) -> DGTrackerStage:
    return DGTrackerStage(dg_tracker=tracker, role=role, timeout=5.0)


def _attach(stage: DGTrackerStage, opponent_ids, validation_result) -> None:
    stage.attach_inputs(
        {
            "opponent_ids": Box[object](data=opponent_ids),
            "validation_result": Box[object](data=validation_result),
        }
    )


@pytest.fixture
def program():
    return Program(code="def entrypoint(): pass", metadata={})


# ===================================================================
# Real program.id (not placeholder) — the bug we just fixed
# ===================================================================


class TestUsesRealProgramId:
    @pytest.mark.asyncio
    async def test_constructor_records_with_real_program_id(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d-opp-1", "d-opp-2"]
        artifact = {
            "role": "constructor",
            "n_opponents": 2,
            "per_opp_delta": [0.1, 0.2],
        }
        _attach(stage, opponent_ids, ({"fitness": 0.5}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        # G is the program (real id), D is the opponent.
        assert pairs == [
            ("d-opp-1", program.id, {"delta": 0.1, "is_valid": 1.0}),
            ("d-opp-2", program.id, {"delta": 0.2, "is_valid": 1.0}),
        ]
        for d_id, g_id, _ in pairs:
            assert d_id != "<program>"
            assert g_id != "<program>"

    @pytest.mark.asyncio
    async def test_improver_records_with_real_program_id(self, tracker, program):
        stage = _make_stage(tracker, "improver")
        opponent_ids = ["g-opp-1", "g-opp-2"]
        artifact = {
            "role": "improver",
            "n_opponents": 2,
            "per_opp_delta": [0.05, 0.15],
        }
        _attach(stage, opponent_ids, ({"fitness": 0.4}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        # D is the program (real id), G is the opponent.
        assert pairs == [
            (program.id, "g-opp-1", {"delta": 0.05, "is_valid": 1.0}),
            (program.id, "g-opp-2", {"delta": 0.15, "is_valid": 1.0}),
        ]


# ===================================================================
# NaN filtering
# ===================================================================


class TestNanFiltering:
    @pytest.mark.asyncio
    async def test_nan_deltas_are_skipped(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d1", "d2", "d3"]
        artifact = {"per_opp_delta": [float("nan"), 0.1, float("nan")]}
        _attach(stage, opponent_ids, ({}, artifact))

        await stage.compute(program)

        # Only d2 (non-NaN) survives the filter.
        pairs = _pairs_of(tracker)
        assert pairs == [("d2", program.id, {"delta": 0.1, "is_valid": 1.0})]

    @pytest.mark.asyncio
    async def test_all_nan_does_not_call_record_batch(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d1", "d2"]
        artifact = {"per_opp_delta": [float("nan"), float("nan")]}
        _attach(stage, opponent_ids, ({}, artifact))

        await stage.compute(program)
        tracker.record_batch.assert_not_awaited()


# ===================================================================
# Negative deltas reach record_batch (which filters them)
# ===================================================================


class TestNegativeDeltas:
    @pytest.mark.asyncio
    async def test_negative_deltas_are_forwarded_to_tracker(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d1", "d2"]
        artifact = {"per_opp_delta": [-0.05, 0.1]}
        _attach(stage, opponent_ids, ({}, artifact))

        await stage.compute(program)

        # Both deltas (positive and negative) reach record_batch; the tracker
        # itself decides which per-key family each delta contributes to.
        pairs = _pairs_of(tracker)
        assert pairs == [
            ("d1", program.id, {"delta": -0.05, "is_valid": 1.0}),
            ("d2", program.id, {"delta": 0.1, "is_valid": 1.0}),
        ]


# ===================================================================
# Alignment validation
# ===================================================================


class TestAlignment:
    @pytest.mark.asyncio
    async def test_length_mismatch_skips_recording(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        _attach(
            stage,
            ["d1", "d2", "d3"],
            ({}, {"per_opp_delta": [0.1, 0.2]}),  # length 2 != 3
        )

        await stage.compute(program)
        tracker.record_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_per_opp_delta_treated_as_empty(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        # Empty opponents + empty per_opp_delta → no pairs.
        _attach(stage, [], ({}, {}))

        await stage.compute(program)
        tracker.record_batch.assert_not_awaited()


# ===================================================================
# F22 mismatch triage — split ERROR into benign (DEBUG) vs real (ERROR)
# ===================================================================


@pytest.fixture
def loguru_sink():
    """Capture loguru messages (loguru does not propagate to pytest caplog)."""
    from loguru import logger

    messages: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda m: messages.append((m.record["level"].name, m.record["message"])),
        level="DEBUG",
    )
    yield messages
    logger.remove(sink_id)


class TestF22MismatchTriage:
    @pytest.mark.asyncio
    async def test_candidate_failed_is_debug_not_error(
        self, tracker, program, loguru_sink
    ):
        # G candidate failed validation: opponent_ids was sampled upstream but
        # evaluate.py returned empty per_opp_delta with is_valid=False.
        stage = _make_stage(tracker, "constructor")
        _attach(
            stage,
            ["d1"],
            (
                {"fitness": -1.0},
                {
                    "role": "constructor",
                    "is_valid": False,
                    "n_opponents": 0,
                    "per_opp_delta": [],
                },
            ),
        )

        await stage.compute(program)

        tracker.record_batch.assert_not_awaited()
        levels = [level for level, _ in loguru_sink]
        assert "ERROR" not in levels, (
            f"candidate-failed must not log ERROR: {loguru_sink}"
        )
        assert any("candidate failed validation" in msg for _, msg in loguru_sink)

    @pytest.mark.asyncio
    async def test_gen_zero_seed_race_is_debug_not_error(
        self, tracker, program, loguru_sink
    ):
        # Gen-0: archive empty → opponent_ids empty, but fallback codes produced
        # synthetic per_opp_delta entries. Benign, skip without ERROR noise.
        stage = _make_stage(tracker, "constructor")
        _attach(
            stage,
            [],
            ({}, {"role": "constructor", "per_opp_delta": [0.1, 0.2]}),
        )

        await stage.compute(program)

        tracker.record_batch.assert_not_awaited()
        levels = [level for level, _ in loguru_sink]
        assert "ERROR" not in levels, f"init-race must not log ERROR: {loguru_sink}"
        assert any("gen-0 seed fallback" in msg for _, msg in loguru_sink)

    @pytest.mark.asyncio
    async def test_genuine_mismatch_still_errors(self, tracker, program, loguru_sink):
        # Both sides non-empty but lengths disagree → real mismatch, keep ERROR.
        stage = _make_stage(tracker, "constructor")
        _attach(
            stage,
            ["d1", "d2"],
            (
                {},
                {"role": "constructor", "is_valid": True, "per_opp_delta": [0.1]},
            ),
        )

        await stage.compute(program)

        tracker.record_batch.assert_not_awaited()
        assert any(
            level == "ERROR" and "possible cache leak" in msg
            for level, msg in loguru_sink
        ), f"genuine mismatch must log ERROR: {loguru_sink}"


# ===================================================================
# Malformed validation payload
# ===================================================================


class TestMalformedPayload:
    @pytest.mark.asyncio
    async def test_non_tuple_validation_result_is_rejected(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        # validation_result.data is not a (metrics, artifact) tuple.
        _attach(stage, ["d1"], {"fitness": 0.5})

        await stage.compute(program)
        tracker.record_batch.assert_not_awaited()


# ===================================================================
# Role validation
# ===================================================================


class TestRoleValidation:
    def test_invalid_role_raises(self, tracker):
        with pytest.raises(ValueError, match="role must be"):
            DGTrackerStage(dg_tracker=tracker, role="invalid", timeout=5.0)


# ===================================================================
# F31 — artifact role vs stage role cross-check (wiring-bug detector)
# ===================================================================


class TestArtifactRoleCrossCheck:
    @pytest.mark.asyncio
    async def test_role_mismatch_skips_recording(self, tracker, program):
        # Stage configured as constructor but artifact tagged as improver →
        # wiring bug, must not record.
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d1", "d2"]
        artifact = {
            "role": "improver",  # mismatched
            "n_opponents": 2,
            "per_opp_delta": [0.1, 0.2],
        }
        _attach(stage, opponent_ids, ({}, artifact))

        await stage.compute(program)
        tracker.record_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_role_match_records_normally(self, tracker, program):
        stage = _make_stage(tracker, "improver")
        opponent_ids = ["g1", "g2"]
        artifact = {
            "role": "improver",
            "n_opponents": 2,
            "per_opp_delta": [0.05, 0.10],
        }
        _attach(stage, opponent_ids, ({}, artifact))

        await stage.compute(program)
        assert _pairs_of(tracker) == [
            (program.id, "g1", {"delta": 0.05, "is_valid": 1.0}),
            (program.id, "g2", {"delta": 0.10, "is_valid": 1.0}),
        ]

    @pytest.mark.asyncio
    async def test_missing_role_field_does_not_block(self, tracker, program):
        # Older artifacts without 'role' should still record (back-compat).
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d1"]
        artifact = {"per_opp_delta": [0.1]}  # no role
        _attach(stage, opponent_ids, ({}, artifact))

        await stage.compute(program)
        tracker.record_batch.assert_awaited_once()


# ===================================================================
# Task 4 (revised): DGTrackerStage emits ONE pipelined record_batch call.
# The metrics-dict schema is authored inside record_batch — the stage
# only hands it a list of (d_id, g_id, delta) tuples.
# ===================================================================


class TestDGTrackerStageBatchesPairs:
    @pytest.mark.asyncio
    async def test_improver_batches_all_pairs_in_one_call(self, tracker, program):
        stage = _make_stage(tracker, "improver")
        opponent_ids = ["g1", "g2"]
        artifact = {"role": "improver", "per_opp_delta": [0.1, 0.2]}
        _attach(stage, opponent_ids, ({"fitness": 0.5}, artifact))
        await stage.compute(program)
        # Single pipelined write — N-RTT-per-opponent is a regression.
        assert tracker.record_batch.await_count == 1
        assert tracker.record_metrics.await_count == 0
        pairs = _pairs_of(tracker)
        assert pairs == [
            (program.id, "g1", {"delta": 0.1, "is_valid": 1.0}),
            (program.id, "g2", {"delta": 0.2, "is_valid": 1.0}),
        ]

    @pytest.mark.asyncio
    async def test_nan_delta_is_dropped_from_batch(self, tracker, program):
        stage = _make_stage(tracker, "improver")
        _attach(
            stage,
            ["g1"],
            ({"fitness": 0.0}, {"role": "improver", "per_opp_delta": [float("nan")]}),
        )
        await stage.compute(program)
        # All-NaN → no pairs → no Redis write at all.
        tracker.record_batch.assert_not_awaited()


# ===================================================================
# Regression guard: DGTrackerStage must populate dg_d_wins / dg_g_resisted
# so TrackerCoverageStage (BD y-axis source) sees non-empty SETs.
# Prior to this fix (commit dfefc096) the stage called record_metrics per
# opponent, which only wrote dg_metrics and silently dropped the inverted
# indices — breaking MAP-Elites BD axes in every adversarial run.
# ===================================================================


# ===================================================================
# Task 4: DGTrackerStage forwards `per_opp_metrics` dicts verbatim.
# When the artifact carries `per_opp_metrics`, each pair's record is the
# full per-opp dict. When only `per_opp_delta` is present (legacy),
# synthesise `{"delta": d, "is_valid": 1.0}` for non-NaN entries.
# Aggregator contract (design §5.3): tracker records everything; the
# aggregator's MetricsContext.is_valid gates per-record downstream —
# so `is_valid=0.0` dicts ARE forwarded, only NaN/None deltas are skipped.
# ===================================================================


class TestForwardsPerOppMetrics:
    @pytest.mark.asyncio
    async def test_improver_forwards_full_pop_b_dict(self, tracker, program):
        stage = _make_stage(tracker, "improver")
        opponent_ids = ["g1", "g2"]
        per_opp_metrics = [
            {
                "pre_q": 0.3,
                "post_q": 0.35,
                "delta": 0.05,
                "score": 0.5,
                "is_valid": 1.0,
            },
            {
                "pre_q": 0.25,
                "post_q": 0.25,
                "delta": 0.0,
                "score": 0.0,
                "is_valid": 1.0,
            },
        ]
        artifact = {
            "role": "improver",
            "per_opp_delta": [0.05, 0.0],
            "per_opp_metrics": per_opp_metrics,
        }
        _attach(stage, opponent_ids, ({"fitness": 0.25}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        assert pairs == [
            (program.id, "g1", per_opp_metrics[0]),
            (program.id, "g2", per_opp_metrics[1]),
        ]

    @pytest.mark.asyncio
    async def test_constructor_forwards_full_pop_a_dict(self, tracker, program):
        stage = _make_stage(tracker, "constructor")
        opponent_ids = ["d1", "d2"]
        per_opp_metrics = [
            {
                "post_q": 0.40,
                "delta": 0.00,
                "resistance_score": 1.0,
                "is_valid": 1.0,
            },
            {
                "post_q": 0.45,
                "delta": 0.05,
                "resistance_score": 0.0,
                "is_valid": 1.0,
            },
        ]
        artifact = {
            "role": "constructor",
            "per_opp_delta": [0.0, 0.05],
            "per_opp_metrics": per_opp_metrics,
        }
        _attach(stage, opponent_ids, ({"fitness": 0.5}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        # constructor role: pair = (opponent_id, program_id, record)
        assert pairs == [
            ("d1", program.id, per_opp_metrics[0]),
            ("d2", program.id, per_opp_metrics[1]),
        ]

    @pytest.mark.asyncio
    async def test_missing_per_opp_metrics_falls_back_to_scalar_synthesis(
        self, tracker, program
    ):
        # Legacy artifact: only per_opp_delta present. Fallback must
        # synthesise {"delta": d, "is_valid": 1.0} for each non-NaN entry.
        stage = _make_stage(tracker, "improver")
        artifact = {"role": "improver", "per_opp_delta": [0.1, 0.2]}
        _attach(stage, ["g1", "g2"], ({}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        assert pairs == [
            (program.id, "g1", {"delta": 0.1, "is_valid": 1.0}),
            (program.id, "g2", {"delta": 0.2, "is_valid": 1.0}),
        ]

    @pytest.mark.asyncio
    async def test_is_valid_false_entry_is_forwarded_not_skipped(
        self, tracker, program
    ):
        # Design §5.3: ConfigurableAggregator filters via MetricsContext.is_valid.
        # The tracker records ALL valid-schema entries; downstream aggregator
        # applies the validity gate. So `is_valid=0.0` dicts are forwarded,
        # not skipped. NaN deltas (no measurement at all) are still dropped.
        stage = _make_stage(tracker, "improver")
        per_opp_metrics = [
            {
                "pre_q": 0.3,
                "post_q": 0.4,
                "delta": 0.1,
                "score": 1.0,
                "is_valid": 1.0,
            },
            {
                "pre_q": 0.0,
                "post_q": 0.0,
                "delta": 0.0,
                "score": 0.0,
                "is_valid": 0.0,
            },
        ]
        artifact = {
            "role": "improver",
            "per_opp_delta": [0.1, 0.0],
            "per_opp_metrics": per_opp_metrics,
        }
        _attach(stage, ["g1", "g2"], ({}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        assert pairs == [
            (program.id, "g1", per_opp_metrics[0]),
            (program.id, "g2", per_opp_metrics[1]),
        ]

    @pytest.mark.asyncio
    async def test_nan_delta_still_skipped_even_with_per_opp_metrics(
        self, tracker, program
    ):
        # NaN/None delta still represents "no measurement" — skip preserves
        # the Task-3 semantic even under the dict-forwarding contract.
        stage = _make_stage(tracker, "improver")
        per_opp_metrics = [
            {
                "pre_q": 0.3,
                "post_q": float("nan"),
                "delta": float("nan"),
                "score": 0.0,
                "is_valid": 0.0,
            },
            {
                "pre_q": 0.3,
                "post_q": 0.35,
                "delta": 0.05,
                "score": 0.5,
                "is_valid": 1.0,
            },
        ]
        artifact = {
            "role": "improver",
            "per_opp_delta": [float("nan"), 0.05],
            "per_opp_metrics": per_opp_metrics,
        }
        _attach(stage, ["g1", "g2"], ({}, artifact))

        await stage.compute(program)

        pairs = _pairs_of(tracker)
        assert pairs == [(program.id, "g2", per_opp_metrics[1])]

    @pytest.mark.asyncio
    async def test_length_mismatch_per_opp_metrics_errors_and_skips(
        self, tracker, program, loguru_sink
    ):
        # If per_opp_metrics is present but its length disagrees with
        # opponent_ids (and per_opp_delta also disagrees), treat as a
        # real mismatch — SKIP batch + ERROR log (same triage as scalars).
        stage = _make_stage(tracker, "improver")
        artifact = {
            "role": "improver",
            "is_valid": True,
            "per_opp_delta": [0.1],  # length 1
            "per_opp_metrics": [
                {
                    "pre_q": 0.3,
                    "post_q": 0.35,
                    "delta": 0.05,
                    "score": 0.5,
                    "is_valid": 1.0,
                }
            ],
        }
        _attach(stage, ["g1", "g2"], ({}, artifact))  # opponents length 2

        await stage.compute(program)

        tracker.record_batch.assert_not_awaited()
        assert any(
            level == "ERROR" and "possible cache leak" in msg
            for level, msg in loguru_sink
        ), f"length mismatch must log ERROR: {loguru_sink}"


class TestInvertedIndexPopulation:
    @pytest.mark.asyncio
    async def test_improver_populates_d_wins_and_g_resisted(self, program):
        import fakeredis.aioredis

        from gigaevo.adversarial.dg_tracker import DGImprovementTracker

        real_tracker = DGImprovementTracker(
            host="localhost", port=6379, db=0, prefix="test"
        )
        real_tracker._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        try:
            stage = _make_stage(real_tracker, "improver")
            artifact = {"role": "improver", "per_opp_delta": [0.1, -0.05]}
            _attach(stage, ["g1", "g2"], ({"fitness": 0.4}, artifact))

            await stage.compute(program)

            d_wins = await real_tracker._redis.smembers(
                real_tracker._d_wins_key(program.id)
            )
            assert d_wins == {"g1"}, (
                "regression: DGTrackerStage must populate dg_d_wins for "
                "TrackerCoverageStage (D's BD y-axis source)"
            )
            g2_resisted = await real_tracker._redis.smembers(
                real_tracker._g_resisted_key("g2")
            )
            assert g2_resisted == {program.id}, (
                "regression: DGTrackerStage must populate dg_g_resisted for "
                "G's fallback BD y-axis source"
            )
        finally:
            await real_tracker.close()
