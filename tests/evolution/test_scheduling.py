"""Tests for gigaevo.evolution.scheduling module."""

from __future__ import annotations

import pathlib

import pytest

from gigaevo.evolution.scheduling.feature_extractor import (
    ChainFeatureExtractor,
    CodeFeatureExtractor,
    CompositeFeatureExtractor,
)
from gigaevo.evolution.scheduling.predictor import (
    ConstantPredictor,
    EvalTimePredictor,
    RidgePredictor,
    SimpleHeuristicPredictor,
)
from gigaevo.evolution.scheduling.prioritizer import (
    FIFOPrioritizer,
    LPTPrioritizer,
    SJFPrioritizer,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(code: str = "def solve(): return 42") -> Program:
    return Program(code=code, state=ProgramState.QUEUED)


def _short_prog() -> Program:
    return _prog("def f(): pass")


def _long_prog() -> Program:
    return _prog("def f():\n" + "    x = 1\n" * 200)


# ---------------------------------------------------------------------------
# FeatureExtractor tests
# ---------------------------------------------------------------------------


class TestCodeFeatureExtractor:
    def test_returns_dict(self) -> None:
        ext = CodeFeatureExtractor()
        features = ext.extract(_prog())
        assert isinstance(features, dict)
        assert "code_length" in features
        assert "num_lines" in features
        assert "num_function_defs" in features
        assert "num_loop_constructs" in features

    def test_code_length_proportional(self) -> None:
        ext = CodeFeatureExtractor()
        short = ext.extract(_short_prog())
        long = ext.extract(_long_prog())
        assert long["code_length"] > short["code_length"]
        assert long["num_lines"] > short["num_lines"]

    def test_counts_loops(self) -> None:
        ext = CodeFeatureExtractor()
        code = "for i in range(10):\n    while True:\n        break"
        features = ext.extract(_prog(code))
        assert features["num_loop_constructs"] == 2.0

    def test_counts_function_defs(self) -> None:
        ext = CodeFeatureExtractor()
        code = "def a(): pass\ndef b(): pass\ndef c(): pass"
        features = ext.extract(_prog(code))
        assert features["num_function_defs"] == 3.0


class TestCompositeFeatureExtractor:
    def test_merges_features(self) -> None:
        class CustomExt:
            def extract(self, program: Program) -> dict[str, float]:
                return {"custom_feature": 42.0}

        comp = CompositeFeatureExtractor([CodeFeatureExtractor(), CustomExt()])
        features = comp.extract(_prog())
        assert "code_length" in features
        assert "custom_feature" in features
        assert features["custom_feature"] == 42.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            CompositeFeatureExtractor([])

    def test_last_writer_wins(self) -> None:
        class Ext1:
            def extract(self, program: Program) -> dict[str, float]:
                return {"shared": 1.0}

        class Ext2:
            def extract(self, program: Program) -> dict[str, float]:
                return {"shared": 2.0}

        comp = CompositeFeatureExtractor([Ext1(), Ext2()])
        assert comp.extract(_prog())["shared"] == 2.0


# ---------------------------------------------------------------------------
# ChainFeatureExtractor tests (real chain programs)
# ---------------------------------------------------------------------------

# Real HoVer baseline (7 steps: 3 tool + 4 LLM, empty system_prompt)
_HOVER_BASELINE = pathlib.Path(
    "problems/chains/hover/static_soft/initial_programs/baseline.py"
).read_text()

# Real HotpotQA baseline (6 steps: 2 tool + 4 LLM, empty system_prompt)
_HOTPOTQA_BASELINE = pathlib.Path(
    "problems/chains/hotpotqa/static/initial_programs/baseline.py"
).read_text()


class TestChainFeatureExtractor:
    def test_hover_baseline_step_counts(self) -> None:
        """HoVer baseline has 3 tool steps and 4 LLM steps."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog(_HOVER_BASELINE))
        assert features["n_tool_steps"] == 3.0
        assert features["n_llm_steps"] == 4.0
        assert features["n_total_steps"] == 7.0

    def test_hotpotqa_baseline_step_counts(self) -> None:
        """HotpotQA baseline has 2 tool steps and 4 LLM steps."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog(_HOTPOTQA_BASELINE))
        assert features["n_tool_steps"] == 2.0
        assert features["n_llm_steps"] == 4.0
        assert features["n_total_steps"] == 6.0

    def test_hover_baseline_no_system_prompt(self) -> None:
        """HoVer baseline has empty system_prompt."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog(_HOVER_BASELINE))
        assert features["has_system_prompt"] == 0.0

    def test_hover_baseline_deep_retrieval(self) -> None:
        """HoVer baseline uses retrieve_deep for the third hop."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog(_HOVER_BASELINE))
        assert features["n_deep_retrieval"] == 1.0

    def test_hotpotqa_no_deep_retrieval(self) -> None:
        """HotpotQA baseline has no retrieve_deep."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog(_HOTPOTQA_BASELINE))
        assert features["n_deep_retrieval"] == 0.0

    def test_hover_baseline_no_examples(self) -> None:
        """HoVer baseline has no few-shot examples."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog(_HOVER_BASELINE))
        assert features["n_examples"] == 0.0

    def test_evolved_program_has_more_string_content(self) -> None:
        """An evolved program with long prompts should have higher string content."""
        ext = ChainFeatureExtractor()
        baseline_feat = ext.extract(_prog(_HOVER_BASELINE))

        # Simulate evolved program: add verbose stage_action + examples
        evolved = _HOVER_BASELINE.replace(
            '"stage_action": (\n'
            '                    "Read all retrieved passages and identify facts '
            'that are relevant "\n'
            '                    "to verifying the claim. Summarize the most '
            'important evidence found."\n'
            "                )",
            '"stage_action": (\n'
            '                    "Read all retrieved passages carefully. For each passage, '
            "extract specific entities, dates, numerical values, and key relationships. "
            "Cross-reference facts across passages. Identify contradictions. "
            "Format as structured bullet points. Include passage numbers as citations. "
            'Omit general background not relevant to the claim."\n'
            "                )",
        )
        evolved_feat = ext.extract(_prog(evolved))

        assert (
            evolved_feat["total_string_content"] > baseline_feat["total_string_content"]
        )

    def test_evolved_with_system_prompt(self) -> None:
        """Evolved program with non-empty system_prompt detected."""
        ext = ChainFeatureExtractor()
        evolved = _HOVER_BASELINE.replace(
            '"system_prompt": ""',
            '"system_prompt": "You are an evidence retrieval assistant."',
        )
        features = ext.extract(_prog(evolved))
        assert features["has_system_prompt"] == 1.0

    def test_evolved_with_examples(self) -> None:
        """Evolved program with few-shot examples detected."""
        ext = ChainFeatureExtractor()
        evolved = _HOVER_BASELINE.replace(
            '"example_reasoning": "<none>"',
            '"example_reasoning": "Example 1:\\nClaim: ...\\nExample 2:\\nClaim: ..."',
            1,  # replace only first occurrence
        )
        features = ext.extract(_prog(evolved))
        assert features["n_examples"] == 2.0

    def test_dependency_fan_in(self) -> None:
        """Max dependency fan-in detected from dependencies lists."""
        ext = ChainFeatureExtractor()
        # HoVer step 5 depends on [2, 4] — fan-in of 2
        features = ext.extract(_prog(_HOVER_BASELINE))
        assert features["max_dependency_fan_in"] == 2.0

    def test_hover_more_complex_than_hotpotqa(self) -> None:
        """HoVer has more steps and deep retrieval => higher predicted complexity."""
        ext = ChainFeatureExtractor()
        hover = ext.extract(_prog(_HOVER_BASELINE))
        hotpotqa = ext.extract(_prog(_HOTPOTQA_BASELINE))

        assert hover["n_total_steps"] > hotpotqa["n_total_steps"]
        assert hover["n_deep_retrieval"] > hotpotqa["n_deep_retrieval"]
        assert hover["code_length"] > hotpotqa["code_length"]

    def test_non_chain_code_graceful(self) -> None:
        """ChainFeatureExtractor handles non-chain code without crashing."""
        ext = ChainFeatureExtractor()
        features = ext.extract(_prog("def solve(): return 42"))
        assert features["n_total_steps"] == 0.0
        assert features["n_examples"] == 0.0
        assert features["has_system_prompt"] == 0.0


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------


class TestConstantPredictor:
    def test_returns_constant(self) -> None:
        p = ConstantPredictor(42.0)
        assert p.predict(_prog()) == 42.0
        assert p.predict(_long_prog()) == 42.0

    def test_always_warm(self) -> None:
        assert ConstantPredictor().is_warm()

    def test_update_is_noop(self) -> None:
        p = ConstantPredictor()
        p.update(_prog(), 100.0)
        assert p.predict(_prog()) == 1.0  # unchanged


class TestSimpleHeuristicPredictor:
    def test_cold_start_uses_default_rate(self) -> None:
        p = SimpleHeuristicPredictor(default_rate=0.5)
        prog = _prog("x" * 200)
        pred = p.predict(prog)
        assert pred == pytest.approx(200 * 0.5)

    def test_cold_start_code_length_floor(self) -> None:
        p = SimpleHeuristicPredictor(default_rate=1.0)
        prog = _prog("x")  # 1 char, below floor of 100
        pred = p.predict(prog)
        assert pred == pytest.approx(100 * 1.0)

    def test_is_warm_after_enough_updates(self) -> None:
        p = SimpleHeuristicPredictor()
        assert not p.is_warm()
        for i in range(5):
            p.update(_prog("x" * 200), 100.0)
        assert p.is_warm()

    def test_learns_from_updates(self) -> None:
        p = SimpleHeuristicPredictor(default_rate=0.1, window_size=5)
        prog = _prog("x" * 200)

        pred_before = p.predict(prog)  # 200 * 0.1 = 20

        # Train with rate = 1.0 (200 chars, 200s eval)
        for _ in range(5):
            p.update(prog, 200.0)

        pred_after = p.predict(prog)
        assert pred_after > pred_before  # learned higher rate

    def test_longer_code_predicts_longer(self) -> None:
        p = SimpleHeuristicPredictor()
        short = _short_prog()
        long = _long_prog()
        assert p.predict(long) > p.predict(short)

    def test_ignores_non_positive_duration(self) -> None:
        p = SimpleHeuristicPredictor()
        p.update(_prog(), 0.0)
        p.update(_prog(), -5.0)
        assert not p.is_warm()  # no valid updates


class TestRidgePredictor:
    def test_cold_start_returns_default(self) -> None:
        p = RidgePredictor(default_prediction=500.0)
        pred = p.predict(_prog())
        assert pred >= 500.0
        assert not p.is_warm()

    def test_warm_after_training(self) -> None:
        p = RidgePredictor(min_samples=3)
        for i in range(3):
            code = "x" * (100 + i * 100)
            p.update(_prog(code), 100.0 + i * 50)
        assert p.is_warm()

    def test_predictions_vary_with_features(self) -> None:
        p = RidgePredictor(min_samples=5)
        # Train: longer code => longer eval
        for length in [200, 400, 600, 800, 1000]:
            code = "x" * length
            p.update(_prog(code), float(length))

        short_pred = p.predict(_prog("x" * 200))
        long_pred = p.predict(_prog("x" * 1000))
        assert long_pred > short_pred

    def test_custom_feature_extractor(self) -> None:
        class CustomExt:
            def extract(self, program: Program) -> dict[str, float]:
                return {"magic": float(len(program.code))}

        p = RidgePredictor(feature_extractor=CustomExt(), min_samples=3)
        for i in range(3):
            p.update(_prog("x" * (100 + i * 100)), 100.0 + i * 50)
        assert p.is_warm()
        pred = p.predict(_prog("x" * 500))
        assert pred > 0

    def test_prediction_floor_at_one(self) -> None:
        p = RidgePredictor(min_samples=3)
        # Train with tiny durations
        for _ in range(3):
            p.update(_prog("x" * 100), 0.01)
        pred = p.predict(_prog("x" * 100))
        assert pred >= 1.0

    def test_ignores_non_positive_duration(self) -> None:
        p = RidgePredictor(min_samples=3)
        p.update(_prog(), 0.0)
        p.update(_prog(), -10.0)
        assert not p.is_warm()


# ---------------------------------------------------------------------------
# Prioritizer tests
# ---------------------------------------------------------------------------


class TestFIFOPrioritizer:
    def test_preserves_order(self) -> None:
        progs = [_prog(f"code_{i}") for i in range(5)]
        result = FIFOPrioritizer().prioritize(progs)
        assert [p.id for p in result] == [p.id for p in progs]

    def test_empty_list(self) -> None:
        assert FIFOPrioritizer().prioritize([]) == []

    def test_does_not_modify_input(self) -> None:
        progs = [_prog("a"), _prog("b")]
        original_ids = [p.id for p in progs]
        FIFOPrioritizer().prioritize(progs)
        assert [p.id for p in progs] == original_ids

    def test_no_predictor(self) -> None:
        assert FIFOPrioritizer().predictor is None


class TestLPTPrioritizer:
    def test_longest_first(self) -> None:
        pred = SimpleHeuristicPredictor(default_rate=1.0)
        # Warm up predictor
        for _ in range(5):
            pred.update(_prog("x" * 200), 200.0)

        short = _prog("x" * 100)
        medium = _prog("x" * 500)
        long = _prog("x" * 1000)

        prioritizer = LPTPrioritizer(pred)
        result = prioritizer.prioritize([short, medium, long])

        # Longest should be first
        assert result[0].id == long.id
        assert result[-1].id == short.id

    def test_falls_back_to_fifo_when_cold(self) -> None:
        pred = SimpleHeuristicPredictor()
        assert not pred.is_warm()

        progs = [_prog(f"code_{i}") for i in range(3)]
        prioritizer = LPTPrioritizer(pred)
        result = prioritizer.prioritize(progs)
        assert [p.id for p in result] == [p.id for p in progs]

    def test_empty_list(self) -> None:
        prioritizer = LPTPrioritizer(ConstantPredictor())
        assert prioritizer.prioritize([]) == []

    def test_has_predictor(self) -> None:
        pred = ConstantPredictor()
        prioritizer = LPTPrioritizer(pred)
        assert prioritizer.predictor is pred

    def test_does_not_modify_input(self) -> None:
        pred = ConstantPredictor()
        progs = [_prog("x" * 500), _prog("x" * 100)]
        original_ids = [p.id for p in progs]
        LPTPrioritizer(pred).prioritize(progs)
        assert [p.id for p in progs] == original_ids


class TestSJFPrioritizer:
    def test_shortest_first(self) -> None:
        pred = SimpleHeuristicPredictor(default_rate=1.0)
        for _ in range(5):
            pred.update(_prog("x" * 200), 200.0)

        short = _prog("x" * 100)
        medium = _prog("x" * 500)
        long = _prog("x" * 1000)

        prioritizer = SJFPrioritizer(pred)
        result = prioritizer.prioritize([long, medium, short])

        assert result[0].id == short.id
        assert result[-1].id == long.id

    def test_falls_back_to_fifo_when_cold(self) -> None:
        pred = SimpleHeuristicPredictor()
        progs = [_prog(f"code_{i}") for i in range(3)]
        result = SJFPrioritizer(pred).prioritize(progs)
        assert [p.id for p in result] == [p.id for p in progs]


# ---------------------------------------------------------------------------
# Integration: predictor + prioritizer work together
# ---------------------------------------------------------------------------


class TestPredictorPrioritizerIntegration:
    def test_lpt_with_trained_heuristic(self) -> None:
        """After training, LPT correctly reorders by predicted eval time."""
        pred = SimpleHeuristicPredictor(default_rate=0.1, window_size=10)

        # Train: code_length strongly correlates with eval time
        for length in [200, 400, 600, 800, 1000]:
            code = "x" * length
            pred.update(_prog(code), float(length) * 2)  # 2s per char

        assert pred.is_warm()
        prioritizer = LPTPrioritizer(pred)

        # Create programs of varying lengths
        short = _prog("x" * 150)
        medium = _prog("x" * 500)
        long = _prog("x" * 2000)

        result = prioritizer.prioritize([medium, short, long])
        # Long should be first, short last
        assert result[0].id == long.id
        assert result[-1].id == short.id

    def test_online_learning_improves_ordering(self) -> None:
        """Predictor learns and improves ordering over time."""
        pred = SimpleHeuristicPredictor(default_rate=0.1, window_size=10)
        prioritizer = LPTPrioritizer(pred)

        # Cold: FIFO order
        progs = [_prog("x" * 100), _prog("x" * 1000)]
        cold_result = prioritizer.prioritize(progs)
        assert [p.id for p in cold_result] == [p.id for p in progs]  # FIFO

        # Train
        for _ in range(5):
            pred.update(_prog("x" * 500), 500.0)

        # Warm: LPT order
        warm_result = prioritizer.prioritize(progs)
        assert warm_result[0].id == progs[1].id  # longer code first


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_custom_extractor_works_without_inheritance(self) -> None:
        """FeatureExtractor is a Protocol — structural subtyping suffices."""

        class MyExtractor:
            def extract(self, program: Program) -> dict[str, float]:
                return {"custom": 99.0}

        ext = MyExtractor()
        # Should work with CompositeFeatureExtractor (accepts FeatureExtractor)
        comp = CompositeFeatureExtractor([ext])
        assert comp.extract(_prog())["custom"] == 99.0

    def test_custom_predictor_with_lpt(self) -> None:
        """Custom EvalTimePredictor subclass works with LPT."""

        class AlwaysHighPredictor(EvalTimePredictor):
            def predict(self, program: Program) -> float:
                return float(len(program.code))

            def update(self, program: Program, actual_duration: float) -> None:
                pass

            def is_warm(self) -> bool:
                return True

        pred = AlwaysHighPredictor()
        prioritizer = LPTPrioritizer(pred)
        short = _prog("x" * 100)
        long = _prog("x" * 1000)
        result = prioritizer.prioritize([short, long])
        assert result[0].id == long.id
