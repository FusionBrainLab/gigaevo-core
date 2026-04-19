"""Tests for the new typed sub-models being introduced for schema v2.

These models are added additively in step 3 of the manifest refactoring:
they exist in gigaevo.experiment.manifest but are not yet wired into
ExperimentManifest (that happens in step 4 when the ContractSection /
LifecycleState / TelemetryLog / ControlPlane groups are added).

The goal of this file is:
  1. Exercise each new model's construction, defaults, and round-trip.
  2. Lock down the target v2 shape now so step 4 has a firm target.
  3. Confirm existing v1 yamls still load via ExperimentManifest — the
     new models must not accidentally change validation behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gigaevo.experiment.manifest import (
    CheckpointAnalysisEntry,
    CheckpointAnalysisInfo,
    CheckpointEntry,
    CheckResult,
    ConfigSpec,
    ExperimentManifest,
    MidRunTestEvalInfo,
    NotificationsSection,
    PrChannelConfig,
    RunMetric,
    TelegramChannelConfig,
    ToolRef,
    TreatmentChecksInfo,
)

REPO_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# RunMetric + CheckpointEntry
# ---------------------------------------------------------------------------


class TestRunMetric:
    def test_minimum_fields(self):
        m = RunMetric(label="R1", gen=3)
        assert m.label == "R1"
        assert m.gen == 3
        assert m.recent_invalidity is None

    def test_accepts_best_fitness_via_extra_allow(self):
        """Problem-specific `best_{metric}` fields must survive round-trip."""
        m = RunMetric.model_validate(
            {"label": "R1", "gen": 5, "best_fitness": 0.82, "recent_invalidity": 0.1}
        )
        dumped = m.model_dump()
        assert dumped["best_fitness"] == 0.82
        assert dumped["recent_invalidity"] == 0.1

    def test_accepts_best_actual_fitness_via_extra_allow(self):
        """Heilbron uses `best_actual_fitness` — must also round-trip."""
        m = RunMetric.model_validate(
            {"label": "A1_G", "gen": 14, "best_actual_fitness": 0.02683}
        )
        assert m.model_dump()["best_actual_fitness"] == 0.02683


class TestCheckpointEntry:
    def test_minimum_fields(self):
        cp = CheckpointEntry(gen=10, timestamp="2026-04-15T00:00:00Z")
        assert cp.gen == 10
        assert cp.run_metrics == []
        assert cp.notes == ""

    def test_with_run_metrics(self):
        cp = CheckpointEntry.model_validate(
            {
                "gen": 20,
                "timestamp": "2026-04-15T07:24:03Z",
                "metric_name": "actual_fitness",  # legacy field, dropped silently
                "run_metrics": [
                    {"label": "A1_G", "gen": 29, "best_actual_fitness": 0.028},
                    {"label": "A1_D", "gen": 29, "best_actual_fitness": 0.031},
                ],
                "notes": "Routine checkpoint",
            }
        )
        assert len(cp.run_metrics) == 2
        assert cp.run_metrics[0].label == "A1_G"
        # Metric preserved via extra="allow" on RunMetric
        assert cp.run_metrics[0].model_dump()["best_actual_fitness"] == 0.028
        # metric_name on the checkpoint itself was removed; canonical source
        # is contract.problem.metric_name.
        assert "metric_name" not in cp.model_dump()


# ---------------------------------------------------------------------------
# Mid-run test eval + checkpoint analysis
# ---------------------------------------------------------------------------


class TestMidRunTestEvalInfo:
    def test_defaults(self):
        info = MidRunTestEvalInfo()
        assert info.completed is False
        assert info.completed_at is None

    def test_results_dict_round_trip(self):
        """Arbitrary per-run results dict (e.g. hover/memory shape) survives."""
        info = MidRunTestEvalInfo.model_validate(
            {
                "completed": True,
                "completed_at": "2026-04-04T22:21:12Z",
                "results": {
                    "R1": {"mean": 0.59, "std": 0.02, "n": 5},
                    "R2": {"mean": 0.58, "std": 0.01, "n": 5},
                },
            }
        )
        assert info.completed is True
        assert info.results["R1"]["mean"] == 0.59

    def test_notes_field_round_trip(self):
        """Older yamls use `notes` rather than structured `results`."""
        info = MidRunTestEvalInfo.model_validate(
            {
                "completed": True,
                "completed_at": "2026-03-24T20:04:00Z",
                "notes": "D2 only (gen 16/25, 64%). Coverage: 51.73%.",
            }
        )
        assert info.notes.startswith("D2 only")


class TestCheckpointAnalysis:
    def test_entry_defaults(self):
        e = CheckpointAnalysisEntry()
        assert e.completed is False
        assert e.summary == ""
        assert e.notes == ""

    def test_mid_run_round_trip(self):
        info = CheckpointAnalysisInfo.model_validate(
            {
                "mid_run": {
                    "completed": True,
                    "completed_at": "2026-04-15T03:27:20Z",
                    "summary": "NULL/SUGGESTIVE. Treatment +0.65pp val.",
                }
            }
        )
        assert info.mid_run.completed is True
        assert "NULL/SUGGESTIVE" in info.mid_run.summary

    def test_extra_stages_preserved(self):
        """Allow future stages like `final` to round-trip via extra='allow'."""
        info = CheckpointAnalysisInfo.model_validate(
            {
                "mid_run": {"completed": True},
                "final": {"completed": False, "notes": "pending"},
            }
        )
        dumped = info.model_dump()
        assert dumped.get("final", {}).get("notes") == "pending"


# ---------------------------------------------------------------------------
# Treatment checks
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_minimum(self):
        r = CheckResult(name="log_pattern_present", passed=True)
        assert r.passed is True
        assert r.detail == ""

    def test_with_detail(self):
        r = CheckResult(name="FetchOpponent", passed=False, detail="absent in run X")
        assert r.passed is False
        assert "absent" in r.detail


class TestTreatmentChecksInfo:
    def test_defaults(self):
        info = TreatmentChecksInfo()
        assert info.completed is False
        assert info.results == []

    def test_with_results_list(self):
        info = TreatmentChecksInfo.model_validate(
            {
                "completed": True,
                "completed_at": "2026-04-12T23:30:00Z",
                "results": [
                    {"name": "MainRunSyncHook", "passed": True},
                    {
                        "name": "ProgressBasedSyncHook",
                        "passed": False,
                        "detail": "not found in logs",
                    },
                ],
            }
        )
        assert info.completed is True
        assert len(info.results) == 2
        assert info.results[1].passed is False


# ---------------------------------------------------------------------------
# Tool refs + config spec
# ---------------------------------------------------------------------------


class TestToolRef:
    def test_minimum(self):
        t = ToolRef(name="diagnose")
        assert t.name == "diagnose"
        assert t.path == ""

    def test_full(self):
        t = ToolRef(
            name="diagnose",
            path="tools/experiment/diagnose.py",
            purpose="runtime diagnostics",
        )
        assert t.path.endswith(".py")
        assert "diagnostics" in t.purpose


class TestConfigSpec:
    def test_defaults(self):
        c = ConfigSpec()
        assert c.pipeline is None
        assert c.flat_overrides == {}

    def test_structured_plus_extra(self):
        c = ConfigSpec.model_validate(
            {
                "pipeline": "adversarial_asymmetric",
                "problem_name": "heilbron_adversarial",
                "llm_model": "Qwen3-235B",
                "max_generations": 50,
                # untyped knobs flow into model_extra automatically
                "num_parents": 1,
                "max_elites_per_generation": 8,
                "stage_timeout": 2400,
            }
        )
        assert c.pipeline == "adversarial_asymmetric"
        assert c.flat_overrides["num_parents"] == 1
        assert c.flat_overrides["stage_timeout"] == 2400


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


class TestPrChannelConfig:
    def test_defaults(self):
        p = PrChannelConfig()
        assert p.enabled is True
        assert p.comment_mode == "rolling"

    def test_new_mode(self):
        p = PrChannelConfig(comment_mode="new")
        assert p.comment_mode == "new"

    def test_rejects_unknown_comment_mode(self):
        with pytest.raises(Exception):
            PrChannelConfig.model_validate({"comment_mode": "bogus"})


class TestTelegramChannelConfig:
    def test_defaults(self):
        t = TelegramChannelConfig()
        assert t.enabled is True
        assert t.chat_id_env == "TELEGRAM_CHAT_ID"
        assert t.token_env == "TELEGRAM_BOT_TOKEN"

    def test_custom_envs(self):
        t = TelegramChannelConfig(
            enabled=False, chat_id_env="EXP_CHAT", token_env="EXP_TOKEN"
        )
        assert t.enabled is False
        assert t.chat_id_env == "EXP_CHAT"


class TestNotificationsSection:
    def test_defaults_enable_both_channels(self):
        n = NotificationsSection()
        assert n.pr.enabled is True
        assert n.telegram.enabled is True

    def test_round_trip(self):
        n = NotificationsSection.model_validate(
            {
                "pr": {"enabled": True, "comment_mode": "rolling"},
                "telegram": {
                    "enabled": False,
                    "chat_id_env": "TELEGRAM_CHAT_ID",
                    "token_env": "TELEGRAM_BOT_TOKEN",
                },
            }
        )
        assert n.pr.comment_mode == "rolling"
        assert n.telegram.enabled is False


# ---------------------------------------------------------------------------
# Regression: existing v1 yamls still load cleanly
# ---------------------------------------------------------------------------


def _all_v1_yamls() -> list[Path]:
    """All non-template experiment.yaml files under experiments/."""
    return [
        p
        for p in (REPO_ROOT / "experiments").glob("*/*/experiment.yaml")
        if "_template" not in str(p)
    ]


class TestExistingYamlsStillLoad:
    """Adding new typed models must not change how existing yamls validate."""

    def test_sanity_at_least_ten_yamls_discovered(self):
        yamls = _all_v1_yamls()
        assert len(yamls) >= 10, (
            f"Expected >=10 real experiment.yaml files, found {len(yamls)}"
        )

    @pytest.mark.parametrize(
        "yaml_path",
        _all_v1_yamls(),
        ids=lambda p: str(p.relative_to(REPO_ROOT)),
    )
    def test_existing_yaml_loads(self, yaml_path: Path):
        raw = yaml.safe_load(yaml_path.read_text())
        # Must still load under current schema (extra fields ignored).
        ExperimentManifest.from_dict(raw)
