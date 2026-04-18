"""Tests for gigaevo.experiment.checks — minimal principled preflight."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gigaevo.experiment.checks import (
    CheckResult,
    Severity,
    _check_gigaevo_python,
    _check_smoke_test,
    _check_treatment_verification,
    run_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(label: str = "A", db: int = 5) -> MagicMock:
    run = MagicMock()
    run.label = label
    run.db = db
    run.pipeline = "standard"
    run.problem_name = "chains/hover"
    run.extra_overrides = [
        "llm_base_url=http://10.0.0.1:4000",
        "model_name=gpt-4o",
    ]
    return run


def _make_manifest(
    *,
    status: str = "implemented",
    runs: list | None = None,
) -> MagicMock:
    m = MagicMock()
    m.lifecycle.status = status
    m.lifecycle.smoke_test.completed = True
    m.lifecycle.smoke_test.completed_at = "2026-04-14T10:00:00Z"
    m.lifecycle.treatment_verification.completed = True
    m.lifecycle.treatment_verification.completed_at = "2026-04-14T10:00:00Z"
    m.contract.runs = runs or [_make_run("A", 5), _make_run("B", 6)]
    m.contract.custom_env = {"OPENAI_API_KEY": "test-key"}
    m.contract.config.extra = {}
    m.contract.problem.has_test_set = False
    return m


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_pass_str(self):
        r = CheckResult(name="test", severity=Severity.CRITICAL)
        r.ok("all good")
        assert r.passed
        assert "PASS" in str(r)

    def test_fail_str(self):
        r = CheckResult(name="test", severity=Severity.CRITICAL)
        r.fail("bad thing")
        assert not r.passed
        assert "CRITICAL" in str(r)

    def test_is_blocking_critical_fail(self):
        r = CheckResult(name="a", severity=Severity.CRITICAL)
        r.fail("bad")
        assert r.is_blocking

    def test_not_blocking_warn(self):
        r = CheckResult(name="b", severity=Severity.WARN)
        r.fail("minor")
        assert not r.is_blocking

    def test_not_blocking_pass(self):
        r = CheckResult(name="c", severity=Severity.CRITICAL)
        r.ok()
        assert not r.is_blocking


# ---------------------------------------------------------------------------
# Status gate (tested via run_checks — early return on bad status)
# ---------------------------------------------------------------------------


class TestStatusGate:
    def test_rejects_non_implemented(self):
        m = _make_manifest(status="preregistered")
        with patch("gigaevo.experiment.checks.load_manifest", return_value=m):
            results = run_checks("hover/test")
        assert len(results) == 1
        assert not results[0].passed
        assert "preregistered" in results[0].message

    def test_load_failure_returns_early(self):
        with patch(
            "gigaevo.experiment.checks.load_manifest",
            side_effect=FileNotFoundError("not found"),
        ):
            results = run_checks("hover/test")
        assert len(results) == 1
        assert not results[0].passed


# ---------------------------------------------------------------------------
# Individual check functions (tested in isolation)
# ---------------------------------------------------------------------------


class TestGigaevoPython:
    def test_not_set(self):
        results: list[CheckResult] = []
        with patch.dict("os.environ", {}, clear=True):
            _check_gigaevo_python(results)
        assert len(results) == 1
        assert not results[0].passed
        assert "not set" in results[0].message

    def test_set_and_exists(self, tmp_path):
        fake_python = tmp_path / "python3"
        fake_python.write_text("#!/bin/sh\n")
        results: list[CheckResult] = []
        with patch.dict("os.environ", {"GIGAEVO_PYTHON": str(fake_python)}):
            _check_gigaevo_python(results)
        assert results[0].passed

    def test_set_but_missing(self):
        results: list[CheckResult] = []
        with patch.dict("os.environ", {"GIGAEVO_PYTHON": "/nonexistent/python3"}):
            _check_gigaevo_python(results)
        assert not results[0].passed
        assert "does not exist" in results[0].message


class TestSmokeTest:
    def test_completed(self):
        m = _make_manifest()
        results: list[CheckResult] = []
        _check_smoke_test(results, m)
        assert results[0].passed

    def test_not_completed(self):
        m = _make_manifest()
        m.lifecycle.smoke_test.completed = False
        results: list[CheckResult] = []
        _check_smoke_test(results, m)
        assert not results[0].passed


class TestTreatmentVerification:
    def test_completed(self):
        m = _make_manifest()
        results: list[CheckResult] = []
        _check_treatment_verification(results, m)
        assert results[0].passed

    def test_not_completed(self):
        m = _make_manifest()
        m.lifecycle.treatment_verification.completed = False
        results: list[CheckResult] = []
        _check_treatment_verification(results, m)
        assert not results[0].passed
