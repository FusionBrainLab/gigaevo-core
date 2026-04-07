"""Tests for tools.experiment.preflight_check — focused on Check 22 (stopping rule)."""

from unittest.mock import MagicMock, patch

from tools.experiment.preflight_check import run_checks


def _make_manifest(stopping_rule=None, status="implemented"):
    """Return a minimal mock manifest suitable for run_checks."""
    raw: dict = {"experiment": {}}
    if stopping_rule is not None:
        raw["experiment"]["stopping_rule"] = stopping_rule

    m = MagicMock()
    m.status = status
    m.runs = []
    m.servers = []
    m._raw = raw
    return m


def _get_check(results, num: int):
    return next(r for r in results if r.num == num)


class TestCheck22StoppingRule:
    """Check 22: stopping rule present and non-vague."""

    def _run(self, stopping_rule=None):
        manifest = _make_manifest(stopping_rule=stopping_rule)
        with patch(
            "tools.experiment.manifest.load_manifest",
            return_value=manifest,
        ):
            results = run_checks("test/smoke")
        return _get_check(results, 22)

    def test_valid_max_generations_passes(self):
        c = self._run("max_generations=50")
        assert c.passed
        assert "max_generations=50" in c.message

    def test_valid_with_metric_threshold_passes(self):
        c = self._run("max_generations=100 OR frontier_fitness>0.90_for_3_consecutive_gens")
        assert c.passed

    def test_missing_stopping_rule_fails_critical(self):
        c = self._run(stopping_rule=None)
        assert not c.passed
        assert c.severity == "CRITICAL"
        assert "not set" in c.message

    def test_empty_string_stopping_rule_fails(self):
        c = self._run(stopping_rule="")
        assert not c.passed
        assert c.severity == "CRITICAL"

    def test_whitespace_only_stopping_rule_fails(self):
        c = self._run(stopping_rule="   ")
        assert not c.passed
        assert c.severity == "CRITICAL"

    def test_tbd_is_vague(self):
        c = self._run(stopping_rule="TBD")
        assert not c.passed
        assert "vague" in c.message
        assert "tbd" in c.message

    def test_todo_is_vague(self):
        c = self._run(stopping_rule="TODO: fill this in later")
        assert not c.passed
        assert "vague" in c.message

    def test_na_is_vague(self):
        c = self._run(stopping_rule="N/A")
        assert not c.passed
        assert "vague" in c.message

    def test_when_results_look_good_is_vague(self):
        c = self._run(stopping_rule="stop when results look good enough")
        assert not c.passed
        assert "vague" in c.message

    def test_when_we_have_enough_is_vague(self):
        c = self._run(stopping_rule="continue until when we have enough data")
        assert not c.passed
        assert "vague" in c.message

    def test_none_literal_is_vague(self):
        c = self._run(stopping_rule="none")
        assert not c.passed
        assert "vague" in c.message

    def test_check_number_and_group(self):
        c = self._run("max_generations=50")
        assert c.num == 22
        assert c.group == "Design"
        assert c.severity == "CRITICAL"
