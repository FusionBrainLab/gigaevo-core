"""Tests for gigaevo.experiment.preflight — checks 9, 15, 22."""

from unittest.mock import MagicMock, patch

from gigaevo.experiment.preflight import run_checks


def _make_manifest(stopping_rule=None, status="implemented", factorial_design=False):
    """Return a minimal mock manifest suitable for run_checks."""
    raw: dict = {"experiment": {}}
    if stopping_rule is not None:
        raw["experiment"]["stopping_rule"] = stopping_rule
    if factorial_design:
        raw["experiment"]["factorial_design"] = True

    m = MagicMock()
    m.status = status
    m.runs = []
    m.servers = []
    m._raw = raw
    return m


def _make_run(label, pipeline="standard", problem_name="my_problem"):
    r = MagicMock()
    r.label = label
    r.pipeline = pipeline
    r.problem_name = problem_name
    r.chain_url = None
    r.mutation_url = None
    r.db = 15
    return r


def _get_check(results, num: int):
    return next(r for r in results if r.num == num)


class TestCheck22StoppingRule:
    """Check 22: stopping rule present and non-vague."""

    def _run(self, stopping_rule=None):
        manifest = _make_manifest(stopping_rule=stopping_rule)
        with patch(
            "gigaevo.experiment.manifest.load_manifest",
            return_value=manifest,
        ):
            results = run_checks("test/smoke")
        return _get_check(results, 22)

    def test_valid_max_generations_passes(self):
        c = self._run("max_generations=50")
        assert c.passed
        assert "max_generations=50" in c.message

    def test_valid_with_metric_threshold_passes(self):
        c = self._run(
            "max_generations=100 OR frontier_fitness>0.90_for_3_consecutive_gens"
        )
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


class TestCheck15SingleIV:
    """Check 15: single IV per run comparison / factorial design support."""

    def _run(self, runs, factorial_design=False, stopping_rule="max_generations=50"):
        manifest = _make_manifest(
            stopping_rule=stopping_rule, factorial_design=factorial_design
        )
        manifest.runs = runs
        with patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest):
            results = run_checks("test/smoke")
        return _get_check(results, 15)

    def test_single_cell_passes(self):
        runs = [_make_run("A"), _make_run("B")]
        c = self._run(runs)
        assert c.passed

    def test_two_cells_passes(self):
        runs = [
            _make_run("A", pipeline="p1", problem_name="prob1"),
            _make_run("B", pipeline="p2", problem_name="prob2"),
        ]
        c = self._run(runs)
        assert c.passed

    def test_three_cells_without_factorial_flag_fails(self):
        runs = [
            _make_run("A", pipeline="p1", problem_name="prob1"),
            _make_run("B", pipeline="p1", problem_name="prob2"),
            _make_run("C", pipeline="p2", problem_name="prob1"),
        ]
        c = self._run(runs, factorial_design=False)
        assert not c.passed
        assert "factorial_design: true" in c.message

    def test_three_cells_with_factorial_flag_passes(self):
        runs = [
            _make_run("A", pipeline="p1", problem_name="prob1"),
            _make_run("B", pipeline="p1", problem_name="prob2"),
            _make_run("C", pipeline="p2", problem_name="prob1"),
        ]
        c = self._run(runs, factorial_design=True)
        assert c.passed
        assert "Factorial design declared" in c.message

    def test_factorial_flag_reports_cell_count(self):
        runs = [
            _make_run("A", pipeline="pip", problem_name="a"),
            _make_run("B", pipeline="pip", problem_name="b"),
            _make_run("C", pipeline="pip", problem_name="c"),
            _make_run("D", pipeline="pip", problem_name="d"),
        ]
        c = self._run(runs, factorial_design=True)
        assert c.passed
        assert "4" in c.message


class TestCheck9LiveWriters:
    """Check 9: dbsize==0, with live writer reporting."""

    def _run(self, dbsize=0, live_pids=None):
        manifest = _make_manifest(stopping_rule="max_generations=50")
        run = _make_run("A")
        run.db = 15
        manifest.runs = [run]

        mock_redis = MagicMock()
        mock_redis.dbsize.return_value = dbsize

        pids = live_pids or []
        with (
            patch("gigaevo.experiment.manifest.load_manifest", return_value=manifest),
            patch("redis.Redis", return_value=mock_redis),
            patch(
                "gigaevo.experiment.preflight._find_run_pids_for_db",
                return_value=pids,
            ),
        ):
            results = run_checks("test/smoke")
        return _get_check(results, 9)

    def test_empty_db_passes(self):
        c = self._run(dbsize=0)
        assert c.passed

    def test_nonempty_db_fails(self):
        c = self._run(dbsize=100)
        assert not c.passed
        assert "flush first" in c.message

    def test_nonempty_db_with_live_writer_reports_pid(self):
        c = self._run(dbsize=100, live_pids=[12345])
        assert not c.passed
        assert "12345" in c.message
        assert "live writer" in c.message
