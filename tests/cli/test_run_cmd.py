"""Tests for `gigaevo run` -- problem resolution and run.py argv building."""

from __future__ import annotations

from pathlib import Path
import sys

import click
from click.testing import CliRunner
import pytest

from gigaevo.cli.run_cmd import build_run_argv, resolve_problem
from gigaevo.problems.layout import ProblemLayout


def make_problem_dir(base: Path, name: str = "toy_problem") -> Path:
    problem_dir = base / name
    problem_dir.mkdir(parents=True)
    for filename in ProblemLayout.required_files():
        (problem_dir / filename).write_text("# stub\n")
    (problem_dir / ProblemLayout.INITIAL_PROGRAMS_DIR).mkdir()
    return problem_dir


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "run.py").write_text("# stub entrypoint\n")
    (repo / "config").mkdir()
    make_problem_dir(repo / "problems", "bundled_problem")
    make_problem_dir(repo / "problems" / "chains" / "hover", "full")
    return repo


class TestResolveProblem:
    def test_path_target_resolves_to_basename_and_abspath(
        self, tmp_path: Path, fake_repo: Path
    ):
        problem_dir = make_problem_dir(tmp_path, "external_problem")
        name, resolved = resolve_problem(str(problem_dir), fake_repo)
        assert name == "external_problem"
        assert resolved == problem_dir.resolve()

    def test_bare_name_resolves_under_repo_problems(self, fake_repo: Path):
        name, resolved = resolve_problem("bundled_problem", fake_repo)
        assert name == "bundled_problem"
        assert resolved == (fake_repo / "problems" / "bundled_problem").resolve()

    def test_local_directory_shadows_bundled_name(
        self, tmp_path: Path, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        local = make_problem_dir(tmp_path / "cwd", "bundled_problem")
        monkeypatch.chdir(tmp_path / "cwd")
        name, resolved = resolve_problem("bundled_problem", fake_repo)
        assert name == "bundled_problem"
        assert resolved == local.resolve()

    def test_missing_required_files_lists_them(self, tmp_path: Path, fake_repo: Path):
        incomplete = tmp_path / "incomplete"
        incomplete.mkdir()
        (incomplete / ProblemLayout.TASK_DESCRIPTION).write_text("desc\n")
        with pytest.raises(click.ClickException) as excinfo:
            resolve_problem(str(incomplete), fake_repo)
        assert ProblemLayout.VALIDATOR in str(excinfo.value)
        assert ProblemLayout.METRICS_FILE in str(excinfo.value)

    def test_nested_bundled_name_resolves(self, fake_repo: Path):
        name, resolved = resolve_problem("chains/hover/full", fake_repo)
        assert name == "chains/hover/full"
        assert (
            resolved == (fake_repo / "problems" / "chains" / "hover" / "full").resolve()
        )

    def test_nonexistent_relative_target_lists_both_locations(
        self, tmp_path: Path, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(click.ClickException) as excinfo:
            resolve_problem("no/such_dir", fake_repo)
        assert "no directory at" in str(excinfo.value)
        assert "no bundled problem at" in str(excinfo.value)

    def test_nonexistent_absolute_target_omits_bundled_hint(self, fake_repo: Path):
        with pytest.raises(click.ClickException) as excinfo:
            resolve_problem("/no/such_dir", fake_repo)
        assert "no bundled problem" not in str(excinfo.value)

    def test_unknown_bare_name_errors(self, fake_repo: Path):
        with pytest.raises(click.ClickException, match="not found"):
            resolve_problem("no_such_problem", fake_repo)

    def test_invalid_local_dir_mentions_shadowed_bundle(
        self, tmp_path: Path, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cwd = tmp_path / "cwd"
        (cwd / "bundled_problem").mkdir(parents=True)
        monkeypatch.chdir(cwd)
        with pytest.raises(click.ClickException) as excinfo:
            resolve_problem("bundled_problem", fake_repo)
        assert "shadows the bundled problem" in str(excinfo.value)


class TestBuildRunArgv:
    def test_injects_name_and_dir(self, tmp_path: Path):
        argv = build_run_argv(tmp_path, "toy", tmp_path / "problems" / "toy", ())
        assert argv[0] == sys.executable
        assert argv[1] == str(tmp_path / "run.py")
        assert "problem.name=toy" in argv
        assert f"problem.dir={tmp_path / 'problems' / 'toy'}" in argv

    def test_user_problem_name_wins(self, tmp_path: Path):
        argv = build_run_argv(
            tmp_path, "toy", tmp_path / "toy", ("problem.name=custom",)
        )
        assert "problem.name=custom" in argv
        assert "problem.name=toy" not in argv
        assert f"problem.dir={tmp_path / 'toy'}" in argv

    def test_user_problem_dir_wins(self, tmp_path: Path):
        argv = build_run_argv(
            tmp_path, "toy", tmp_path / "toy", ("problem.dir=/elsewhere",)
        )
        assert "problem.dir=/elsewhere" in argv
        assert f"problem.dir={tmp_path / 'toy'}" not in argv

    def test_overrides_and_flags_pass_through_in_order(self, tmp_path: Path):
        overrides = ("max_mutants=5", "--cfg", "job")
        argv = build_run_argv(tmp_path, "toy", tmp_path / "toy", overrides)
        assert argv[-3:] == list(overrides)

    @pytest.mark.parametrize(
        "override",
        ["+problem.name=x", "++problem.name=x", "~problem.name", "problem.name@pkg=x"],
    )
    def test_advanced_override_forms_suppress_name_injection(
        self, tmp_path: Path, override: str
    ):
        argv = build_run_argv(tmp_path, "toy", tmp_path / "toy", (override,))
        assert "problem.name=toy" not in argv
        assert override in argv

    def test_advanced_override_form_suppresses_dir_injection(self, tmp_path: Path):
        argv = build_run_argv(tmp_path, "toy", tmp_path / "toy", ("++problem.dir=/x",))
        assert f"problem.dir={tmp_path / 'toy'}" not in argv


class TestRunCommand:
    def test_run_help_exits_zero(self):
        from gigaevo.cli import main

        result = CliRunner().invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "problem" in result.output.lower()

    def test_root_help_lists_run(self):
        from gigaevo.cli import main

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_execs_run_py_with_composed_argv(
        self, tmp_path: Path, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from gigaevo.cli import main
        import gigaevo.cli.run_cmd as run_cmd

        problem_dir = make_problem_dir(tmp_path, "external_problem")
        captured: dict = {}

        def fake_execv(executable: str, argv: list[str]) -> None:
            captured["executable"] = executable
            captured["argv"] = argv

        monkeypatch.setattr(run_cmd, "find_repo_root", lambda: fake_repo)
        monkeypatch.setattr(run_cmd.os, "execv", fake_execv)

        cwd_before = Path.cwd()
        result = CliRunner().invoke(
            main,
            ["run", str(problem_dir), "max_mutants=1", "--cfg", "job"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert Path.cwd() == cwd_before
        assert captured["executable"] == sys.executable
        argv = captured["argv"]
        assert argv[1] == str(fake_repo / "run.py")
        assert "problem.name=external_problem" in argv
        assert f"problem.dir={problem_dir.resolve()}" in argv
        assert argv[-3:] == ["max_mutants=1", "--cfg", "job"]

    def test_double_dash_forwards_help_flag(
        self, tmp_path: Path, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from gigaevo.cli import main
        import gigaevo.cli.run_cmd as run_cmd

        problem_dir = make_problem_dir(tmp_path, "external_problem")
        captured: dict = {}
        monkeypatch.setattr(run_cmd, "find_repo_root", lambda: fake_repo)
        monkeypatch.setattr(
            run_cmd.os, "execv", lambda _exe, argv: captured.update(argv=argv)
        )

        result = CliRunner().invoke(
            main, ["run", str(problem_dir), "--", "--help"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert captured["argv"][-1] == "--help"

    def test_invalid_target_exits_nonzero(
        self, tmp_path: Path, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from gigaevo.cli import main
        import gigaevo.cli.run_cmd as run_cmd

        monkeypatch.setattr(run_cmd, "find_repo_root", lambda: fake_repo)
        result = CliRunner().invoke(main, ["run", str(tmp_path / "missing_dir")])
        assert result.exit_code != 0
        assert "not found" in result.output
