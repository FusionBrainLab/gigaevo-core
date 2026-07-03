"""Tests for DirectoryProgramLoader glob pattern."""

from __future__ import annotations

from gigaevo.problems.initial_loaders import DirectoryProgramLoader


class _Storage:
    def __init__(self):
        self.added = []

    async def add(self, program):
        self.added.append(program)


def _make_problem_dir(tmp_path):
    initial = tmp_path / "initial_programs"
    initial.mkdir()
    (initial / "seed.py").write_text("def entrypoint(): return 1\n")
    (initial / "chain.json").write_text('{"steps": []}')
    return tmp_path


async def test_default_pattern_loads_only_python(tmp_path):
    problem_dir = _make_problem_dir(tmp_path)
    programs = await DirectoryProgramLoader(problem_dir).load(_Storage())
    assert [p.metadata["strategy_name"] for p in programs] == ["seed"]


async def test_json_pattern_loads_only_json(tmp_path):
    problem_dir = _make_problem_dir(tmp_path)
    loader = DirectoryProgramLoader(problem_dir, pattern="*.json")
    programs = await loader.load(_Storage())
    assert [p.metadata["strategy_name"] for p in programs] == ["chain"]
    assert programs[0].code == '{"steps": []}'


async def test_missing_dir_returns_empty(tmp_path):
    assert (
        await DirectoryProgramLoader(tmp_path, pattern="*.json").load(_Storage()) == []
    )
