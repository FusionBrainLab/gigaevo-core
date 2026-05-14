"""Tests for the `gigaevo metrics` CLI subcommand."""

from __future__ import annotations

import json

from click.testing import CliRunner
import fakeredis

from gigaevo.cli import main


def _entry(
    step: int, value: float, ts: float = 1700000000.0, kind: str = "scalar"
) -> str:
    """Serialize one history entry matching RedisMetricsBackend's format."""
    return json.dumps({"s": step, "t": ts, "v": value, "k": kind})


def _populate(
    server: fakeredis.FakeServer,
    db: int,
    prefix: str,
    tag: str,
    steps: list[tuple[int, float]],
    kind: str = "scalar",
) -> None:
    """Populate `latest` + `history:<safe_tag>` for one tag, one prefix."""
    r = fakeredis.FakeRedis(server=server, db=db, decode_responses=True)
    key_prefix = f"{prefix}:metrics"
    safe_tag = tag.replace("/", ":").replace(" ", "_")
    # latest hash — enumerated by `list_tags`
    if steps:
        r.hset(f"{key_prefix}:latest", tag, str(steps[-1][1]))
    for step, value in steps:
        r.rpush(
            f"{key_prefix}:history:{safe_tag}",
            _entry(step, value, kind=kind),
        )


def _make_obj(server: fakeredis.FakeServer) -> dict:
    return {
        "redis_factory": lambda db: fakeredis.FakeRedis(
            server=server, db=db, decode_responses=True
        ),
    }


class TestMetricsBasic:
    def test_plain_output_one_record_per_line(self):
        """Default plain output emits one tab-separated record per line."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "loss", [(1, 0.5), (2, 0.4)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().splitlines() if line]
        assert len(lines) == 2
        assert "loss" in lines[0]
        assert "step=1" in lines[0]
        assert "value=0.5" in lines[0]
        assert "wall=" in lines[0]

    def test_empty_redis_produces_no_records(self):
        """No data → empty output, exit 0."""
        server = fakeredis.FakeServer()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""


class TestMetricsTagGlob:
    def test_glob_filters_to_matching_tags(self):
        """--tag glob restricts output to fnmatch-matching tag names."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "llm/tokens/input", [(1, 100.0)])
        _populate(server, 4, "p", "llm/tokens/output", [(1, 50.0)])
        _populate(server, 4, "p", "valid/frontier/fitness", [(1, 0.9)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--tag", "llm/tokens/*"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "llm/tokens/input" in result.output
        assert "llm/tokens/output" in result.output
        assert "valid/frontier/fitness" not in result.output

    def test_exact_tag_pattern_matches_single_tag(self):
        """An exact tag name (no glob chars) selects only that tag."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "a/b", [(1, 1.0)])
        _populate(server, 4, "p", "a/b/c", [(1, 2.0)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--tag", "a/b"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().splitlines() if line]
        assert len(lines) == 1
        assert "a/b\t" in lines[0]


class TestMetricsStepFilter:
    def test_since_until_filters_records(self):
        """--since/--until trim the per-tag history."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "loss", [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--since", "2", "--until", "3"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().splitlines() if line]
        assert len(lines) == 2
        assert "step=2" in lines[0]
        assert "step=3" in lines[1]


class TestMetricsTail:
    def test_tail_keeps_last_n_per_tag(self):
        """--tail N keeps only the last N records *per tag*."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "a", [(i, float(i)) for i in range(1, 6)])
        _populate(server, 4, "p", "b", [(i, float(i)) for i in range(1, 6)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--tail", "2"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().splitlines() if line]
        # 2 tags × 2 tail = 4 records
        assert len(lines) == 4
        a_lines = [line for line in lines if line.startswith("a\t")]
        b_lines = [line for line in lines if line.startswith("b\t")]
        assert len(a_lines) == 2
        assert len(b_lines) == 2
        assert "step=4" in a_lines[0]
        assert "step=5" in a_lines[1]


class TestMetricsFormat:
    def test_tsv_emits_header_row(self):
        """--format tsv prepends a tab-separated header."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "loss", [(1, 0.5)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--format", "tsv"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = result.output.strip().splitlines()
        assert lines[0].split("\t") == ["tag", "step", "wall", "kind", "value"]
        assert lines[1].startswith("loss\t1\t")

    def test_json_emits_list_of_objects(self):
        """--format json emits a parseable JSON array."""
        server = fakeredis.FakeServer()
        _populate(server, 4, "p", "loss", [(1, 0.5), (2, 0.4)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--format", "json"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]["tag"] == "loss"
        assert data[0]["step"] == 1
        assert data[0]["value"] == 0.5
        assert data[0]["kind"] == "scalar"
        assert "wall" in data[0]


class TestMetricsKindFilter:
    def test_default_kind_scalar_drops_non_scalar(self):
        """Default `--kind scalar` excludes records with other kinds."""
        server = fakeredis.FakeServer()
        # Manually push a 'text' history entry on the same tag.
        r = fakeredis.FakeRedis(server=server, db=4, decode_responses=True)
        r.hset("p:metrics:latest", "mixed", "x")
        r.rpush("p:metrics:history:mixed", _entry(1, 0.5, kind="scalar"))
        r.rpush("p:metrics:history:mixed", _entry(2, 0.0, kind="text"))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().splitlines() if line]
        assert len(lines) == 1
        assert "step=1" in lines[0]

    def test_kind_all_includes_every_record(self):
        """--kind all returns every history entry regardless of kind."""
        server = fakeredis.FakeServer()
        r = fakeredis.FakeRedis(server=server, db=4, decode_responses=True)
        r.hset("p:metrics:latest", "mixed", "x")
        r.rpush("p:metrics:history:mixed", _entry(1, 0.5, kind="scalar"))
        r.rpush("p:metrics:history:mixed", _entry(2, 0.0, kind="text"))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@4:O", "metrics", "--kind", "all"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.strip().splitlines() if line]
        assert len(lines) == 2


class TestMetricsMultipleRuns:
    def test_multiple_runs_include_label(self):
        """With >1 -r flag, plain output includes a label= field."""
        server = fakeredis.FakeServer()
        _populate(server, 1, "p", "loss", [(1, 0.1)])
        _populate(server, 2, "p", "loss", [(1, 0.2)])

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-r", "p@1:A", "-r", "p@2:B", "metrics"],
            obj=_make_obj(server),
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "label=A" in result.output
        assert "label=B" in result.output
