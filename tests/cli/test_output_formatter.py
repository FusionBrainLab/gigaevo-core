"""Tests for OutputFormatter: structured output in table/json/csv/markdown."""

from __future__ import annotations

import json
from unittest.mock import patch

from gigaevo.cli.output_formatter import OutputFormatter

SAMPLE_ROWS = [
    {"run": "A", "gen": 10, "fitness": 0.65},
    {"run": "B", "gen": 15, "fitness": 0.72},
]

SAMPLE_COLUMNS = ["run", "gen", "fitness"]


class TestFormatTable:
    def test_produces_table(self):
        fmt = OutputFormatter(format_name="table")
        result = fmt.render(SAMPLE_ROWS, columns=SAMPLE_COLUMNS)
        assert "A" in result
        assert "B" in result
        assert "gen" in result.lower() or "Gen" in result


class TestFormatJson:
    def test_produces_valid_json(self):
        fmt = OutputFormatter(format_name="json")
        result = fmt.render(SAMPLE_ROWS)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["run"] == "A"

    def test_json_preserves_types(self):
        fmt = OutputFormatter(format_name="json")
        result = fmt.render(SAMPLE_ROWS)
        parsed = json.loads(result)
        assert parsed[1]["gen"] == 15
        assert parsed[1]["fitness"] == 0.72


class TestFormatCsv:
    def test_produces_csv_with_header(self):
        fmt = OutputFormatter(format_name="csv")
        result = fmt.render(SAMPLE_ROWS, columns=SAMPLE_COLUMNS)
        lines = result.strip().splitlines()
        assert lines[0].strip() == "run,gen,fitness"
        assert len(lines) == 3

    def test_csv_data_rows(self):
        fmt = OutputFormatter(format_name="csv")
        result = fmt.render(SAMPLE_ROWS, columns=SAMPLE_COLUMNS)
        lines = result.strip().splitlines()
        assert "A" in lines[1]
        assert "10" in lines[1]


class TestFormatMarkdown:
    def test_produces_markdown_table(self):
        fmt = OutputFormatter(format_name="markdown")
        result = fmt.render(SAMPLE_ROWS, columns=SAMPLE_COLUMNS)
        lines = result.strip().split("\n")
        assert "|" in lines[0]
        assert "---" in lines[1]
        assert len(lines) == 4


class TestPipeDetection:
    def test_auto_detect_pipe_to_json(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            fmt = OutputFormatter(format_name=None)
            assert fmt.effective_format == "json"

    def test_auto_detect_terminal_to_table(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = True
            fmt = OutputFormatter(format_name=None)
            assert fmt.effective_format == "table"

    def test_explicit_format_overrides_pipe(self):
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            fmt = OutputFormatter(format_name="table")
            assert fmt.effective_format == "table"


class TestQuietMode:
    def test_echo_is_noop_when_quiet(self):
        fmt = OutputFormatter(format_name="json", quiet=True)
        result = fmt.echo(SAMPLE_ROWS)
        assert result is None
