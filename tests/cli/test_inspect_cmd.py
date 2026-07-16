"""Tests for non-blocking Redis prefix discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import redis

from gigaevo.cli import main
from gigaevo.cli.inspect_cmd import discover_prefixes


def _redis_with_keys(keys_by_pattern: dict[str, list[str]]) -> MagicMock:
    client = MagicMock()

    def scan_iter(*, match: str, count: int):
        assert count == 1000
        return iter(keys_by_pattern.get(match, []))

    client.scan_iter.side_effect = scan_iter
    client.keys.side_effect = AssertionError("Redis KEYS must not be used")
    return client


def test_discovers_active_and_completed_runs_without_keys() -> None:
    client = _redis_with_keys(
        {
            "*:__instance_lock__": ["active/run:__instance_lock__"],
            "*:run_state": ["completed/run:run_state"],
            "*:program:*": ["legacy/run:program:abc"],
            "*:metrics:*": [
                "metrics/run:metrics:latest",
                "metrics/run:metrics:history:path:metrics:value",
            ],
            "*:status:*": ["status/run:status:DONE"],
            "*:archive": ["archive/run:archive"],
        }
    )
    with patch("gigaevo.cli.inspect_cmd.redis.Redis", return_value=client):
        prefixes = discover_prefixes("localhost", 6379, 4)

    assert prefixes == [
        "active/run",
        "archive/run",
        "completed/run",
        "legacy/run",
        "metrics/run",
        "status/run",
    ]
    client.keys.assert_not_called()
    client.close.assert_called_once()


def test_inspect_prints_each_discovered_prefix() -> None:
    runner = CliRunner()
    with patch(
        "gigaevo.cli.inspect_cmd.discover_prefixes",
        return_value=["a", "b"],
    ):
        result = runner.invoke(main, ["inspect", "--db", "4"])
    assert result.exit_code == 0
    assert "db=4  prefix=a" in result.output
    assert "db=4  prefix=b" in result.output


def test_inspect_reports_connection_errors_without_traceback() -> None:
    runner = CliRunner()
    with patch(
        "gigaevo.cli.inspect_cmd.discover_prefixes",
        side_effect=redis.ConnectionError("refused"),
    ):
        result = runner.invoke(main, ["inspect", "--db", "4"])

    assert result.exit_code == 1
    assert "db=4" in result.output
    assert "refused" in result.output
