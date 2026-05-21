"""Tests for problems.prompts.utils — CallLog aggregation across retries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    """Load a problems/ submodule by file path.

    tests/problems/__init__.py shadows the top-level ``problems`` namespace
    package when tests import via ``from problems.* import …``, so we wire
    the dependency graph explicitly here.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_types = _load_module("prompts_types_under_test", "problems/prompts/types.py")
sys.modules.setdefault("problems", ModuleType("problems"))
sys.modules.setdefault("problems.prompts", ModuleType("problems.prompts"))
sys.modules["problems.prompts.types"] = _types

# problems.prompts.utils imports from gigaevo.* and problems.prompts.client at
# module load. To avoid pulling those heavy dependencies, we exercise the
# pure helper directly by re-importing it under a stub module name.
_utils_path = _ROOT / "problems" / "prompts" / "utils.py"
# Inject a stub ``problems.prompts.client`` so import does not require the
# real client (which transitively imports the OpenAI SDK).
_stub_client = ModuleType("problems.prompts.client")


class _StubLLMClient:
    pass


_stub_client.LLMClient = _StubLLMClient
sys.modules.setdefault("problems.prompts.client", _stub_client)

# Likewise stub the gigaevo redis storage and tools.utils dependencies; the
# helper we test does not touch them.
_stub_redis = ModuleType("gigaevo.database.redis_program_storage")


class _StubRedisProgramStorage:
    pass


class _StubRedisProgramStorageConfig:
    pass


_stub_redis.RedisProgramStorage = _StubRedisProgramStorage
_stub_redis.RedisProgramStorageConfig = _StubRedisProgramStorageConfig
sys.modules.setdefault("gigaevo", ModuleType("gigaevo"))
sys.modules.setdefault("gigaevo.database", ModuleType("gigaevo.database"))
sys.modules["gigaevo.database.redis_program_storage"] = _stub_redis

_stub_tools = ModuleType("tools.utils")


class _StubRedisRunConfig:
    pass


_stub_tools.RedisRunConfig = _StubRedisRunConfig
sys.modules.setdefault("tools", ModuleType("tools"))
sys.modules["tools.utils"] = _stub_tools

_utils = _load_module("prompts_utils_under_test", "problems/prompts/utils.py")

CallLog = _types.CallLog
_aggregate_call_logs = _utils._aggregate_call_logs


def test_aggregate_empty_logs_returns_zero_callog() -> None:
    """An empty log list must not IndexError; aggregator returns zero-cost log."""
    aggregated = _aggregate_call_logs([])
    assert aggregated.prompt_tokens == 0
    assert aggregated.completion_tokens == 0
    assert aggregated.cost == 0.0
    assert aggregated.cost_utilization == 0.0


def test_aggregate_single_log_returns_equivalent_values() -> None:
    only = CallLog(
        prompt_tokens=10,
        completion_tokens=20,
        cost=0.5,
        cost_utilization=0.05,
    )
    aggregated = _aggregate_call_logs([only])
    assert aggregated.prompt_tokens == 10
    assert aggregated.completion_tokens == 20
    assert aggregated.cost == 0.5
    assert aggregated.cost_utilization == 0.05


def test_aggregate_multiple_logs_sums_all_fields() -> None:
    """Multi-attempt retries must contribute their token + cost totals.

    Regression: the previous implementation read call_logs[0], silently
    discarding every retry attempt beyond the first.
    """
    logs = [
        CallLog(
            prompt_tokens=10, completion_tokens=20, cost=0.5, cost_utilization=0.05
        ),
        CallLog(
            prompt_tokens=12, completion_tokens=25, cost=0.6, cost_utilization=0.06
        ),
        CallLog(
            prompt_tokens=15, completion_tokens=30, cost=0.7, cost_utilization=0.07
        ),
    ]
    aggregated = _aggregate_call_logs(logs)
    assert aggregated.prompt_tokens == 37
    assert aggregated.completion_tokens == 75
    assert round(aggregated.cost, 9) == 1.8
    assert round(aggregated.cost_utilization, 9) == 0.18
