"""Token-usage accounting survives the chain runner's per-call ``client.copy()``."""

from problems.chains.client import CallLog, LLMClient
from problems.chains.usage import ZERO_USAGE, LogAggregatingLLMClient, usage_totals


def _log(prompt, completion, cost):
    return CallLog(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost=cost,
        cost_utilization=0.0,
    )


def test_base_client_copy_isolates_logs():
    """Guard: the stock client keeps isolating logs; aggregation is opt-in."""
    parent = LLMClient(model="Qwen/Qwen3-8B")
    parent.copy()._call_logs.append(_log(1, 1, 0.0))
    assert parent.call_logs == []


def test_copies_share_one_log_list():
    parent = LogAggregatingLLMClient(model="Qwen/Qwen3-8B")
    parent.copy()._call_logs.append(_log(10, 5, 0.0))
    parent.copy()._call_logs.append(_log(20, 7, 0.0))
    assert len(parent.call_logs) == 2


def test_copy_of_a_copy_still_shares_logs():
    """The runner copies copies; aggregation must survive arbitrary copy depth."""
    parent = LogAggregatingLLMClient(model="Qwen/Qwen3-8B")
    grandchild = parent.copy().copy()
    assert isinstance(grandchild, LogAggregatingLLMClient)
    grandchild._call_logs.append(_log(10, 5, 0.0))
    assert len(parent.call_logs) == 1


def test_clear_logs_keeps_copies_attached():
    """clear_logs must empty the shared list in place; the base reassignment
    would silently detach every existing copy from the aggregate."""
    parent = LogAggregatingLLMClient(model="Qwen/Qwen3-8B")
    child = parent.copy()
    child._call_logs.append(_log(10, 5, 0.0))
    parent.clear_logs()
    assert parent.call_logs == []
    child._call_logs.append(_log(20, 7, 0.0))
    assert len(parent.call_logs) == 1


def test_usage_totals_sums_across_calls():
    parent = LogAggregatingLLMClient(model="Qwen/Qwen3-8B")
    parent.copy()._call_logs.append(_log(10, 5, 0.001))
    parent.copy()._call_logs.append(_log(20, 7, 0.002))
    totals = usage_totals(parent)
    assert totals["prompt_tokens"] == 30
    assert totals["completion_tokens"] == 12
    assert totals["total_tokens"] == 42
    assert totals["n_llm_calls"] == 2
    assert abs(totals["llm_cost"] - 0.003) < 1e-9


def test_usage_totals_empty_matches_zero_usage():
    parent = LogAggregatingLLMClient(model="Qwen/Qwen3-8B")
    assert usage_totals(parent) == ZERO_USAGE
