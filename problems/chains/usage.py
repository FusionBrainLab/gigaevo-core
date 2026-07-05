"""Token-usage accounting for chain LLM clients.

The chain runners call ``client.copy()`` for every concurrent LLM call so each
gets an isolated ``_call_logs`` list (thread safety). Usage therefore
accumulates in throwaway copies and the parent client's ``call_logs`` stays
empty. ``LogAggregatingLLMClient`` shares ONE ``_call_logs`` list across all
copies so per-call usage survives to the validator, which reduces it with
``usage_totals`` into metrics that persist on the program.
"""

from __future__ import annotations

from problems.chains.client import LLMClient

ZERO_USAGE: dict[str, float] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "n_llm_calls": 0,
    "llm_cost": 0.0,
}


class LogAggregatingLLMClient(LLMClient):
    """LLM client whose per-call token logs survive the runner's ``copy()`` fan-out."""

    def copy(self) -> LLMClient:
        clone = super().copy()
        clone._call_logs = self._call_logs
        return clone

    def clear_logs(self) -> None:
        # in place — the base reassignment would detach existing copies
        self._call_logs.clear()


def usage_totals(client: LLMClient) -> dict[str, float]:
    """Sum per-call token usage collected on ``client`` into metric fields."""
    logs = client.call_logs
    prompt = sum(log.prompt_tokens for log in logs)
    completion = sum(log.completion_tokens for log in logs)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "n_llm_calls": len(logs),
        "llm_cost": sum(log.cost for log in logs),
    }
