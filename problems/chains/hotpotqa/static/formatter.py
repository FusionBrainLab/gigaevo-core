"""HotpotQA failure formatter for reflective mutation context."""

import random
from typing import Any

from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.formatter import FormatterStage


class HotpotQAFailureFormatter(FormatterStage):
    """Formats per-sample failure cases from validate.py into a structured
    markdown block for the mutation LLM.

    Input: list of dicts with keys: question, gold, predicted (all failures).
    Output: formatted string appended to MutationContextStage as failure analysis.

    Randomly samples 10 failures on each call (non-cacheable) to prevent the
    mutation LLM from overfitting to a fixed set of examples across generations.
    """

    cache_handler = NO_CACHE  # re-sample on every DAG run

    def format_value(self, data: Any) -> str:
        if not data:
            return ""

        failures = data if isinstance(data, list) else []
        sample = random.sample(failures, min(10, len(failures)))
        lines = [
            f"## Failure Analysis"
            f" ({len(sample)} of {len(failures)} failure(s) randomly sampled)\n"
        ]
        for i, f in enumerate(sample, 1):
            q = f.get("question", "?")
            gold = f.get("gold", "?")
            pred = f.get("predicted") or "(extraction failure — no answer found)"
            lines.append(f"### Case {i}")
            lines.append(f"**Question**: {q}")
            lines.append(f"**Expected**: {gold}")
            lines.append(f"**Predicted**: {pred}")
            lines.append("")

        lines.append(
            "_Use these failure cases to identify systematic weaknesses in the "
            "current prompt (e.g. missing disambiguation, wrong step instructions, "
            "answer format issues) and propose targeted improvements._"
        )
        return "\n".join(lines)
