"""HotpotQA failure formatter for reflective mutation context."""

from typing import Any

from gigaevo.programs.stages.formatter import FormatterStage


class HotpotQAFailureFormatter(FormatterStage):
    """Formats per-sample failure cases from validate.py into a structured
    markdown block for the mutation LLM.

    Input: list of dicts with keys: question, gold, predicted (up to 10 items).
    Output: formatted string appended to MutationContextStage as failure analysis.
    """

    def format_value(self, data: Any) -> str:
        if not data:
            return ""

        failures = data if isinstance(data, list) else []
        lines = [f"## Failure Analysis ({len(failures)} sample(s) where prediction was wrong)\n"]
        for i, f in enumerate(failures[:10], 1):
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
