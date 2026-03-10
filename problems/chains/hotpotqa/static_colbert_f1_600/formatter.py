"""HotpotQA richer failure formatter for ColBERT + F1-600 variant.

Extends HotpotQAASIFormatter by showing full passage text (Title | text) for
missing gold documents instead of just titles. This matches the feedback depth
that GEPA provides to its mutation LLM, giving the model actionable content to
improve retrieval queries.
"""

import random
from typing import Any

from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.formatter import FormatterStage


class HotpotQAColBERTFormatter(FormatterStage):
    """Formats ColBERT failure cases with full passage text for missing gold docs.

    Input: list of dicts with keys: question, gold, predicted, hop1_retrieved,
           hop2_retrieved, n_gold, hop1_missing (titles), hop2_missing (titles),
           hop1_missing_passages (full 'Title | text' strings),
           hop2_missing_passages (full 'Title | text' strings).

    Output: formatted markdown block for the mutation LLM.

    Randomly samples 10 failures per call (non-cacheable) to prevent overfitting
    to a fixed set of examples across generations.

    The full passage text for missing gold docs gives the mutation LLM concrete
    content to understand what the query should have retrieved — enabling it to
    craft better query generation prompts for that retrieval step.
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
            n_gold = f.get("n_gold", 0)
            hop1_ret = f.get("hop1_retrieved", 0)
            hop2_ret = f.get("hop2_retrieved", 0)
            hop1_missing = f.get("hop1_missing", [])
            hop2_missing = f.get("hop2_missing", [])
            hop1_passages = f.get("hop1_missing_passages", hop1_missing)
            hop2_passages = f.get("hop2_missing_passages", hop2_missing)

            lines.append(f"### Case {i}")
            lines.append(f"**Question**: {q}")
            lines.append(f"**Expected**: {gold}")
            lines.append(f"**Predicted**: {pred}")

            # Hop 1 retrieval summary with full passage text for missing docs
            if n_gold == 0:
                hop1_label = "N/A (no gold docs)"
                lines.append(f"**Hop 1 retrieval** ({hop1_ret}/{n_gold} gold): {hop1_label}")
            elif hop1_ret == n_gold:
                lines.append(
                    f"**Hop 1 retrieval** ({hop1_ret}/{n_gold} gold): ✓ all {n_gold} retrieved"
                )
            else:
                lines.append(
                    f"**Hop 1 retrieval** ({hop1_ret}/{n_gold} gold): MISSING — "
                    f"the following gold document(s) were not retrieved:"
                )
                for passage in hop1_passages:
                    lines.append(f"  - {passage}")

            # Hop 2 retrieval summary with full passage text for missing docs
            if n_gold == 0:
                hop2_label = "N/A (no gold docs)"
                lines.append(f"**Hop 2 retrieval** ({hop2_ret}/{n_gold} gold): {hop2_label}")
            elif hop2_ret == n_gold:
                lines.append(
                    f"**Hop 2 retrieval** ({hop2_ret}/{n_gold} gold): ✓ all {n_gold} retrieved"
                )
            else:
                lines.append(
                    f"**Hop 2 retrieval** ({hop2_ret}/{n_gold} gold): MISSING — "
                    f"the following gold document(s) were not retrieved:"
                )
                for passage in hop2_passages:
                    lines.append(f"  - {passage}")

            lines.append("")

        lines.append(
            "_Use these failure cases to identify systematic weaknesses. "
            "MISSING retrieval entries show the full content of gold documents "
            "that the retriever failed to surface — use this to understand what "
            "the query at that hop should have contained to retrieve these documents. "
            "Cases where retrieval succeeded but prediction was still wrong suggest "
            "reasoning or answer extraction issues._"
        )
        return "\n".join(lines)
