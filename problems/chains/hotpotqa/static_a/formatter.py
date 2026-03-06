"""HotpotQA ASI failure formatter for reflective mutation context."""

import random
from typing import Any

from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.formatter import FormatterStage


class HotpotQAASIFormatter(FormatterStage):
    """Formats ASI-enhanced failure cases from static_a/validate.py into structured
    markdown for the mutation LLM.

    Input: list of dicts with keys: question, gold, predicted, hop1_retrieved,
           hop2_retrieved, n_gold, hop1_missing, hop2_missing (all failures).
    Output: formatted markdown block appended to MutationContextStage.

    Randomly samples 10 failures on each call (non-cacheable) to prevent the
    mutation LLM from overfitting to a fixed set of examples across generations.

    The per-hop retrieval diagnostics surface which gold supporting documents were
    missed by BM25 at each retrieval step, giving the mutation LLM concrete signal
    about whether failures stem from retrieval gaps vs. reasoning gaps.
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

            lines.append(f"### Case {i}")
            lines.append(f"**Question**: {q}")
            lines.append(f"**Expected**: {gold}")
            lines.append(f"**Predicted**: {pred}")

            # Hop 1 retrieval summary
            if n_gold == 0:
                hop1_label = "N/A (no gold docs)"
            elif hop1_ret == n_gold:
                hop1_label = f"✓ all {n_gold} retrieved"
            else:
                missing_str = ", ".join(f'"{t}"' for t in hop1_missing)
                hop1_label = f"MISSING — {missing_str}"
            lines.append(f"**Hop 1 retrieval** ({hop1_ret}/{n_gold} gold): {hop1_label}")

            # Hop 2 retrieval summary
            if n_gold == 0:
                hop2_label = "N/A (no gold docs)"
            elif hop2_ret == n_gold:
                hop2_label = f"✓ all {n_gold} retrieved"
            else:
                missing_str = ", ".join(f'"{t}"' for t in hop2_missing)
                hop2_label = f"MISSING — {missing_str}"
            lines.append(f"**Hop 2 retrieval** ({hop2_ret}/{n_gold} gold): {hop2_label}")
            lines.append("")

        lines.append(
            "_Use these failure cases to identify systematic weaknesses. "
            "MISSING retrieval entries indicate the BM25 query at that hop failed "
            "to surface key supporting documents — consider improving the query "
            "generation prompt for that step. Cases where retrieval succeeded but "
            "prediction was still wrong suggest reasoning or answer extraction issues._"
        )
        return "\n".join(lines)
