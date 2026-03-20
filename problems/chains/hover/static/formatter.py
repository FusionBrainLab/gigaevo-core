"""HoVer failure feedback formatter for mutation context."""

import random
from typing import Any

from loguru import logger

from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.formatter import FormatterStage


class HoVerFeedbackFormatter(FormatterStage):
    """Formats per-hop retrieval failure cases from validate.py into structured
    markdown for the mutation LLM.

    Input: list of dicts with keys: claim, gold_titles, n_gold, hop{1,2,3}_found,
           hop{1,2,3}_missing, all_found, all_missing, hop{2,3}_query.
    Output: formatted markdown block appended to MutationContextStage.

    Randomly samples 10 failures on each call (non-cacheable) to prevent the
    mutation LLM from overfitting to a fixed set of examples across generations.
    """

    cache_handler = NO_CACHE

    def format_value(self, data: Any) -> str:
        if not data:
            logger.debug("[HoVerFeedbackFormatter] No failure data to format")
            return ""

        failures = data if isinstance(data, list) else []
        logger.info(
            "[HoVerFeedbackFormatter] Formatting {} failure(s) (sampling up to 10)",
            len(failures),
        )
        sample = random.sample(failures, min(10, len(failures)))
        lines = [
            f"## Failure Analysis"
            f" ({len(sample)} of {len(failures)} failure(s) randomly sampled)\n"
        ]
        for i, f in enumerate(sample, 1):
            claim = f.get("claim", "?")
            n_gold = f.get("n_gold", 0)
            all_missing = f.get("all_missing", [])

            lines.append(f"### Case {i}")
            lines.append(f"**Claim**: {claim}")
            lines.append(
                f"**Gold documents** ({n_gold}): "
                + ", ".join(f'"{t}"' for t in f.get("gold_titles", []))
            )
            lines.append(
                f"**Still missing after all 3 hops** ({len(all_missing)}): "
                + (", ".join(f'"{t}"' for t in all_missing) or "none")
            )

            for hop in range(1, 4):
                found = f.get(f"hop{hop}_found", [])
                missing = f.get(f"hop{hop}_missing", [])
                if not missing:
                    label = f"all {n_gold} found"
                else:
                    missing_str = ", ".join(f'"{t}"' for t in missing)
                    label = f"MISSING {missing_str}"
                lines.append(
                    f"**Hop {hop} retrieval** "
                    f"({len(found)}/{n_gold} gold): {label}"
                )

            hop2_q = f.get("hop2_query", "")
            hop3_q = f.get("hop3_query", "")
            if hop2_q:
                lines.append(f"**Hop 2 query**: {hop2_q}")
            if hop3_q:
                lines.append(f"**Hop 3 query**: {hop3_q}")
            lines.append("")

        lines.append(
            "_Use these failure cases to identify systematic retrieval weaknesses. "
            "MISSING entries indicate the BM25 query at that hop failed to surface "
            "key supporting documents. Consider improving the query generation prompt "
            "for the hop that consistently misses documents, or adjusting the "
            "summarization prompts to better highlight bridging entities._"
        )
        return "\n".join(lines)
