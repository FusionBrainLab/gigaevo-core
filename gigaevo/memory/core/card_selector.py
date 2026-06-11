from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory._vendor.GAM_root.gam.schemas.result import ExperimentalDecision
from gigaevo.prompts import MemorySelectorPrompts


def _role_block() -> str:
    try:
        selector_role = MemorySelectorPrompts.system().format()
    except Exception as exc:
        logger.warning(
            "[Memory][CardSelector] selector system.txt load failed: {}", exc
        )
        selector_role = ""
    return f"{selector_role.rstrip()}\n\n" if selector_role else ""


def _build_parent_blocks(parents: list[Any]) -> str:
    blocks: list[str] = []
    for i, parent in enumerate(parents):
        formatted_context = parent.metadata.get(MUTATION_CONTEXT_METADATA_KEY) or ""
        block = f"""=== Parent {i + 1} ===
```python
{parent.code}
```

{formatted_context}
"""
        blocks.append(block)
    return "\n\n".join(blocks)


class LLMCardSelector(BaseModel):
    """Owns the selector-LLM contract: builds the research query that mirrors the
    mutation agent's context, and parses the structured ``final_decision`` back
    into an ordered candidate-id shortlist (fail-to-empty on any bad shape)."""

    model_config = ConfigDict(frozen=True)

    _warned_no_final_decision: bool = PrivateAttr(default=False)

    def _warn_no_final_decision(self, raw_memory: Any) -> None:
        if self._warned_no_final_decision:
            return
        self._warned_no_final_decision = True
        mode = raw_memory.get("pipeline_mode") if isinstance(raw_memory, dict) else None
        logger.warning(
            "[Memory][CardSelector] raw_memory carries no final_decision "
            "(pipeline_mode={}); every selection will be empty. Structured card "
            "selection requires gam_pipeline_mode=experimental.",
            mode or "unknown",
        )

    def _parse_final_decision(self, raw_memory: Any) -> ExperimentalDecision:
        empty = ExperimentalDecision(mode="final", top_ideas=[], additional_queries=[])
        if not isinstance(raw_memory, dict):
            self._warn_no_final_decision(raw_memory)
            return empty
        final = raw_memory.get("final_decision")
        if not isinstance(final, dict):
            self._warn_no_final_decision(raw_memory)
            return empty
        try:
            return ExperimentalDecision.model_validate(final)
        except ValidationError as exc:
            logger.warning(
                "[Memory][CardSelector] final_decision shape invalid: {}", exc
            )
            return empty

    def build_query(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
    ) -> str:
        core = self.build_core_request(
            parents=parents,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            max_cards=max_cards,
        )
        return f"{_role_block()}{core}"

    def build_core_request(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
    ) -> str:
        parent_blocks = _build_parent_blocks(parents)
        return (
            "MUTATION INPUTS\n\n"
            "TASK DESCRIPTION:\n"
            f"{task_description.strip() or '<empty>'}\n\n"
            "AVAILABLE METRICS:\n"
            f"{metrics_description.strip() or '<empty>'}\n\n"
            "MUTATION MODE:\n"
            f"{mutation_mode.strip() or 'rewrite'}\n\n"
            "PARENTS (same parent code + mutation context given to mutation agent):\n"
            f"{parent_blocks}\n\n"
            f"Search your memory database and pick up to {max_cards} card(s) per "
            "the selection criteria above. Emit only their `card_id` values via "
            "the structured-output schema; emit zero entries if no card overlaps "
            "the candidate mechanism."
        )

    def shortlist(self, raw_memory: Any) -> list[str]:
        decision = self._parse_final_decision(raw_memory)
        # Ordered dedup: a repeated id would otherwise get two auction draws
        # (inflated win probability) and render twice in the prompt.
        return list(
            dict.fromkeys(idea.card_id for idea in decision.top_ideas if idea.card_id)
        )
