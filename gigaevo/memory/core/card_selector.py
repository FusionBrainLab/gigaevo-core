from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory._vendor.GAM_root.gam.schemas.result import Decision
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


def _build_parent_blocks(
    parents: list[Any], parent_contexts: list[str] | None = None
) -> str:
    blocks: list[str] = []
    for i, parent in enumerate(parents):
        if parent_contexts is not None:
            formatted_context = parent_contexts[i] if i < len(parent_contexts) else ""
        else:
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
        logger.warning(
            "[Memory][CardSelector] raw_memory carries no final_decision; "
            "every selection will be empty."
        )

    def _parse_final_decision(self, raw_memory: Any) -> Decision:
        empty = Decision(mode="final", top_ideas=[], additional_queries=[])
        if not isinstance(raw_memory, dict):
            self._warn_no_final_decision(raw_memory)
            return empty
        final = raw_memory.get("final_decision")
        if not isinstance(final, dict):
            self._warn_no_final_decision(raw_memory)
            return empty
        try:
            return Decision.model_validate(final)
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
        parent_contexts: list[str] | None = None,
    ) -> str:
        core = self.build_core_request(
            parents=parents,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            max_cards=max_cards,
            parent_contexts=parent_contexts,
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
        parent_contexts: list[str] | None = None,
    ) -> str:
        parent_blocks = _build_parent_blocks(parents, parent_contexts)
        return (
            "MUTATION INPUTS\n\n"
            "TASK DESCRIPTION:\n"
            f"{task_description.strip() or '<empty>'}\n\n"
            "AVAILABLE METRICS:\n"
            f"{metrics_description.strip() or '<empty>'}\n\n"
            "MUTATION MODE:\n"
            f"{mutation_mode.strip() or 'rewrite'}\n\n"
            "PARENTS (parent code + this-pass lineage card + live evolutionary snapshot):\n"
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
