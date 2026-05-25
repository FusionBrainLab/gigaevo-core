"""Memory selector backed by gigaevo.memory red search agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory._vendor.GAM_root.gam.schemas.result import ExperimentalDecision
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import GamConfig, MemoryConfig
from gigaevo.memory.write_pipeline_config import load_config as _load_memory_config
from gigaevo.programs.program import Program
from gigaevo.prompts import MemorySelectorPrompts


class MemorySelection(BaseModel):
    """Result of memory card selection for mutation guidance."""

    model_config = ConfigDict(frozen=True)

    cards: list[str]
    card_ids: list[str]


class MemorySelectorAgent:
    """Select relevant memory ideas via the gigaevo.memory red agent."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | None = None,
        namespace: str | None = None,
        use_api: bool | None = None,
    ) -> None:
        self._checkpoint_dir_override = checkpoint_dir
        self._namespace_override = namespace
        self._use_api_override = use_api
        self._search_lock = asyncio.Lock()
        self._backend_error: str | None = None
        self.memory = self._create_memory_backend()

    @staticmethod
    def _resolve_memory_backend_class(use_api: bool) -> type[Any]:
        """Resolve the platform backend class (API mode only)."""
        if use_api:
            from gigaevo.memory_platform import AmemGamMemory as platform_backend

            return platform_backend

        from gigaevo.memory.shared_memory.memory import AmemGamMemory as legacy_backend

        return legacy_backend

    def _create_memory_backend(self) -> Any | None:
        try:
            repo_root = Path(__file__).resolve().parents[3]
            load_dotenv(dotenv_path=repo_root / ".env", override=True)

            cfg = _load_memory_config()

            # Apply instance-level overrides
            memory_dir = (
                Path(self._checkpoint_dir_override)
                if self._checkpoint_dir_override
                else cfg.memory_dir
            )
            namespace = self._namespace_override or cfg.namespace
            use_api = (
                self._use_api_override
                if self._use_api_override is not None
                else cfg.use_api
            )

            if use_api:
                memory_backend_cls = self._resolve_memory_backend_class(use_api)
                memory = memory_backend_cls(
                    checkpoint_path=str(memory_dir),
                    base_url=cfg.memory_api_url,
                    use_api=use_api,
                    namespace=namespace,
                    author=cfg.author,
                    channel=cfg.channel,
                    search_limit=cfg.search_limit,
                    enable_llm_synthesis=cfg.enable_llm_synthesis,
                    enable_memory_evolution=cfg.should_evolve,
                    enable_llm_card_enrichment=cfg.fill_missing_fields_with_llm,
                    rebuild_interval=cfg.rebuild_interval,
                    enable_bm25=cfg.enable_bm25,
                    sync_batch_size=cfg.sync_batch_size,
                    sync_on_init=cfg.sync_on_init,
                    allowed_gam_tools=cfg.allowed_gam_tools,
                    gam_top_k_by_tool=cfg.gam_top_k_by_tool,
                    gam_pipeline_mode=cfg.gam_pipeline_mode,
                )
            else:
                mem_config = MemoryConfig(
                    checkpoint_path=memory_dir,
                    search_limit=cfg.search_limit,
                    rebuild_interval=cfg.rebuild_interval,
                    enable_llm_synthesis=cfg.enable_llm_synthesis,
                    enable_memory_evolution=cfg.should_evolve,
                    enable_llm_card_enrichment=cfg.fill_missing_fields_with_llm,
                    gam=GamConfig(
                        enable_bm25=cfg.enable_bm25,
                        allowed_tools=cfg.allowed_gam_tools,
                        top_k_by_tool=cfg.gam_top_k_by_tool,
                        pipeline_mode=cfg.gam_pipeline_mode or "default",
                    ),
                )
                memory = AmemGamMemory(config=mem_config)

            logger.info(
                "[Memory][SelectorAgent] Using memory backend "
                "(class={}, use_api={}, namespace={}, channel={}, checkpoint={})",
                type(memory).__module__,
                use_api,
                namespace,
                cfg.channel,
                memory_dir,
            )
            return memory
        except Exception as exc:
            self._backend_error = str(exc)
            logger.warning(
                "[Memory][SelectorAgent] Failed to initialize red memory backend: {}",
                exc,
            )
            return None

    async def arun(
        self,
        *,
        input: list[Program],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        memory_text: str,
        max_cards: int = 1,
    ) -> list[str]:
        selection = await self.select(
            input=input,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            memory_text=memory_text,
            max_cards=max_cards,
        )
        return selection.cards

    async def select(
        self,
        *,
        input: list[Program],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        memory_text: str,
        max_cards: int = 1,
    ) -> MemorySelection:
        if max_cards <= 0:
            return MemorySelection(cards=[], card_ids=[])
        if self.memory is None:
            logger.warning(
                "[Memory][SelectorAgent] Memory backend unavailable: {}",
                self._backend_error or "unknown error",
            )
            return MemorySelection(cards=[], card_ids=[])

        query = self._build_request(
            parents=input,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            max_cards=max_cards,
        )
        _ = memory_text  # legacy input kept for API compatibility; red search ignores it

        try:
            async with self._search_lock:
                research_result = await asyncio.to_thread(self.memory.research, query)
        except Exception as exc:
            logger.warning("[Memory][SelectorAgent] Red memory search failed: {}", exc)
            return MemorySelection(cards=[], card_ids=[])

        decision = self._parse_final_decision(research_result.raw_memory)
        card_ids = [
            idea.card_id for idea in decision.top_ideas[:max_cards] if idea.card_id
        ]
        cards = self._fetch_card_texts(card_ids)

        if cards:
            logger.debug(
                "[Memory][SelectorAgent] Selected {} memory idea(s) via red agent (ids={})",
                len(cards),
                card_ids,
            )
        else:
            logger.debug(
                "[Memory][SelectorAgent] Red agent returned no relevant memories"
            )
        return MemorySelection(cards=cards, card_ids=card_ids)

    def _build_request(
        self,
        *,
        parents: list[Program],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        max_cards: int,
    ) -> str:
        parent_blocks = self._build_parent_blocks(parents)
        try:
            selector_role = MemorySelectorPrompts.system().format()
        except Exception as exc:
            logger.warning(
                "[Memory][SelectorAgent] selector system.txt load failed: {}", exc
            )
            selector_role = ""
        role_block = f"{selector_role.rstrip()}\n\n" if selector_role else ""
        return (
            f"{role_block}"
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

    @staticmethod
    def _parse_final_decision(raw_memory: Any) -> ExperimentalDecision:
        empty = ExperimentalDecision(mode="final", top_ideas=[], additional_queries=[])
        if not isinstance(raw_memory, dict):
            return empty
        final = raw_memory.get("final_decision")
        if not isinstance(final, dict):
            return empty
        try:
            return ExperimentalDecision.model_validate(final)
        except ValidationError as exc:
            logger.warning(
                "[Memory][SelectorAgent] final_decision shape invalid: {}", exc
            )
            return empty

    def _fetch_card_texts(self, card_ids: list[str]) -> list[str]:
        if self.memory is None:
            return []
        texts: list[str] = []
        for cid in card_ids:
            try:
                card = self.memory.get_card(cid)
            except Exception as exc:
                logger.warning(
                    "[Memory][SelectorAgent] get_card({}) failed: {}", cid, exc
                )
                continue
            content = self._render_card(card)
            if content:
                texts.append(content)
        return texts

    @staticmethod
    def _render_card(card: Any) -> str:
        if card is None:
            return ""
        if isinstance(card, dict):
            description = str(card.get("description") or "")
        else:
            description = str(getattr(card, "description", "") or "")
        return description.strip()

    def _build_parent_blocks(self, parents: list[Program]) -> str:
        """Build formatted parent blocks to mirror mutation agent context."""
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
