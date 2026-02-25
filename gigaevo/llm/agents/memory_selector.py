"""Memory selector backed by evo_memory_agent_api red search agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from gigaevo.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.programs.program import Program

try:
    from evo_memory_agent_api.runtime_config import (
        deep_get,
        load_settings,
        resolve_local_path,
        resolve_settings_path,
        to_bool,
        to_int,
        to_list,
        to_str,
    )
    from evo_memory_agent_api.shared_memory.memory import AmemGamMemory
except Exception as exc:
    AmemGamMemory = None  # type: ignore[assignment]
    _BACKEND_IMPORT_ERROR: Exception | None = exc
else:
    _BACKEND_IMPORT_ERROR = None


@dataclass
class MemorySelection:
    cards: list[str]
    card_ids: list[str]


class MemorySelectorAgent:
    """Select relevant memory ideas via the evo_memory_agent_api red agent."""

    _RESULT_CHAR_LIMIT = 6000

    def __init__(self) -> None:
        self._search_lock = asyncio.Lock()
        self._backend_error: str | None = None
        self.memory = self._create_memory_backend()

    def _create_memory_backend(self) -> Any | None:
        if AmemGamMemory is None:
            message = (
                "evo_memory_agent_api is unavailable"
                f"{': ' + str(_BACKEND_IMPORT_ERROR) if _BACKEND_IMPORT_ERROR else ''}"
            )
            self._backend_error = message
            logger.warning("[MemorySelectorAgent] {}", message)
            return None

        try:
            repo_root = Path(__file__).resolve().parents[3]
            memory_api_root = repo_root / "evo_memory_agent_api"
            load_dotenv(dotenv_path=repo_root / ".env", override=True)

            settings_path = resolve_settings_path()
            settings = load_settings(settings_path)

            memory_dir = resolve_local_path(
                memory_api_root,
                deep_get(settings, "paths.checkpoint_dir"),
                default_relative="memory_usage_store/api_exp4",
            )
            memory_api_url = os.getenv(
                "MEMORY_API_URL",
                to_str(deep_get(settings, "api.base_url"), default="http://localhost:8000"),
            )
            namespace = os.getenv(
                "MEMORY_NAMESPACE",
                to_str(deep_get(settings, "api.namespace"), default="default"),
            )
            channel = to_str(deep_get(settings, "api.channel"), default="latest")
            author = to_str(deep_get(settings, "api.author"), default=None)
            use_api = to_bool(
                os.getenv("MEMORY_USE_API"),
                default=to_bool(deep_get(settings, "api.use_api"), default=True),
            )
            enable_bm25 = to_bool(deep_get(settings, "gam.enable_bm25"), default=False)
            allowed_gam_tools = [
                str(tool).strip()
                for tool in to_list(deep_get(settings, "gam.allowed_tools"))
                if str(tool).strip()
            ]
            gam_pipeline_mode = to_str(
                os.getenv("MEMORY_GAM_PIPELINE_MODE"),
                default=to_str(deep_get(settings, "gam.pipeline_mode"), default="default"),
            )

            raw_top_k_by_tool = deep_get(settings, "gam.top_k_by_tool", default={})
            gam_top_k_by_tool: dict[str, int] = {}
            if isinstance(raw_top_k_by_tool, dict):
                for tool_name, raw_value in raw_top_k_by_tool.items():
                    tool = str(tool_name).strip()
                    if not tool:
                        continue
                    value = max(1, to_int(raw_value, default=5))
                    gam_top_k_by_tool[tool] = value

            runtime_enable_llm_synthesis = to_bool(
                deep_get(settings, "runtime.enable_llm_synthesis"), default=True
            )
            runtime_enable_memory_evolution = to_bool(
                deep_get(settings, "runtime.should_evolve"), default=True
            )
            runtime_fill_missing_fields = to_bool(
                deep_get(settings, "runtime.fill_missing_fields_with_llm"), default=True
            )
            search_limit = max(1, to_int(deep_get(settings, "runtime.search_limit"), default=5))
            rebuild_interval = max(
                1,
                to_int(deep_get(settings, "runtime.rebuild_interval"), default=10),
            )
            sync_batch_size = max(
                10,
                to_int(deep_get(settings, "runtime.sync_batch_size"), default=100),
            )
            sync_on_init = to_bool(deep_get(settings, "runtime.sync_on_init"), default=True)

            memory = AmemGamMemory(
                checkpoint_path=str(memory_dir),
                base_url=memory_api_url,
                use_api=use_api,
                namespace=namespace,
                author=author,
                channel=channel,
                search_limit=search_limit,
                enable_llm_synthesis=runtime_enable_llm_synthesis,
                enable_memory_evolution=runtime_enable_memory_evolution,
                enable_llm_card_enrichment=runtime_fill_missing_fields,
                rebuild_interval=rebuild_interval,
                enable_bm25=enable_bm25,
                sync_batch_size=sync_batch_size,
                sync_on_init=sync_on_init,
                allowed_gam_tools=allowed_gam_tools,
                gam_top_k_by_tool=gam_top_k_by_tool,
                gam_pipeline_mode=gam_pipeline_mode,
            )
            logger.info(
                "[MemorySelectorAgent] Using evo_memory_agent_api backend "
                "(use_api={}, namespace={}, channel={}, checkpoint={})",
                use_api,
                namespace,
                channel,
                memory_dir,
            )
            return memory
        except Exception as exc:
            self._backend_error = str(exc)
            logger.warning(
                "[MemorySelectorAgent] Failed to initialize red memory backend: {}", exc
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
                "[MemorySelectorAgent] Memory backend unavailable: {}",
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

        result_text = ""
        raw_card_ids: list[str] = []
        try:
            async with self._search_lock:
                result_text, raw_card_ids = await asyncio.to_thread(
                    self._search_with_ids, query
                )
        except Exception as exc:
            logger.warning("[MemorySelectorAgent] Red memory search failed: {}", exc)
            return MemorySelection(cards=[], card_ids=[])

        cards = self._parse_search_result(result_text, max_cards=max_cards)
        parsed_card_ids = self._extract_card_ids_from_text(result_text)
        card_ids = self._merge_card_ids(
            primary=raw_card_ids,
            secondary=parsed_card_ids,
            max_cards=max_cards,
        )

        if cards:
            logger.debug(
                "[MemorySelectorAgent] Selected {} memory idea(s) via red agent (ids={})",
                len(cards),
                card_ids,
            )
        else:
            logger.debug("[MemorySelectorAgent] Red agent returned no relevant memories")
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
            "Search your memory database and return mutation guidance ideas.\n"
            f"Return exactly {max_cards} concise ideas as a numbered list.\n"
            "Each item should be one line and actionable for mutation."
        )

    def _parse_search_result(self, result: str, *, max_cards: int) -> list[str]:
        text = result.strip()
        if not text:
            return []
        if "No relevant memories found" in text:
            return []

        numbered = [m.strip() for m in re.findall(r"(?m)^\d+\.\s.+$", text)]
        if numbered:
            return [self._truncate(item) for item in numbered[:max_cards]]
        bulleted = [m.strip() for m in re.findall(r"(?m)^(?:-|\*)\s+.+$", text)]
        if bulleted:
            return [self._truncate(item) for item in bulleted[:max_cards]]
        return [self._truncate(text)]

    def _search_with_ids(self, query: str) -> tuple[str, list[str]]:
        # Prefer direct GAM output so we can reliably extract selected card ids.
        research_agent = getattr(self.memory, "research_agent", None)
        if research_agent is not None and hasattr(research_agent, "research"):
            try:
                research_result = research_agent.research(query)
                integrated_memory = str(
                    getattr(research_result, "integrated_memory", "") or ""
                )
                raw_memory = getattr(research_result, "raw_memory", None)
                card_ids = self._extract_card_ids_from_raw_memory(raw_memory)
                return integrated_memory, card_ids
            except Exception as exc:
                logger.warning(
                    "[MemorySelectorAgent] Direct GAM research failed, falling back to plain search: {}",
                    exc,
                )

        result_text = str(self.memory.search(query) or "")
        card_ids = self._extract_card_ids_from_text(result_text)
        return result_text, card_ids

    def _extract_card_ids_from_raw_memory(self, raw_memory: Any) -> list[str]:
        if not isinstance(raw_memory, dict):
            return []

        card_ids: list[str] = []

        final_decision = raw_memory.get("final_decision")
        if isinstance(final_decision, dict):
            top_ideas = final_decision.get("top_ideas")
            if isinstance(top_ideas, list):
                for item in top_ideas:
                    if not isinstance(item, dict):
                        continue
                    card_id = str(item.get("card_id") or "").strip()
                    if card_id:
                        card_ids.append(card_id)

        if not card_ids:
            evidence_sources = raw_memory.get("evidence_sources")
            if isinstance(evidence_sources, list):
                for source in evidence_sources:
                    card_id = str(source or "").strip()
                    if card_id:
                        card_ids.append(card_id)

        return self._dedupe_keep_order(card_ids)

    def _extract_card_ids_from_text(self, result: str) -> list[str]:
        text = (result or "").strip()
        if not text:
            return []

        ids: list[str] = []

        # Query: <...>\n\nTop relevant memory cards:\n1. <card_id> [category] ...
        for match in re.finditer(r"(?m)^\d+\.\s+([^\s\[]+)\s+\[[^\]]+\]\s+", text):
            ids.append(match.group(1).strip())

        # JSON-like outputs from GAM internals: {"card_id": "..."}.
        for match in re.finditer(r'card_id"\s*:\s*"([^"]+)"', text):
            ids.append(match.group(1).strip())
        for match in re.finditer(r"card_id\s*[:=]\s*([A-Za-z0-9._:-]+)", text):
            ids.append(match.group(1).strip())

        # Fallback for UUID-like card ids.
        for match in re.finditer(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            text,
        ):
            ids.append(match.group(0).strip())

        return self._dedupe_keep_order(ids)

    def _merge_card_ids(
        self,
        *,
        primary: list[str],
        secondary: list[str],
        max_cards: int,
    ) -> list[str]:
        merged = self._dedupe_keep_order([*primary, *secondary])
        if max_cards <= 0:
            return []
        return merged[:max_cards]

    def _dedupe_keep_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            item = str(value or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _truncate(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) <= self._RESULT_CHAR_LIMIT:
            return normalized
        return normalized[: self._RESULT_CHAR_LIMIT].rstrip() + "\n...[truncated]"

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
