"""LangGraph agent that emits a schema-constrained structured diff instead of raw genome text."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from gigaevo.evolution.mutation.allowed_changes import AllowedChanges, DiffSchema
from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.exceptions import MutationError
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.agents.mutation import compute_citation_integrity
from gigaevo.llm.models import (
    MultiModelRouter,
    get_last_token_usage,
    get_selected_model,
)
from gigaevo.llm.token_tracking import llm_stage_context
from gigaevo.monitoring.emit import emit as _emit_event
from gigaevo.monitoring.events import LLMCall
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program
from gigaevo.prompts.fetcher import FixedDirPromptFetcher, PromptFetcher


class DiffMutationState(TypedDict, total=False):
    parents: list[Program]
    parents_map: dict[str, str]
    diff_schema: DiffSchema
    messages: list[BaseMessage]
    llm_response: Any
    child_code: str
    diff_payload: dict[str, Any]
    metadata: dict[str, Any]


class DiffMutationAgent(LangGraphAgent):
    StateSchema = DiffMutationState

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        allowed_changes: AllowedChanges,
        task_description: str,
        metrics_context: MetricsContext,
        prompts_dir: str | Path | None = None,
        prompt_fetcher: PromptFetcher | None = None,
    ):
        self._allowed = allowed_changes
        fetcher = prompt_fetcher or FixedDirPromptFetcher(prompts_dir)
        self._user_template = fetcher.fetch("structured_diff", "user").text
        self._system_prompt = fetcher.fetch("structured_diff", "system").text.format(
            task_description=task_description,
            metrics_description=MetricsFormatter(
                metrics_context
            ).format_metrics_description(),
            allowed_changes=allowed_changes.describe(),
        )
        self._llm = llm
        super().__init__(llm)

    def build_prompt(self, state: DiffMutationState) -> DiffMutationState:
        blocks = [self._allowed.render_parents(state["parents_map"])]
        for ns, program in zip(state["parents_map"], state["parents"]):
            context = str(
                program.metadata.get(MUTATION_CONTEXT_METADATA_KEY) or ""
            ).strip()
            if context:
                blocks.append(f"=== Parent {ns} evaluation context ===\n{context}")
        state["messages"] = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=self._user_template.format(parent_blocks="\n\n".join(blocks))
            ),
        ]
        return state

    async def acall_llm(self, state: DiffMutationState) -> DiffMutationState:
        # json_schema is the only structured-output transport the vLLM proxy serves
        # (function_calling 400s without --tool-call-parser; probed 2026-07-02).
        # The {"name","schema"} form skips LangChain's convert_to_openai_function,
        # which KeyErrors on union-root schemas (no top-level "properties").
        schema = state["diff_schema"].json_schema
        structured = self._llm.with_structured_output(
            {"name": str(schema.get("title", "structured_diff")), "schema": schema},
            method="json_schema",
        )
        t0 = time.monotonic()
        error_type: str | None = None
        ok = False
        try:
            with llm_stage_context(self.__class__.__name__):
                state["llm_response"] = await structured.ainvoke(state["messages"])
            ok = True
            if "metadata" not in state:
                state["metadata"] = {}
            model_used = get_selected_model()
            if model_used:
                state["metadata"]["model_used"] = model_used
            return state
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            try:
                usage = get_last_token_usage()
                _emit_event(
                    LLMCall(
                        stage=self.__class__.__name__,
                        endpoint="",
                        model=str(get_selected_model() or "unknown"),
                        attempt=1,
                        ok=ok,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        tokens_in=usage.context if usage else 0,
                        tokens_out=usage.generated if usage else 0,
                        error_type=error_type,
                    )
                )
            except Exception:
                pass

    def parse_response(self, state: DiffMutationState) -> DiffMutationState:
        payload = state["llm_response"]
        try:
            diff = state["diff_schema"].validate(payload)
        except Exception as e:
            raise MutationError(f"diff_schema_error: {e}") from e
        state["child_code"] = self._allowed.apply(diff, state["parents_map"])
        if isinstance(payload, dict):
            state["diff_payload"] = payload
        elif isinstance(diff, BaseModel):
            state["diff_payload"] = diff.model_dump()
        else:
            state["diff_payload"] = {"payload": payload}
        state.setdefault("metadata", {})["citation_integrity"] = (
            self._citation_integrity(state["diff_payload"], state.get("messages", []))
        )
        return state

    @staticmethod
    def _citation_integrity(
        diff_payload: dict[str, Any], messages: list[BaseMessage]
    ) -> dict[str, int]:
        """Ground the diff's letter-parent insight/card citations against the prompt.

        Reads the shared evidence fields off the diff payload; a genome family
        that omits them (no DiffStructuredOutputBase) simply grounds nothing.
        """
        pairs = [
            (str(entry.get("parent", "")), int(entry.get("insight", 0)))
            for entry in diff_payload.get("insight_ids_used", [])
            if isinstance(entry, dict)
        ]
        cards = [c for c in diff_payload.get("card_ids_used", []) if isinstance(c, str)]
        integrity = compute_citation_integrity(pairs, cards, messages)
        logger.info(
            "[DiffMutationAgent] Citation integrity: insights {}/{} grounded, "
            "cards {}/{} grounded",
            integrity["grounded"],
            integrity["cited"],
            integrity["cards_grounded"],
            integrity["cards_cited"],
        )
        return integrity

    async def arun(
        self,
        *,
        parents: list[Program],
        parents_map: dict[str, str],
        diff_schema: DiffSchema,
    ) -> dict[str, Any]:
        state: DiffMutationState = {
            "parents": parents,
            "parents_map": parents_map,
            "diff_schema": diff_schema,
            "metadata": {},
        }
        final = await self.graph.ainvoke(state)
        return {
            "code": final["child_code"],
            "diff": final["diff_payload"],
            "metadata": final.get("metadata", {}),
        }
