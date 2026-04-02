# research_agent.py
"""
ResearchAgent Module

This module defines the ResearchAgent for the GAM (General-Agentic-Memory) framework.

- ResearchAgent is responsible for research tasks, reasoning, and advanced information retrieval.
- It interacts with the MemoryAgent to store and access past knowledge as abstracts (memory is represented as a list[str], without events/tags).
- ResearchAgent uses explicit research functions to process queries and generate insights.
- Prompts within the module are placeholders for future extensions, such as customizable instructions or templates.

The module focuses on providing clear abstraction and extensible interfaces for research-related agent functionalities.
"""

from __future__ import annotations

import json
from typing import Any

from GAM_root.gam.generator import AbsGenerator
from GAM_root.gam.prompts import (
    ExperimentalDecision_PROMPT,
    GenerateRequests_PROMPT,
    InfoCheck_PROMPT,
    Integrate_PROMPT,
    Planning_PROMPT,
)
from GAM_root.gam.schemas import (
    EXPERIMENTAL_DECISION_SCHEMA,
    GENERATE_REQUESTS_SCHEMA,
    INFO_CHECK_SCHEMA,
    INTEGRATE_SCHEMA,
    PLANNING_SCHEMA,
    ExperimentalDecision,
    Hit,
    InMemoryMemoryStore,
    MemoryState,
    MemoryStore,
    PageStore,
    ReflectionDecision,
    ResearchOutput,
    Result,
    Retriever,
    SearchPlan,
    ToolRegistry,
    TopIdea,
)
from loguru import logger

_VECTOR_TOOLS = {
    "vector",
    "vector_description",
    "vector_task_description",
    "vector_explanation_summary",
}
_DEFAULT_TOP_K_BY_TOOL = {
    "keyword": 5,
    "vector": 5,
    "vector_description": 5,
    "vector_task_description": 5,
    "vector_explanation_summary": 5,
    "page_index": 5,
}
_PIPELINE_MODES = {"default", "experimental"}


class ResearchAgent:
    """
    Public API:
      - research(request, memory_state=None) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: ToolRegistry | None = None,
        retrievers: dict[str, Retriever] | None = None,
        generator: AbsGenerator | None = None,  # 必须传入Generator实例
        max_iters: int = 3,
        allowed_tools: list[str] | None = None,
        top_k_by_tool: dict[str, int] | None = None,
        dir_path: str | None = None,  # 新增：文件系统存储路径
        system_prompts: dict[str, str] | None = None,  # 新增：system prompts字典
        pipeline_mode: str = "default",
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        self._allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self._top_k_by_tool = self._normalize_top_k_by_tool(top_k_by_tool)
        self.pipeline_mode = self._normalize_pipeline_mode(pipeline_mode)

        # 初始化 system_prompts，默认值为空字符串
        default_system_prompts = {"planning": "", "integration": "", "reflection": ""}
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            # 合并用户提供的 prompts 和默认值
            self.system_prompts = {**default_system_prompts, **system_prompts}

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                # 调用 retriever 的 build 方法，传递 page_store
                r.build(self.page_store)
                logger.debug(f"Successfully built {name} retriever")
            except Exception as e:
                logger.error(f"Failed to build {name} retriever: {e}")
                pass

    @staticmethod
    def _normalize_pipeline_mode(pipeline_mode: Any) -> str:
        mode = str(pipeline_mode or "default").strip().lower()
        if mode in _PIPELINE_MODES:
            return mode
        return "default"

    @staticmethod
    def _normalize_allowed_tools(allowed_tools: list[str] | None) -> set[str]:
        supported_tools = {"keyword", "page_index", *_VECTOR_TOOLS}
        if not allowed_tools:
            return supported_tools

        normalized = {str(tool).strip() for tool in allowed_tools if str(tool).strip()}
        filtered = {tool for tool in normalized if tool in supported_tools}
        return filtered or supported_tools

    @staticmethod
    def _normalize_top_k_by_tool(
        top_k_by_tool: dict[str, int] | None,
    ) -> dict[str, int]:
        normalized = dict(_DEFAULT_TOP_K_BY_TOOL)
        if not isinstance(top_k_by_tool, dict):
            return normalized

        for tool_name, raw_value in top_k_by_tool.items():
            tool = str(tool_name).strip()
            if tool not in normalized:
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                normalized[tool] = value
        return normalized

    def _tool_top_k(self, tool: str) -> int:
        return self._top_k_by_tool.get(tool, _DEFAULT_TOP_K_BY_TOOL.get(tool, 5))

    @staticmethod
    def _normalize_query_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        cleaned: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text:
                cleaned.append(text)
        return cleaned

    def _vector_queries_for_tool(self, plan: SearchPlan, tool: str) -> list[str]:
        if tool == "vector":
            return self._normalize_query_list(plan.vector_queries)
        if tool == "vector_description":
            return self._normalize_query_list(
                plan.vector_description_queries
            ) or self._normalize_query_list(plan.vector_queries)
        if tool == "vector_task_description":
            return self._normalize_query_list(
                plan.vector_task_description_queries
            ) or self._normalize_query_list(plan.vector_queries)
        if tool == "vector_explanation_summary":
            return self._normalize_query_list(
                plan.vector_explanation_summary_queries
            ) or self._normalize_query_list(plan.vector_queries)
        return []

    def _filter_tools(self, tools: list[str]) -> list[str]:
        return [tool for tool in tools if tool in self._allowed_tools]

    # ---- Public ----
    def research(self, request: str, memory_state: str | None = None) -> ResearchOutput:
        # 在开始研究前，确保检索器索引是最新的
        self._update_retrievers()

        if self.pipeline_mode == "experimental":
            return self._research_experimental(request, memory_state=memory_state)
        return self._research_default(request, memory_state=memory_state)

    def _research_default(
        self, request: str, memory_state: str | None = None
    ) -> ResearchOutput:
        temp = Result()
        iterations: list[dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):
            # Load current memory state dynamically
            current_memory_state = self.memory_store.load()
            plan = self._planning(
                next_request,
                current_memory_state,
                memory_state_override=memory_state,
            )
            plan.tools = self._filter_tools(plan.tools)
            logger.debug("[GAM] Plan:")
            logger.debug(json.dumps(plan.__dict__, ensure_ascii=True, indent=2))

            temp = self._search(plan, temp, request)
            logger.debug("[GAM] Retrieval result:")
            logger.debug(json.dumps(temp.__dict__, ensure_ascii=True, indent=2))

            decision = self._reflection(request, temp)
            logger.debug("[GAM] Reflection:")
            logger.debug(json.dumps(decision.__dict__, ensure_ascii=True, indent=2))

            iterations.append(
                {
                    "step": step,
                    "plan": plan.__dict__,
                    "temp_memory": temp.__dict__,
                    "decision": decision.__dict__,
                }
            )

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request

        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
            "pipeline_mode": self.pipeline_mode,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def _research_experimental(
        self, request: str, memory_state: str | None = None
    ) -> ResearchOutput:
        iterations: list[dict[str, Any]] = []
        next_request = request
        retrieved_ideas_by_id: dict[str, dict[str, Any]] = {}
        final_decision: ExperimentalDecision | None = None

        for step in range(self.max_iters):
            current_memory_state = self.memory_store.load()
            plan = self._planning(
                next_request,
                current_memory_state,
                memory_state_override=memory_state,
            )
            plan.tools = self._filter_tools(plan.tools)
            logger.debug("[GAM][experimental] Plan:")
            logger.debug(json.dumps(plan.__dict__, ensure_ascii=True, indent=2))

            retrieved = self._search_no_integrate(plan, Result(), request)
            iteration_ideas = self._parse_retrieved_ideas(retrieved.content)
            for idea in iteration_ideas:
                card_id = str(idea.get("card_id") or "").strip()
                if not card_id:
                    continue
                if card_id not in retrieved_ideas_by_id:
                    retrieved_ideas_by_id[card_id] = idea

            aggregated_ideas = list(retrieved_ideas_by_id.values())
            decision = self._reflection_experimental(
                request=request,
                retrieved_ideas=aggregated_ideas,
            )
            logger.debug("[GAM][experimental] Reflection decision:")
            logger.debug(json.dumps(decision.model_dump(), ensure_ascii=True, indent=2))

            iterations.append(
                {
                    "step": step,
                    "plan": plan.__dict__,
                    "retrieved": retrieved.__dict__,
                    "retrieved_ideas": iteration_ideas,
                    "decision": decision.model_dump(),
                }
            )

            if decision.mode == "final":
                final_decision = self._ensure_top_ideas(
                    decision,
                    available_card_ids=list(retrieved_ideas_by_id.keys()),
                )
                break

            next_request = self._next_request_from_queries(
                original_request=request,
                queries=decision.additional_queries,
            )

        if final_decision is None:
            fallback_decision = ExperimentalDecision(
                mode="final",
                top_ideas=[],
                additional_queries=[],
            )
            final_decision = self._ensure_top_ideas(
                fallback_decision,
                available_card_ids=list(retrieved_ideas_by_id.keys()),
            )

        final_output = self._format_top_ideas(final_decision.top_ideas)
        raw = {
            "iterations": iterations,
            "pipeline_mode": self.pipeline_mode,
            "final_decision": final_decision.model_dump(),
            "retrieved_ideas": list(retrieved_ideas_by_id.values()),
            "evidence_sources": list(retrieved_ideas_by_id.keys()),
        }
        return ResearchOutput(integrated_memory=final_output, raw_memory=raw)

    @staticmethod
    def _next_request_from_queries(original_request: str, queries: list[str]) -> str:
        cleaned = [q for q in queries if str(q or "").strip()]
        if not cleaned:
            return original_request
        lines = [original_request, "", "Follow-up retrieval focus:"]
        for idx, query in enumerate(cleaned, 1):
            lines.append(f"{idx}. {query}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_text(value: str, max_chars: int = 12000) -> str:
        text = str(value or "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 16] + "\n...[truncated]"

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value or "").strip()
        return [text] if text else []

    def _card_map_by_id(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for page in self.page_store.load():
            meta = getattr(page, "meta", None)
            if not isinstance(meta, dict):
                continue
            card = meta.get("amem")
            card_id = str(meta.get("amem_id") or "").strip()
            if isinstance(card, dict):
                card_id = str(card.get("id") or card_id).strip()
                if card_id and card_id not in out:
                    out[card_id] = card
                continue
            if card_id and card_id not in out:
                out[card_id] = {
                    "id": card_id,
                    "description": str(getattr(page, "content", "") or ""),
                }
        return out

    @staticmethod
    def _extract_explanation_summary(card: dict[str, Any]) -> str:
        explanation = card.get("explanation")
        if isinstance(explanation, dict):
            return str(explanation.get("summary") or "").strip()
        return ""

    def _build_retrieved_ideas(self, hits: list[Hit]) -> list[dict[str, Any]]:
        card_map = self._card_map_by_id()
        ideas: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for hit in hits:
            card_id = str(hit.page_id or "").strip()
            if not card_id or card_id in seen_ids:
                continue
            seen_ids.add(card_id)

            card = card_map.get(card_id, {})
            description = str(card.get("description") or "").strip()
            if not description:
                description = str(hit.snippet or "").strip()

            evidence_summary = self._extract_explanation_summary(card)
            if not evidence_summary:
                evidence_summary = str(hit.snippet or "").strip()

            idea: dict[str, Any] = {
                "card_id": card_id,
                "description": description,
                "evidence_summary": evidence_summary,
            }
            source = str(hit.source or "").strip()
            if source:
                idea["evidence_source"] = source
            score = hit.meta.get("score") if isinstance(hit.meta, dict) else None
            if isinstance(score, (int, float)):
                idea["score"] = float(score)
            ideas.append(idea)

        return ideas

    def _parse_retrieved_ideas(self, payload: Any) -> list[dict[str, Any]]:
        raw = payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return []
            try:
                raw = json.loads(text)
            except Exception:
                return []

        if not isinstance(raw, list):
            return []

        ideas: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id") or "").strip()
            if not card_id or card_id in seen_ids:
                continue
            seen_ids.add(card_id)

            idea: dict[str, Any] = {
                "card_id": card_id,
                "description": str(item.get("description") or "").strip(),
                "evidence_summary": str(item.get("evidence_summary") or "").strip(),
            }
            evidence_source = str(item.get("evidence_source") or "").strip()
            if evidence_source:
                idea["evidence_source"] = evidence_source
            score = item.get("score")
            if isinstance(score, (int, float)):
                idea["score"] = float(score)
            ideas.append(idea)
        return ideas

    def _extract_original_list_field(
        self,
        card: dict[str, Any],
        keys: list[str],
    ) -> list[str]:
        usage = card.get("usage")
        usage_dict = usage if isinstance(usage, dict) else {}
        for key in keys:
            top_level = self._as_string_list(card.get(key))
            if top_level:
                return top_level
            usage_value = self._as_string_list(usage_dict.get(key))
            if usage_value:
                return usage_value
        return []

    def _reflection_experimental(
        self,
        request: str,
        retrieved_ideas: list[dict[str, Any]],
    ) -> ExperimentalDecision:
        normalized_ideas = self._parse_retrieved_ideas(retrieved_ideas)
        card_ids = [str(item.get("card_id") or "").strip() for item in normalized_ideas]
        ideas_payload = self._truncate_text(
            json.dumps(normalized_ideas, ensure_ascii=True, indent=2)
            if normalized_ideas
            else "[]"
        )

        system_prompt = self.system_prompts.get("reflection")
        template_prompt = ExperimentalDecision_PROMPT.format(
            request=request,
            retrieved_ideas=ideas_payload,
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(
                prompt=prompt, schema=EXPERIMENTAL_DECISION_SCHEMA
            )
            data = response.get("json") or json.loads(response["text"])
        except Exception as e:
            logger.error(f"Error in experimental reflection: {e}")
            return ExperimentalDecision(
                mode="continue", top_ideas=[], additional_queries=[]
            )

        raw_mode = str((data or {}).get("mode") or "continue").strip().lower()
        mode = "final" if raw_mode == "final" else "continue"

        additional_queries = self._normalize_query_list(
            (data or {}).get("additional_queries")
        )
        raw_ideas = (data or {}).get("top_ideas")
        top_ideas: list[TopIdea] = []
        seen_ids: set[str] = set()

        if isinstance(raw_ideas, list):
            for item in raw_ideas:
                if isinstance(item, str):
                    item = {"card_id": item}
                if not isinstance(item, dict):
                    continue
                card_id = str(item.get("card_id") or "").strip()
                if not card_id or card_id in seen_ids:
                    continue
                if not card_ids or card_id not in card_ids:
                    continue
                seen_ids.add(card_id)
                top_ideas.append(
                    TopIdea(
                        card_id=card_id,
                    )
                )

        if mode == "final":
            return ExperimentalDecision(
                mode="final", top_ideas=top_ideas[:3], additional_queries=[]
            )
        return ExperimentalDecision(
            mode="continue", top_ideas=[], additional_queries=additional_queries[:5]
        )

    def _ensure_top_ideas(
        self,
        decision: ExperimentalDecision,
        available_card_ids: list[str],
    ) -> ExperimentalDecision:
        if decision.mode != "final":
            return ExperimentalDecision(
                mode="final", top_ideas=[], additional_queries=[]
            )

        top_ideas = list(decision.top_ideas[:3])
        selected_ids = {idea.card_id for idea in top_ideas}
        for card_id in available_card_ids:
            cid = str(card_id or "").strip()
            if not cid or cid in selected_ids:
                continue
            top_ideas.append(
                TopIdea(
                    card_id=cid,
                )
            )
            selected_ids.add(cid)
            if len(top_ideas) >= 3:
                break
        return ExperimentalDecision(
            mode="final", top_ideas=top_ideas[:3], additional_queries=[]
        )

    def _format_top_ideas(self, top_ideas: list[TopIdea]) -> str:
        if not top_ideas:
            return "No final top ideas available from experimental pipeline."

        card_map = self._card_map_by_id()
        lines = ["Top selected memory ideas (experimental):", ""]
        for idx, idea in enumerate(top_ideas, 1):
            card = card_map.get(idea.card_id, {})
            description = str(card.get("description") or "").strip()
            evidence_summary = self._extract_explanation_summary(card)
            lines.append(
                f"{idx}. DESCRIPTION: {description or '(not provided in original card)'}"
            )
            lines.append("WHEN_TO_USE:")
            lines.append(
                f"- {evidence_summary or '(not provided in evidence summary)'}"
            )
            lines.append("")
        return "\n".join(lines).strip()

    def _update_retrievers(self):
        """确保检索器索引是最新的"""
        # 检查是否有新的页面需要更新索引
        current_page_count = len(self.page_store.load())

        # 如果页面数量发生变化，更新所有检索器索引
        if (
            hasattr(self, "_last_page_count")
            and current_page_count != self._last_page_count
        ):
            logger.debug(
                f"检测到页面数量变化 ({self._last_page_count} -> {current_page_count})，更新检索器索引..."
            )
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    logger.debug(f"✅ Updated {name} retriever index")
                except Exception as e:
                    logger.error(f"❌ Failed to update {name} retriever: {e}")

        # 更新页面计数
        self._last_page_count = current_page_count

    # ---- Internal ----
    def _planning(
        self,
        request: str,
        memory_state: MemoryState,
        planning_prompt: str | None = None,
        memory_state_override: str | None = None,
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """

        if memory_state_override is not None:
            memory_context = memory_state_override
        elif not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)

        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        # 调试：打印prompt长度
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # 粗略估算：1 token ≈ 4 字符
        logger.debug(
            f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)"
        )

        try:
            response = self.generator.generate_single(
                prompt=prompt, schema=PLANNING_SCHEMA
            )
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                tools=data.get("tools", []),
                # keyword_collection=[request],
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                vector_description_queries=data.get("vector_description_queries", []),
                vector_task_description_queries=data.get(
                    "vector_task_description_queries", []
                ),
                vector_explanation_summary_queries=data.get(
                    "vector_explanation_summary_queries", []
                ),
                page_index=data.get("page_index", []),
            )
        except Exception as e:
            logger.error(f"Error in planning: {e}")
            return SearchPlan(
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                vector_description_queries=[],
                vector_task_description_queries=[],
                vector_explanation_summary_queries=[],
                page_index=[],
            )

    def _search(
        self,
        plan: SearchPlan,
        result: Result,
        question: str,
        searching_prompt: str | None = None,
    ) -> Result:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        all_hits: list[Hit] = []

        # Execute each planned tool and collect all hits
        for tool in self._filter_tools(plan.tools):
            hits: list[Hit] = []
            logger.debug(f"[GAM] Action selected: {tool}")

            if tool == "keyword":
                if plan.keyword_collection:
                    tool_top_k = self._tool_top_k(tool)
                    logger.debug("[GAM] Keyword queries:")
                    for q in plan.keyword_collection:
                        logger.debug(f"  - {q}")
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword(
                        [combined_keywords], top_k=tool_top_k
                    )
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    if hits:
                        logger.debug("[GAM] Keyword hits:")
                        for i, hit in enumerate(hits, 1):
                            score = hit.meta.get("score") if hit.meta else None
                            score_str = f" score={score}" if score is not None else ""
                            page_id = hit.page_id if hit.page_id else "n/a"
                            logger.debug(
                                f"  {i:02d}. source={hit.source} page_id={page_id}{score_str}"
                            )
                            logger.debug(f"      {hit.snippet}")
                    else:
                        logger.debug("[GAM] Keyword hits: (none)")
                    all_hits.extend(hits)

            elif tool in _VECTOR_TOOLS:
                vector_queries = self._vector_queries_for_tool(plan, tool)
                if vector_queries:
                    tool_top_k = self._tool_top_k(tool)
                    logger.debug(f"[GAM] {tool} queries:")
                    for q in vector_queries:
                        logger.debug(f"  - {q}")
                    # 对每个向量查询都进行独立的搜索，然后在retriever层面聚合得分
                    vector_results = self._search_by_vector_tool(
                        tool_name=tool,
                        query_list=vector_queries,
                        top_k=tool_top_k,
                    )
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    if hits:
                        logger.debug(f"[GAM] {tool} hits:")
                        for i, hit in enumerate(hits, 1):
                            score = hit.meta.get("score") if hit.meta else None
                            score_str = f" score={score}" if score is not None else ""
                            page_id = hit.page_id if hit.page_id else "n/a"
                            logger.debug(
                                f"  {i:02d}. source={hit.source} page_id={page_id}{score_str}"
                            )
                            logger.debug(f"      {hit.snippet}")
                    else:
                        logger.debug(f"[GAM] {tool} hits: (none)")
                    all_hits.extend(hits)

            elif tool == "page_index":
                if plan.page_index:
                    tool_top_k = self._tool_top_k(tool)
                    target_page_index = plan.page_index[:tool_top_k]
                    page_results = self._search_by_page_index(target_page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        # Deduplicate hits by page_id
        if not all_hits:
            return result

        # 按 page_id 去重 hits，避免同一个 page 被多个 tool 检索到时重复添加
        unique_hits: dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: list[Hit] = []  # 没有 page_id 的 hits
        for hit in all_hits:
            if hit.page_id:
                # 如果这个 page_id 还没出现过，或者当前 hit 的得分更高（如果有的话），则更新
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    # 如果已有该 page_id 的 hit，比较得分（如果有的话），保留得分更高的
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = (
                        existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    )
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                # 没有 page_id 的 hits 也保留
                hits_without_id.append(hit)

        # 合并有 page_id 和没有 page_id 的 hits，按得分排序
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(
            all_unique_hits,
            key=lambda h: h.meta.get("score", 0) if h.meta else 0,
            reverse=True,
        )

        # 统一进行一次 integrate
        return self._integrate(sorted_hits, result, question)

    def _search_no_integrate(
        self, plan: SearchPlan, result: Result, question: str
    ) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        all_hits: list[Hit] = []

        # Execute each planned tool and collect hits
        for tool in self._filter_tools(plan.tools):
            hits: list[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    tool_top_k = self._tool_top_k(tool)
                    # 将多个关键词拼接成一个字符串进行搜索
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword(
                        [combined_keywords], top_k=tool_top_k
                    )
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)

            elif tool in _VECTOR_TOOLS:
                vector_queries = self._vector_queries_for_tool(plan, tool)
                if vector_queries:
                    tool_top_k = self._tool_top_k(tool)
                    # 对每个向量查询都进行独立的搜索，然后在retriever层面聚合得分
                    vector_results = self._search_by_vector_tool(
                        tool_name=tool,
                        query_list=vector_queries,
                        top_k=tool_top_k,
                    )
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)

            elif tool == "page_index":
                if plan.page_index:
                    tool_top_k = self._tool_top_k(tool)
                    target_page_index = plan.page_index[:tool_top_k]
                    page_results = self._search_by_page_index(target_page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        # Format all hits as text content without integration
        if not all_hits:
            return result

        # 按 page_id 去重 hits，避免同一个 page 被多个 tool 检索到时重复添加
        unique_hits: dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: list[Hit] = []  # 没有 page_id 的 hits
        for hit in all_hits:
            if hit.page_id:
                # 如果这个 page_id 还没出现过，或者当前 hit 的得分更高（如果有的话），则更新
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    # 如果已有该 page_id 的 hit，比较得分（如果有的话），保留得分更高的
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = (
                        existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    )
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                # 没有 page_id 的 hits 也保留
                hits_without_id.append(hit)

        # 按得分排序（如果有的话）
        # 合并有 page_id 和没有 page_id 的 hits
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(
            all_unique_hits,
            key=lambda h: h.meta.get("score", 0) if h.meta else 0,
            reverse=True,
        )

        ideas = self._build_retrieved_ideas(sorted_hits)
        if not ideas:
            return result
        sources = [
            str(item.get("card_id") or "").strip()
            for item in ideas
            if str(item.get("card_id") or "").strip()
        ]
        formatted_content = json.dumps(ideas, ensure_ascii=True, indent=2)

        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources,
        )

    def _integrate(
        self,
        hits: list[Hit],
        result: Result,
        question: str,
        integration_prompt: str | None = None,
    ) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """

        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")

            if hit.page_id:
                sources.append(hit.page_id)

        evidence_context = "\n".join(evidence_text) if evidence_text else "无搜索结果"

        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(
            question=question, evidence_context=evidence_context, result=result.content
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(
                prompt=prompt, schema=INTEGRATE_SCHEMA
            )
            data = response.get("json") or json.loads(response["text"])

            # 处理 sources：确保是字符串列表（如果LLM返回的是整数，转换为字符串）
            llm_sources = data.get("sources", sources)
            if llm_sources:
                # 将整数或混合类型转换为字符串列表
                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources

            return Result(content=data.get("content", ""), sources=sources)
        except Exception as e:
            logger.error(f"Error in integration: {e}")
            return result

    # ---- search channels ----
    def _search_by_keyword(
        self, query_list: list[str], top_k: int = 3
    ) -> list[list[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                # BM25Retriever 返回 List[List[Hit]]
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                logger.error(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: list[list[Hit]] = []
        for query in query_list:
            query_hits: list[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(
                        Hit(page_id=str(i), snippet=snippet, source="keyword", meta={})
                    )
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(
        self, query_list: list[str], top_k: int = 3
    ) -> list[list[Hit]]:
        return self._search_by_vector_tool("vector", query_list, top_k=top_k)

    def _search_by_vector_tool(
        self,
        tool_name: str,
        query_list: list[str],
        top_k: int = 3,
    ) -> list[list[Hit]]:
        r = self.retrievers.get(tool_name)
        if r is None and tool_name != "vector":
            r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                logger.error(f"Error in vector search ({tool_name}): {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: list[int]) -> list[list[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                # IndexRetriever 现在期望 List[str]，将 page_index 转换为逗号分隔的字符串
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                logger.error(f"Error in page index search: {e}")
                return []

        # fallback: 直接通过 page_store 获取页面
        out: list[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(
                    Hit(
                        page_id=str(idx),
                        snippet=p.content,
                        source="page_index",
                        meta={},
                    )
                )
        return [out]  # 包装成 List[List[Hit]] 格式

    # ---- reflection & summarization ----
    def _reflection(
        self, request: str, result: Result, reflection_prompt: str | None = None
    ) -> ReflectionDecision:
        """
        - "whether information is enough"
        - "if not, generate remaining information as a new request"
        """

        try:
            system_prompt = self.system_prompts.get("reflection")

            # 调试：打印reflection prompt长度
            result_content_chars = len(result.content)
            estimated_result_tokens = result_content_chars // 4
            logger.debug(
                f"[DEBUG] Reflection result.content length: {result_content_chars} chars (~{estimated_result_tokens} tokens)"
            )

            # Step 1: Check for completeness of information
            template_check_prompt = InfoCheck_PROMPT.format(
                request=request, result=result.content
            )
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            logger.debug(
                f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)"
            )

            check_response = self.generator.generate_single(
                prompt=check_prompt, schema=INFO_CHECK_SCHEMA
            )
            check_data = check_response.get("json") or json.loads(
                check_response["text"]
            )

            enough = check_data.get("enough", False)

            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)

            # Step 2: Generate a list of new requests
            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request, result=result.content
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            logger.debug(
                f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)"
            )

            generate_response = self.generator.generate_single(
                prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA
            )
            generate_data = generate_response.get("json") or json.loads(
                generate_response["text"]
            )

            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None

            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)

            return ReflectionDecision(enough=False, new_request=new_request)

        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return ReflectionDecision(enough=False, new_request=None)
