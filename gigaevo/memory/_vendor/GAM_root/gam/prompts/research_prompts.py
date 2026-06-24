from __future__ import annotations

TOOL_GUIDANCE = {
    "vector": """"vector"
   - WHAT IT DOES:
     Semantic retrieval by meaning over all memory vector stores combined:
     description + task_description + explanation_summary.
     This is good for high-level matching between request context and memory applicability.
   - HOW TO USE:
     Write each query in "vector_queries" as a short natural-language sentence that clearly states what relevance signal you need, using full context and entities from MEMORY and REQUEST.""",
    "vector_description": """"vector_description"
   - WHAT IT DOES:
     Semantic retrieval over only the "description" field of each card.
   - HOW TO USE:
     Write queries in "vector_description_queries" that maximize useful "what happened/what changed" description matches.""",
    "vector_task_description": """"vector_task_description"
   - WHAT IT DOES:
     Semantic retrieval over only the "task_description" field of each card.
   - HOW TO USE:
     Write queries in "vector_task_description_queries" that describe the task framing, constraints, or problem setting whose cards you want.""",
    "vector_explanation_summary": """"vector_explanation_summary"
   - WHAT IT DOES:
     Semantic retrieval over only the "explanation.summary" field of each card.
   - HOW TO USE:
     Write queries in "vector_explanation_summary_queries" based on problems/program insights: failures, weaknesses, instability, and "why" signals.""",
    "page_index": """"page_index"
   - WHAT IT DOES:
     Directly re-reads full pages (by page ID) that are already known to be relevant.
     MEMORY may mention specific page IDs or indices that correspond to important configs, attributes, or names.
   - HOW TO USE:
     Return a list of those integer page indices in "page_index" (e.g. [0, 2, 5]), max 5 pages.
     You MUST NOT invent or guess page indices.""",
}


def render_tool_section(active_tools: list[str]) -> str:
    blocks = [
        f"{i}. {TOOL_GUIDANCE[tool]}"
        for i, tool in enumerate(
            (t for t in active_tools if t in TOOL_GUIDANCE), start=1
        )
    ]
    return "\n\n".join(blocks)


Planning_PROMPT = """
You are the PlanningAgent. Your job is to generate a concrete retrieval plan for selecting the most relevant memory cards for a REQUEST.
You must use the REQUEST and the current MEMORY (which contains abstracts of all messages so far).

REQUEST:
{request}

MEMORY:
{memory}

A-MEM CARD STRUCTURE
Retrieved pages represent memory cards with this structure:

{{
  "amem_id": "<card_id>",
  "amem": {{
    "id": "<card_id>",
    "category": "<string>",
    "description": "<string>",
    "task_description": "<string>",
    "strategy": "<string>",
    "keywords": [<string>, ...],
    "evolution_statistics": {{ ... }},
    "explanation": {{
      "explanations": [<string>, ...],
      "summary": "<string>"
    }},
    "links": [<string>, ...]
  }}
}}

Important mapping notes:
- `amem_id` and `amem.id` refer to the same card identity.
- `description` is the core memory claim/fact.
- `task_description` is the task/problem context and constraints.
- `explanation.summary` is the compact rationale/"why".
- Retrieval snippets may be full card text OR a field-focused snippet (e.g., description-only, task_description-only, explanation.summary-only).

PLANNING PROCEDURE
1. Interpret the REQUEST using the context in MEMORY. Identify what information is needed to select the best memory cards.
2. Decide which of the available retrieval tools are useful for the request. You may assign multiple tools to maximize coverage.
3. Build the final plan:
   - "tools": choose from [{tool_names}].
   - Fill the query fields for the tools you selected, as described in the tool guidance below.

AVAILABLE RETRIEVAL TOOLS
You may select one, several, or all of them in the same plan; combining tools is encouraged when it improves coverage.

{tool_section}

RULES
- Make queries independent of each other, not near-duplicates of one keyword or sentence.
- Be specific. Avoid vague items like "get more details" or "research background".
- Every string in "keyword_collection", "vector_queries", "vector_description_queries",
  "vector_task_description_queries", and "vector_explanation_summary_queries"
  must be directly usable as a retrieval query.
- Only use tools from [{tool_names}]; never invent tools or page indices — if unsure about a page index, leave "page_index" empty.
- You are only planning retrieval here, not selecting final cards.
- Leave the query fields of unselected tools empty.
"""

Decision_PROMPT = """
You are the ReflectionSelectionAgent.

You are given:
- REQUEST: original memory-selection request.
- RETRIEVED_IDEAS: retrieved candidate ideas. Each item contains:
  - card_id
  - description: the candidate mechanism or memory claim
  - evidence_summary: why the candidate helped or mattered
  - optional task_description_summary / task_description: where it worked
  - optional strategy / category / keywords: intent and topical fit signals
  - optional works_with / links: known related card ids
  - optional evidence_source / score: retrieval provenance signals
  - optional efficacy: compact empirical reputation signal when available

Your objective:
Decide ONE of the following:
1) We have enough evidence -> return the final top {max_cards} ideas.
2) We need more evidence -> return additional retrieval queries.

REQUEST:
{request}

RETRIEVED_IDEAS:
{retrieved_ideas}

Decision rules:
- Choose mode = "final" only when evidence is sufficient to confidently choose the top {max_cards}.
- Choose mode = "continue" when evidence is missing/unclear.
- Do not invent card IDs. Use IDs only from RETRIEVED_IDEAS.card_id.
- Keep output factual and grounded in RETRIEVED_IDEAS.
- Rank by task fit first: the selected idea must apply to this REQUEST's problem, constraints, mutation mode, and objective.
- Use efficacy as supporting evidence or a tie-breaker, never as a reason to select a task-mismatched card.
- Treat missing efficacy as neutral, not negative. Treat cautionary/non-positive efficacy as a risk signal.
- Prefer candidates whose task_description/task_description_summary match the REQUEST over candidates with only generic semantic overlap.
- Prefer a small diverse slate of distinct mechanisms over multiple variants of the same mechanism.

Idea quality bar — select only crisp ideas:
- A crisp idea names one concrete mechanism: a specific knob, component, or transformation, the direction of change, and why it should help on this REQUEST.
- Reject vague or umbrella ideas ("improve the model", "tune hyperparameters", "adjust the pipeline") even when they appear relevant.
- Prefer an idea with an exact target and delta over a broader idea with more textual overlap.
- When two ideas describe the same mechanism, keep the sharper one and drop the duplicate.

If mode = "final":
- Return at most {max_cards} items in "top_ideas". Select fewer when fewer ideas are genuinely relevant; never pad with weak ideas just to reach {max_cards}.
- Each top idea references a card only by its card_id. Do NOT rewrite, summarize, or generate card content.
- Leave "additional_queries" empty.

If mode = "continue":
- Return 1-5 concrete "additional_queries".
- Leave "top_ideas" empty.
"""
