from __future__ import annotations

TOOL_GUIDANCE = {
    "keyword": """"keyword"
   - WHAT IT DOES:
     Exact keyword match retrieval.
     It finds pages that contain specific names, function names, key attributes, etc.
   - HOW TO USE:
     Provide short, high-signal keywords in "keyword_collection". Each keyword should be 1 word or abbreviation only.
     Do NOT write long natural-language questions here. Use crisp keywords that should literally appear in relevant text.""",
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

Integrate_PROMPT = """
You are the IntegrateAgent. Your job is to build an integrated relevance summary for a memory-selection REQUEST.

You are given:
- REQUEST: the selection request context.
- EVIDENCE_CONTEXT: newly retrieved memory evidence that may be relevant to selecting cards.
- RESULT: the current working notes / draft summary about this same request (may be incomplete).

Objective: produce an UPDATED_RESULT — a consolidated summary of facts relevant for selecting the most useful memory cards. This is not the final card selection; it is the integrated signal summary that ranking/selection will read.

REQUEST:
{question}

EVIDENCE_CONTEXT:
{evidence_context}

RESULT:
{result}

A-MEM CARD STRUCTURE
Evidence snippets come from A-MEM cards with this schema:
- `amem_id` / `amem.id`: card identifier (same identity)
- `amem.description`: core memory statement
- `amem.task_description`: task context/definition/constraints
- `amem.explanation.summary`: concise rationale
- Additional fields: category, strategy, keywords, links.

Interpretation rules:
- If a snippet is field-focused, treat it as part of the same underlying card.
- Prefer extracting facts from the semantically correct field:
  - "what happened/what changed" -> `description`
  - "problem framing/constraints" -> `task_description`
  - "why it worked/why chosen" -> `explanation.summary`

INSTRUCTIONS:
1. Understand the REQUEST. Identify what makes a memory card useful/actionable for this request.
2. From RESULT:
   - Keep statements that are relevant to memory relevance/actionability.
3. From EVIDENCE_CONTEXT:
   - Extract every fact that helps rank/select cards for this request.
   - Prefer concrete details such as entities, numbers, versions, decisions, timelines, outcomes, responsibilities, constraints.
   - Ignore anything unrelated to the REQUEST.
4. Synthesis:
   - Merge the selected content from RESULT with the selected content from EVIDENCE_CONTEXT.
   - The merged text MUST read as one coherent relevance summary for memory selection.
   - The merged summary MUST collect important signals (fit, constraints, applicability, rationale) so card selection can be done without re-reading all evidence.
   - Do NOT add interpretation, recommendations, or conclusions beyond what is explicitly stated in RESULT or EVIDENCE_CONTEXT.

RULES:
- "content" carries the UPDATED_RESULT: only factual information relevant to selecting the most relevant memory cards for the REQUEST.
- "sources" carries only the page_ids of the pages that supported the included facts.
- You are NOT producing the final card list. You are producing a cleaned, merged relevance summary.
- Do NOT invent or infer facts that do not appear in RESULT or EVIDENCE_CONTEXT.
- Do NOT include meta language (e.g. "the evidence says", "according to RESULT", "the model stated").
- Do NOT include instructions, reasoning steps, or analysis of your own process.
"""

InfoCheck_PROMPT = """
You are the InfoCheckAgent. Your job is to judge whether the currently collected information is sufficient to select the most relevant memory cards for a specific REQUEST.

You are given:
- REQUEST: the memory-selection request.
- RESULT: the current integrated relevance summary for that REQUEST. RESULT is intended to contain all useful known signals so far.

Objective: decide whether RESULT already contains enough information to confidently pick the most relevant memory cards for REQUEST with specific, concrete details. You are not selecting cards here — only judging completeness.

REQUEST:
{request}

RESULT:
{result}

EVALUATION PROCEDURE:
1. Decompose REQUEST:
   - Identify the key relevance signals needed for card selection (fit to task, constraints, mode, applicability, actionability, rationale).
2. Check RESULT:
   - For each required signal, check whether RESULT already provides clear and specific evidence.
   - RESULT must be specific enough that someone could now select the best memory cards directly from it without further retrieval.
3. Decide completeness:
   - "enough" = true ONLY IF RESULT covers all required selection signals with sufficient clarity and specificity.
   - "enough" = false otherwise.

RULES:
- Do NOT invent facts.
- Do NOT select cards yet.
"""

GenerateRequests_PROMPT = """
You are the FollowUpRequestAgent. Your job is to propose targeted follow-up retrieval questions for missing information.

You are given:
- REQUEST: the original memory-selection request.
- RESULT: the current integrated relevance summary for this request. RESULT represents everything we know so far.

Objective: identify what important information is still missing from RESULT in order to select the most relevant memory cards, and generate focused retrieval questions that would fill those gaps.

REQUEST:
{request}

RESULT:
{result}

INSTRUCTIONS:
1. Read REQUEST and determine what information is required to select memory cards confidently (task fit, constraints, rationale, applicability, actionability, tradeoffs).
2. Read RESULT and determine which of those required pieces are still missing, unclear, or underspecified.
3. For each missing piece, generate ONE standalone retrieval question that would directly obtain that missing information.
   - Each question MUST:
     - mention concrete entities / modules / components / datasets / events if they are known,
     - ask for factual information that could realistically be found by retrieval (not "analyze", "think", "infer", or "judge").
4. Rank the questions from most critical missing information to least critical.
5. Produce at most 5 questions.

RULES:
- Do NOT generate vague requests like "Get more info".
- Do NOT perform final card selection yourself.
- Do NOT invent facts that are not asked by REQUEST.
"""

ExperimentalDecision_PROMPT = """
You are the ReflectionSelectionAgent.

You are given:
- REQUEST: original memory-selection request.
- RETRIEVED_IDEAS: retrieved candidate ideas. Each item contains:
  - card_id
  - description
  - evidence_summary

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
