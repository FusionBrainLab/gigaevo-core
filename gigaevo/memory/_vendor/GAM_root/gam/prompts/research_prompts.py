from __future__ import annotations

Planning_PROMPT = """
You are the PlanningAgent. Your job is to generate a concrete retrieval plan for selecting the most relevant memory cards for a REQUEST.
You must use the REQUEST and the current MEMORY (which contains abstracts of all messages so far).

REQUEST:
{request}

MEMORY:
{memory}

A-MEM PAGE/CARD STRUCTURE (IMPORTANT)
Retrieved pages represent memory cards with this structure:

{{
  "amem_id": "<card_id>",
  "amem": {{
    "id": "<card_id>",
    "category": "<string>",
    "description": "<string>",
    "task_description": "<string>",
    "strategy": "<string>",
    "last_generation": <int>,
    "programs": [<string>, ...],
    "aliases": [<any>, ...],
    "keywords": [<string>, ...],
    "evolution_statistics": {{ ... }},
    "explanation": {{
      "explanations": [<string>, ...],
      "summary": "<string>"
    }},
    "works_with": [<string>, ...],
    "links": [<string>, ...],
    "usage": {{ ... }}
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
2. Decide which retrieval tools are useful for the request. You may assign multiple tools to maximize coverage:
   - Use "keyword" for exact entities / functions / key attributes.
   - Use "vector" for broad semantic search across all card text fields.
   - Use "vector_description" to search only memory descriptions.
   - Use "vector_task_description" to search only task descriptions.
   - Use "vector_explanation_summary" to search only explanation summaries. Here it would be logical to search for descriptions relevant to program insights.
   - Use "page_index" if MEMORY already points to clearly relevant page indices.
3. Build the final plan:
   - "tools": choose from ["keyword","vector","vector_description","vector_task_description","vector_explanation_summary","page_index"].
   - "keyword_collection": a list of short keyword-style queries you will issue.
   - "vector_queries": semantic queries for broad search over all vector fields.
   - "vector_description_queries": semantic queries for description-only vector search.
   - "vector_explanation_summary_queries": semantic queries for explanation.summary-only vector search.
   - "page_index": a list of integer page indices you plan to read fully.

AVAILABLE RETRIEVAL TOOLS:
All of the following retrieval tools are available to you. You may select one, several, or all of them in the same plan to maximize coverage. Parallel use of multiple tools is allowed and encouraged if it helps memory selection quality.

1. "keyword"
   - WHAT IT DOES:
     Exact keyword match retrieval.
     It finds pages that contain specific names, function names, key attributes, etc.
   - HOW TO USE:
     Provide short, high-signal keywords. Each keyword should be 1 word or abbreviation only.
     Do NOT write long natural-language questions here. Use crisp keywords that should literally appear in relevant text.

2. "vector"
   - WHAT IT DOES:
     Semantic retrieval by meaning over all memory vector stores combined:
     description + task_description + explanation_summary.
     This is good for high-level matching between request context and memory applicability.
   - HOW TO USE:
     Write each query as a short natural-language sentence that clearly states what relevance signal you need, using full context and entities from MEMORY and REQUEST.
     Example style: "How does the DenseRetriever assign GPUs during index building?"

3. "vector_description"
   - WHAT IT DOES:
     Semantic retrieval using only the "description" vector store.
   - HOW TO USE:
     Let the query be chosen by the LLM to maximize useful "what happened/what changed" description matches.

4. "vector_explanation_summary"
   - WHAT IT DOES:
     Semantic retrieval using only the "explanation.summary" vector store.
   - HOW TO USE:
     Base this query on problems/program insights: failures, weaknesses, instability, and "why" signals.

5. "page_index"
   - WHAT IT DOES:
     Directly ask to re-read full pages (by page ID) that are already known to be relevant.
     MEMORY may mention specific page IDs or indices that correspond to important configs, attributes, or names.
     Use this if you already know specific page indices that should be inspected in full.
   - HOW TO USE:
     Return a list of those integer page indices (e.g. [0, 2, 5]), max 5 pages.
     You MUST NOT invent or guess page indices.

RULES
- Avoid simple repetition. Whether it's keywords or sentences for search, make them as independent as possible rather than duplicated.
- Be specific. Avoid vague items like "get more details" or "research background".
- Every string in "keyword_collection", "vector_queries", "vector_description_queries",
  "vector_task_description_queries", and "vector_explanation_summary_queries"
  must be directly usable as a retrieval query.
- You may include multiple tools. Do NOT limit yourself to a single tool if more than one is useful.
- Do NOT invent tools. Only use "keyword", "vector", "vector_description", "vector_task_description", "vector_explanation_summary", "page_index".
- Do NOT invent page indices. If you are not sure about a page index, return [].
- You are only planning retrieval. Do NOT provide final card selection output here.

THINKING STEP
- Before producing the output, think through the procedure and choices inside <think>...</think>.
- Keep the <think> concise but sufficient to validate decisions.
- After </think>, output ONLY the JSON object specified below. The <think> section must NOT be included in the JSON.

OUTPUT JSON SPEC
Return ONE JSON object with EXACTLY these keys:
- "tools": array of strings from ["keyword","vector","vector_description","vector_task_description","vector_explanation_summary","page_index"] (required)
- "keyword_collection": array of strings (required), max 5
- "vector_queries": array of strings (required), max 2
- "vector_description_queries": array of strings (required), max 2
- "vector_task_description_queries": array of strings (required), max 2
- "vector_explanation_summary_queries": array of strings (required), max 2
- "page_index": array of integers (required), max 5.

All keys MUST appear.
After the <think> section, return ONLY the JSON object. Do NOT include any commentary or explanation outside the JSON.
"""

Integrate_PROMPT = """
You are the IntegrateAgent. Your job is to build an integrated relevance summary for a memory-selection REQUEST.

YOU ARE GIVEN:
- REQUEST: the selection request context.
- EVIDENCE_CONTEXT: newly retrieved memory evidence that may be relevant to selecting cards.
- RESULT: the current working notes / draft summary about this same request (may be incomplete).

YOUR OBJECTIVE:
Produce an UPDATED_RESULT that is a consolidated summary of facts that are relevant for selecting the most useful memory cards.
This is NOT the final card selection output. It is an integrated summary of signals that help ranking/selecting cards.

The UPDATED_RESULT must:
1. Keep useful, correct, on-topic relevance signals from RESULT.
2. Add any new, relevant, well-supported facts from EVIDENCE_CONTEXT.
3. Remove anything that is off-topic for the REQUEST.

REQUEST:
{question}

EVIDENCE_CONTEXT:
{evidence_context}

RESULT:
{result}

A-MEM PAGE/CARD STRUCTURE (IMPORTANT)
Evidence snippets come from A-MEM cards with this schema:
- `amem_id` / `amem.id`: card identifier (same identity)
- `amem.description`: core memory statement
- `amem.task_description`: task context/definition/constraints
- `amem.explanation.summary`: concise rationale
- Additional fields: category, strategy, keywords, links, programs, usage, etc.

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
- "content" MUST ONLY include factual information relevant to selecting the most relevant memory cards for the REQUEST.
- You are NOT producing the final card list. You are producing a cleaned, merged relevance summary.
- Do NOT invent or infer facts that do not appear in RESULT or EVIDENCE_CONTEXT.
- Do NOT include meta language (e.g. "the evidence says", "according to RESULT", "the model stated").
- Do NOT include instructions, reasoning steps, or analysis of your own process.
- Do NOT include any keys other than "content" and "sources".
- "sources" should only include the page_ids of the pages that supported the included facts.

THINKING STEP
- Before producing the output, think about selection and synthesis steps inside <think>...</think>.
- Keep the <think> concise but sufficient to ensure correctness and relevance.
- After </think>, output ONLY the JSON object. The <think> section must NOT be included in the JSON.

OUTPUT JSON SPEC:
Return ONE JSON object with EXACTLY:
- "content": string. This is the UPDATED_RESULT used for memory selection; if no useful information exists, provide "".
- "sources": array of strings/objects.

Both keys MUST be present.
After the <think> section, return ONLY the JSON object. Do NOT output Markdown, comments, headings, or explanations outside the JSON.
"""

InfoCheck_PROMPT = """
You are the InfoCheckAgent. Your job is to judge whether the currently collected information is sufficient to select the most relevant memory cards for a specific REQUEST.

YOU ARE GIVEN:
- REQUEST: the memory-selection request.
- RESULT: the current integrated relevance summary for that REQUEST. RESULT is intended to contain all useful known signals so far.

YOUR OBJECTIVE:
Decide whether RESULT already contains enough information to confidently pick the most relevant memory cards for REQUEST with specific, concrete details.
You are NOT selecting cards here. You are only judging completeness.

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

THINKING STEP
- Before producing the output, perform your decomposition and evaluation inside <think>...</think>.
- Keep the <think> concise but ensure it verifies completeness rigorously.
- After </think>, output ONLY the JSON object with the key specified below. The <think> section must NOT be included in the JSON.

OUTPUT REQUIREMENTS:
Return ONE JSON object with EXACTLY this key:
- "enough": Boolean. true if RESULT is sufficient to perform card selection confidently; false otherwise.

RULES:
- Do NOT invent facts.
- Do NOT select cards yet.
- Do NOT include any explanation, reasoning, or extra keys.
- After the <think> section, return ONLY the JSON object.
"""

GenerateRequests_PROMPT = """
You are the FollowUpRequestAgent. Your job is to propose targeted follow-up retrieval questions for missing information.

YOU ARE GIVEN:
- REQUEST: the original memory-selection request.
- RESULT: the current integrated relevance summary for this request. RESULT represents everything we know so far.

YOUR OBJECTIVE:
Identify what important information is still missing from RESULT in order to select the most relevant memory cards, and generate focused retrieval questions that would fill those gaps.

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

THINKING STEP
- Before producing the output, reason about gaps and prioritize inside <think>...</think>.
- Keep the <think> concise but ensure prioritization makes sense.
- After </think>, output ONLY the JSON object specified below. The <think> section must NOT be included in the JSON.

OUTPUT FORMAT:
Return ONE JSON object with EXACTLY this key:
- "new_requests": array of strings (0 to 5 items). Each string is one retrieval question.

RULES:
- Do NOT include any extra keys besides "new_requests".
- After the <think> section, do NOT include explanations, reasoning steps, or Markdown outside the JSON.
- Do NOT generate vague requests like "Get more info".
- Do NOT perform final card selection yourself.
- Do NOT invent facts that are not asked by REQUEST.
After the <think> section, return ONLY the JSON object.
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
1) We have enough evidence -> return final top 3 ideas.
2) We need more evidence -> return additional retrieval queries.

REQUEST:
{request}

RETRIEVED_IDEAS:
{retrieved_ideas}

Decision rules:
- Choose mode = "final" only when evidence is sufficient to confidently choose top 3.
- Choose mode = "continue" when evidence is missing/unclear.
- Do not invent card IDs. Use IDs only from RETRIEVED_IDEAS.card_id.
- Keep output factual and grounded in RETRIEVED_IDEAS.

If mode = "final":
- Return exactly 3 items in "top_ideas" when at least 3 ideas are available; otherwise return as many as available.
- "additional_queries" must be [].
- Each top idea must contain:
  - card_id
- Do NOT rewrite, summarize, or generate card content fields. Only select IDs.

If mode = "continue":
- "top_ideas" must be [].
- Return 1-5 concrete "additional_queries".

THINKING STEP
- Think inside <think>...</think> and then output only JSON.

OUTPUT JSON:
{{
  "mode": "final" | "continue",
  "top_ideas": [
    {{
      "card_id": "string"
    }}
  ],
  "additional_queries": ["string"]
}}

After the <think> section, return ONLY the JSON object.
"""
