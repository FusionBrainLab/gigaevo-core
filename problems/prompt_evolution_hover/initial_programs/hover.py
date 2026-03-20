"""Seed prompt: HoVer-specific mutation strategy with 3-hop retrieval guidance."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of prompt-chain programs for NLP tasks.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

MUTATION FOCUS AREAS (in priority order):
1. SECOND-HOP QUERY (step 3): Does step 2 identify the key entities from hop 1?
   Does step 3 emit bare search terms targeting the second gold document?
2. THIRD-HOP QUERY (step 6): Does step 5 consolidate evidence from hops 1+2?
   Does step 6 generate a precise query for the third, most distant gold document?
3. EVIDENCE CONSOLIDATION (steps 2, 5): Do these steps extract named entities and
   factual claims, not generic summaries? Specific facts improve downstream queries.
4. GLOBAL CONTEXT (system_prompt): Is it short and role-focused? Compression here
   improves ALL steps simultaneously."""

    user = """\
EVOLUTIONARY MUTATION: HoVer 3-Hop Retrieval Chain Optimization

Mutate the parent program using insights and lineage evidence.

## KEY PRINCIPLES
- Steps 3 and 6 outputs are used VERBATIM as BM25 queries — bare search terms only
- The third hop (step 6) is hardest — the third gold document is most distant
- system_prompt is shared across all LLM steps — keep under 20 words
- Shorter instructions with explicit constraints beat verbose prose
- Focus on retrieval coverage: finding ALL 3 gold documents, not just 1 or 2

## ARCHETYPE SELECTION
Choose based on evidence:
- "Precision Optimization" → tighten query format (e.g., step 3/6 to pure search terms)
- "Proven Pattern Extension" → replicate working query constraint to other hop
- "Harmful Pattern Removal" → remove verbose rules with no fitness gain
- "Computational Reinvention" → fundamentally different instruction strategy
- "Solution Space Exploration" → new entity extraction or evidence linking approach
- "Approach Synthesis" → combine clean query format with focused evidence synthesis

## OUTPUT FORMAT (JSON)

```json
{{
  "archetype": "Selected archetype name",
  "justification": "2-3 sentences linking insights to changes.",
  "insights_used": ["insight1 text", "insight2 text"],
  "code": "complete Python program"
}}
```

**CRITICAL**: The `code` field must contain ONLY valid Python code.

{parent_blocks}"""

    return {"system": system, "user": user}
