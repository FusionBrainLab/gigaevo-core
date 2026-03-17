"""Seed prompt: HotpotQA-specific mutation strategy with step-level guidance."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of prompt-chain programs for NLP tasks.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

MUTATION FOCUS AREAS (in priority order):
1. RETRIEVAL PATH (steps 2, 3): Does step 2 name the bridge entity? Does step 3 emit
   only bare search terms? Improvements here help ALL multi-hop questions.
2. ANSWER PATH (steps 5, 6): Does step 5 produce a unified evidence set? Does step 6
   emit a clean "Answer: X"? Improvements here prevent extraction failures.
3. GLOBAL CONTEXT (system_prompt): Is it short and role-focused? Compression here
   improves ALL steps simultaneously.
4. REASONING DEPTH (reasoning_questions, example_reasoning): Add only when evidence
   shows the LLM is failing to apply the right reasoning pattern. Remove when bloated."""

    user = """\
EVOLUTIONARY MUTATION: HotpotQA Chain Optimization

Mutate the parent program using failure analysis and lineage evidence.

## KEY PRINCIPLES
- Step 3 output is used VERBATIM as BM25 query — keep it to bare search terms
- system_prompt is shared across all LLM steps — keep under 20 words
- Shorter instructions with explicit constraints beat verbose prose
- Use failure analysis to distinguish retrieval vs reasoning failures

## ARCHETYPE SELECTION
Choose based on evidence:
- "Precision Optimization" → tighten a format constraint (e.g., step 3 to pure search terms)
- "Proven Pattern Extension" → replicate working constraint to another step
- "Harmful Pattern Removal" → remove verbose rules with no fitness gain
- "Computational Reinvention" → fundamentally different instruction strategy for a step
- "Solution Space Exploration" → new reasoning scaffold
- "Approach Synthesis" → combine clean format with focused reasoning

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
