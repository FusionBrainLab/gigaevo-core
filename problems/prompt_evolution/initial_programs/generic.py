"""Seed prompt: generic mutation strategy with broad archetype coverage."""


def entrypoint() -> dict:
    system = """\
You are an expert in evolutionary optimization of prompt-chain programs for NLP tasks.

OBJECTIVE:
{task_description}

AVAILABLE METRICS:
{metrics_description}

---

MUTATION STRATEGY:
1. CODE QUALITY: Improve algorithmic correctness, efficiency, and robustness.
2. PATTERN LEVERAGE: Extend beneficial patterns identified in insights.
3. FAILURE REMOVAL: Eliminate harmful patterns documented in lineage.
4. EXPLORATION: Try fundamentally different approaches when evidence is weak."""

    user = """\
EVOLUTIONARY MUTATION: Adaptive Code Evolution

Transform the program using program insights and historical lineage intelligence.

## INTELLIGENCE INPUTS

**PROGRAM INSIGHTS**: [category][tag](severity) — concrete evidence about current program
- Tags: beneficial (PRESERVE), harmful (REMOVE), fragile (ROBUSTIFY), neutral (IGNORE)

**LINEAGE INSIGHTS**: Historical mutation outcomes
- strategy, description (≤50 words), measured delta

**FAILURE ANALYSIS**: Wrong predictions with retrieval coverage — use to distinguish
retrieval failures (fix step 3 query) from reasoning failures (fix steps 5-6).

## ARCHETYPE SELECTION

### EXPLOITATION (strong evidence)
1. **Precision Optimization** → Fine-tune proven patterns
2. **Proven Pattern Extension** → Replicate successful strategies
3. **Harmful Pattern Removal** → Eliminate documented failure modes

### EXPLORATION (weak/negative evidence)
4. **Computational Reinvention** → Novel algorithmic paradigms
5. **Solution Space Exploration** → New problem-solving strategies
6. **Approach Synthesis** → Combine multiple techniques

### HYBRID (mixed evidence)
7. **Guided Innovation** → Preserve proven + introduce targeted improvements
8. **Conservative Exploration** → Explore within safe boundaries

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
