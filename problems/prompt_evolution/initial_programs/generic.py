"""Seed prompt: generic mutation system + user prompt with frozen constraints import."""

from gigaevo.prompts.hotpotqa.mutation.frozen import SYSTEM_CONSTRAINTS


def entrypoint() -> dict:
    strategy = """\
MUTATION FOCUS AREAS (general):
1. CODE QUALITY: Improve algorithmic correctness, efficiency, and robustness.
2. PATTERN LEVERAGE: Extend beneficial patterns identified in insights.
3. FAILURE REMOVAL: Eliminate harmful patterns documented in lineage.
4. EXPLORATION: Try fundamentally different approaches when evidence is weak."""

    user = """\
EVOLUTIONARY MUTATION: Adaptive Code Evolution

Transform the program using program insights and historical lineage intelligence with intelligent exploration/exploitation balance.

## INTELLIGENCE INPUTS

**PROGRAM INSIGHTS**: category [tag] (severity): concrete evidence
- Categories: LLM-generated 1-2 word labels
- Tags: beneficial, harmful, neutral, fragile, rigid
- Severity: high, medium, low

**TAG MEANINGS (evolutionary guidance)**:
- **beneficial**: Current pattern is good → PRESERVE/EXTEND this approach
- **harmful**: Current pattern is bad → REMOVE/AVOID this approach
- **fragile**: Current pattern is risky → IMPROVE/ROBUSTIFY this approach
- **rigid**: Current pattern is inflexible → MAKE MORE ADAPTABLE
- **neutral**: Current pattern has no clear impact → IGNORE for evolution

**LINEAGE INSIGHTS**: Historical mutation outcomes and their measured effects
- strategy: imitation/generalization/avoidance/exploration/refinement (past action taken)
- description: causal explanation with quantified impact (≤50 words)
- delta: measured performance impact (relative change)

**EVOLUTIONARY STATISTICS**: Population-level context
- Generation history table: fitness trends across generations (← marks current)
- Is useful to gauge population progress and inform exploration/exploitation balance

## EVOLUTIONARY ARCHETYPE SELECTION

Choose your evolutionary approach based on evidence strength and risk tolerance:

### EXPLOITATION ARCHETYPES (Evidence-Driven Refinement)
1. **Precision Optimization** → Fine-tune proven patterns
2. **Proven Pattern Extension** → Replicate successful strategies
3. **Harmful Pattern Removal** → Eliminate documented failure modes

### EXPLORATION ARCHETYPES (Innovation-Driven Change)
4. **Computational Reinvention** → Novel algorithmic paradigms
5. **Solution Space Exploration** → New problem-solving strategies
6. **Approach Synthesis** → Combine multiple techniques

### HYBRID ARCHETYPES (Balanced Approach)
7. **Guided Innovation** → Preserve proven + introduce targeted improvements
8. **Conservative Exploration** → Explore within safe boundaries

## OUTPUT FORMAT (STRUCTURED JSON)

Your response will be parsed as JSON. Format:

```json
{{
  "archetype": "Selected archetype name",
  "justification": "2-3 sentences: which insights you're acting on, what strategy you're using, and how this mutation is expected to improve fitness.",
  "insights_used": ["insight1 text", "insight2 text", ...],
  "code": "complete Python program"
}}
```

**CRITICAL**:
- Include 1-3 insights in `insights_used` from the provided program insights
- The `code` field must contain ONLY valid Python code
- Focus on code quality — the justification should support your changes, not the other way around

{parent_blocks}"""

    return {"system": SYSTEM_CONSTRAINTS + "\n\n" + strategy, "user": user}
