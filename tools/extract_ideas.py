import asyncio
import csv
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm

# Ensure we can import from utils if running from tools directory
try:
    from utils import RedisRunConfig, fetch_evolution_dataframe
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from utils import RedisRunConfig, fetch_evolution_dataframe

SYSTEM_PROMPT = """
You are an expert at distilling code into precise algorithmic specifications.

# OBJECTIVE
Extract a **concise specification** of WHAT the code does. A developer should be able to implement equivalent code from this.

# KEY PRINCIPLES

1. **Consolidate related concepts** - don't break into mechanical steps
   - BAD: "Split PRNG key" → "Generate uniform values" → "Mirror array" → "Add to base"
   - GOOD: "Symmetric perturbation: uniform[-0.1, 0.1] on first half, mirrored"

2. **Describe WHAT, not WHY** - no rationale or explanations
   - BAD: "Use exp(u) to ensure positivity"
   - GOOD: "f = exp(u)"

3. **Skip obvious mechanics** - assume developer competence
   - SKIP: "Initialize optimizer", "JIT-compile", "return result", "convert to NumPy"
   - KEEP: Non-obvious algorithms, specific formulas, key hyperparameters

4. **One idea = one concept** - not one line of code
   - BAD: 14 items describing every operation
   - GOOD: 5-8 items capturing the essential concepts

# WHAT TO INCLUDE
- Mathematical formulas with specific constants
- Initialization strategies
- Key hyperparameters (merged, not listed separately)
- Non-obvious algorithmic choices

# WHAT TO EXCLUDE
- Standard optimizer/library usage
- Obvious control flow
- Boilerplate (PRNG splits, type conversions, returns)
- Purpose/rationale

# OUTPUT FORMAT
Numbered list. Each idea = one concept, one line, with specific values.
Aim for conciseness - fewer ideas that capture more.
"""

USER_PROMPT_TEMPLATE = """
Extract a concise specification from this code:

```python
{}
```

Specification (consolidate related concepts):
"""


async def main():
    # 1. Fetch Data
    print("Fetching dataframe from Redis...")
    config = RedisRunConfig(redis_db=8, redis_prefix="heilbron")
    df = await fetch_evolution_dataframe(config)
    print(f"Loaded dataframe with {len(df)} rows.")

    # 2. Setup Model
    model = ChatOpenAI(
        model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Ensure program_id exists, otherwise generate indices
    if "program_id" in df.columns:
        program_ids = list(df["program_id"])
    else:
        print("Warning: 'program_id' column not found. Using index as ID.")
        program_ids = list(df.index)

    codes = list(df["code"])

    # 3. Process with Progress Tracking
    print("Starting LLM processing...")

    output_file = "extracted_ideas_heilbron.csv"

    # Open CSV file for writing
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["program_id", "ideas"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Use tqdm for progress bar
        for pid, code in tqdm(
            zip(program_ids, codes), total=len(codes), desc="Extracting Ideas"
        ):
            if not code or not isinstance(code, str):
                writer.writerow({"program_id": pid, "ideas": ""})
                continue

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=USER_PROMPT_TEMPLATE.format(code)),
            ]

            try:
                # Invoke model
                res = await model.ainvoke(messages)
                content = res.content

                # Write to CSV immediately
                writer.writerow({"program_id": pid, "ideas": content})
                csvfile.flush()

            except Exception as e:
                print(f"Error processing snippet {pid}: {e}")
                writer.writerow({"program_id": pid, "ideas": f"ERROR: {str(e)}"})

    print(f"Processing complete. Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
