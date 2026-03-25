import os
import json
from pathlib import Path

from agentic_memory.memory_system import AgenticMemorySystem
from agent.agent_class import LLMService


# -----------------------------
# Helpers (safe printing + diff)
# -----------------------------
def _safe_get(obj, name, default=None):
    return getattr(obj, name, default)


def pretty_print_memory(mem, title=None):
    if title:
        print(f"\n{'='*10} {title} {'='*10}")

    mem_id = _safe_get(mem, "id", None) or _safe_get(mem, "memory_id", None)
    print(f"ID:        {mem_id}")
    print(f"Content:   {_safe_get(mem, 'content', '')}")
    print(f"Category:  {_safe_get(mem, 'category', None)}")
    print(f"Timestamp: {_safe_get(mem, 'timestamp', None)}")
    print(f"Tags:      {_safe_get(mem, 'tags', [])}")
    print(f"Keywords:  {_safe_get(mem, 'keywords', [])}")
    print(f"Context:   {_safe_get(mem, 'context', '')}")

    links = (
        _safe_get(mem, "links", None)
        or _safe_get(mem, "linked_memories", None)
        or _safe_get(mem, "linked_ids", None)
        or _safe_get(mem, "relations", None)
        or []
    )
    print(f"Links:     {links}")


def summarize_diff(before, after, label="Memory evolution check"):
    print(f"\n--- {label} ---")
    fields = ["content", "tags", "keywords", "context", "category", "timestamp", "links"]
    for f in fields:
        b = _safe_get(before, f, None)
        a = _safe_get(after, f, None)
        if b != a:
            print(f"* {f} changed:")
            print(f"  - before: {b}")
            print(f"  - after : {a}")


def _memory_to_dict(mem):
    return {
        "id": _safe_get(mem, "id", None) or _safe_get(mem, "memory_id", None),
        "content": _safe_get(mem, "content", ""),
        "category": _safe_get(mem, "category", None),
        "timestamp": _safe_get(mem, "timestamp", None),
        "tags": _safe_get(mem, "tags", []) or [],
        "keywords": _safe_get(mem, "keywords", []) or [],
        "context": _safe_get(mem, "context", ""),
        "links": (
            _safe_get(mem, "links", None)
            or _safe_get(mem, "linked_memories", None)
            or _safe_get(mem, "linked_ids", None)
            or _safe_get(mem, "relations", None)
            or []
        ),
        "retrieval_count": _safe_get(mem, "retrieval_count", None),
        "last_accessed": _safe_get(mem, "last_accessed", None),
        "evolution_history": _safe_get(mem, "evolution_history", []) or [],
    }


def export_memories_jsonl(memory_system, memory_ids, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for mid in memory_ids:
            mem = memory_system.read(mid)
            record = _memory_to_dict(mem)
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


# -----------------------------
# Ingest from list[str]
# -----------------------------
def add_memories_from_list(memory_system, memories, category="heilbron"):
    """
    Takes a list of strings and adds them to the memory system.
    Everything except category is blank/default (tags=[], timestamp=None).
    Returns a list of memory IDs (same order as input).
    """
    ids = []
    for i, text in enumerate(memories, start=1):
        text = (text or "").strip()
        if not text:
            # Skip empty items so you don't store blank memories
            continue

        # Analyze memory content to capture keywords/context/tags before saving
        analysis = memory_system.analyze_content(text)
        tags = analysis.get("tags") or []
        keywords = analysis.get("keywords") or []
        context = analysis.get("context") or "General"

        mid = memory_system.add_note(
            content=text,
            tags=tags,
            category=category,    # required by you
            timestamp=None,       # blank
            keywords=keywords,
            context=context,
        )
        ids.append(mid)
    return ids


# -----------------------------
# Demo script
# -----------------------------
def main():
    # Configure OpenRouter-backed LLMService
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or "sk-or-v1-REDACTED"
    if not openrouter_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY env var. "
            "Set it first, e.g.: export OPENROUTER_API_KEY='...'"
        )

    llm_service = LLMService(
        service="openrouter_openai",
        model_name="openai/gpt-4.1-mini",
        api_key=openrouter_key,
        temperature=0,
        max_tokens=0,
    )

    # Initialize the memory system 🚀
    memory_system = AgenticMemorySystem(
        model_name="all-MiniLM-L6-v2",  # Embedding model for ChromaDB
        llm_backend="custom",           # Use external LLMService
        llm_service=llm_service,
    )

    print("\n==============================")
    print("A-MEM Demo: ingest from list[str]")
    print("==============================\n")

    # -------------------------------------------------------
    # 1) Your memories: list of strings only
    # -------------------------------------------------------
    memories = [
  "Farthest Point Sampling replaced Sobol; created superior initial spread, foundational for +0.02519 fitness gain.",
  "Added stagnation restart (500 iters); escaped local minima, critical for the +0.02519 metric improvement.",
  "Symmetric point distribution (5 left + axis + 5 right) eliminated parent's clustering; directly enabled +0.02231 area gain.",
  "20k-iteration simulated annealing targeting min area; explains full +0.02231 metric gain through systematic bottleneck optimization.",
  "Removed parent's rigid row-based grid (5->1 points) causing density clustering; removal eliminated small triangles, contributing to +0.02070 area gain.",
  "Introduced simulated annealing with temperature-controlled moves; explored configurations beyond parent's static grid, enabling +0.02070 min_area gain.",
  "Generalized perturbations to target min-triangle vertices (lines 40-50); direct refinement of critical regions increased min_area by 0.02070.",
  "Reduced boundary repulsion to 1e-6, preventing edge clustering; directly enabled +0.01855 area gain by eliminating degenerate boundary triangles.",
  "Enforced reflection symmetry via left/right mirroring; eliminated asymmetric clusters, directly contributing +0.01796 min_area gain.",
  "Introduced simulated annealing (T=0.01) to escape local minima; accepted worse moves, enabling 0.01796 improvement.",
  "Targeted perturbations to points in smallest triangle (found by triple loop); focused optimization on critical regions for +0.01796 gain.",
  "Reduced boundary repulsion threshold from 0.02 to 1e-6; enabled optimal boundary placement, explaining +0.01224 fitness gain.",
  "Removed hex grid generation; eliminated structured clustering artifacts causing small triangles, primary driver of +0.00916 gain.",
  "Replaced hexagonal lattice with Halton sequence (5000 points); eliminated structured clustering; enabled better space coverage for +0.00902 gain.",
  "Changed boundary buffer to max(0.005, 0.02*best_fitness); balanced exploration near boundaries; reduced constraint violations by 63%.",
  "Triangle area caching with periodic clearing reduced computation time by ~35%, enabling more effective exploration within iteration limit.",
  "Vectorized triangle calculation with precise tolerance (0.002*fitness); reduced false collinearity detection; improved gradient accuracy by 4x."
]

    # -------------------------------------------------------
    # 2) Add/index them (category=heilbron, everything else blank)
    # -------------------------------------------------------
    print("1) Adding memories from list...\n")
    memory_ids = add_memories_from_list(memory_system, memories, category="heilbron")

    # Read back and print each memory so you can see what A-MEM generated
    for idx, mid in enumerate(memory_ids, start=1):
        mem = memory_system.read(mid)
        pretty_print_memory(mem, title=f"Added memory #{idx}")

    # -------------------------------------------------------
    # 2.5) Export A-mem memories for GAM reuse
    # -------------------------------------------------------
    export_path = Path(__file__).resolve().parent / "amem_exports" / "amem_memories.jsonl"
    export_memories_jsonl(memory_system, memory_ids, export_path)
    print(f"\nExported A-mem memories to: {export_path}")

    # -------------------------------------------------------
    # 3) Retrieval example
    # -------------------------------------------------------
    print("\n2) Retrieval example (search_agentic) ...\n")
    q = "memory1"
    print(f">>> QUERY: {q!r} (k=5)")
    results = memory_system.search_agentic(q, k=5)

    for i, r in enumerate(results, start=1):
        rid = r.get("id") if isinstance(r, dict) else _safe_get(r, "id", None)
        rcontent = r.get("content") if isinstance(r, dict) else _safe_get(r, "content", "")
        print(f"  [{i}] id={rid} | {rcontent}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
