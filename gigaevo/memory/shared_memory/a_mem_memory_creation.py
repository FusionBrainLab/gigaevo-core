import json
import sys

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_AGENT_ROOT = _THIS_DIR.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from A_mem.agentic_memory.memory_system import AgenticMemorySystem
from openai_inference import OpenAIInferenceService

from gigaevo.memory import config


# -----------------------------
# Helpers (safe printing + diff)
# -----------------------------
def _safe_get(obj, name, default=None):
    return getattr(obj, name, default)


def pretty_print_memory(mem, title=None):
    if title:
        print(f"\n{'=' * 10} {title} {'=' * 10}")

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
    fields = [
        "content",
        "tags",
        "keywords",
        "context",
        "category",
        "timestamp",
        "links",
    ]
    for f in fields:
        b = _safe_get(before, f, None)
        a = _safe_get(after, f, None)
        if b != a:
            print(f"* {f} changed:")
            print(f"  - before: {b}")
            print(f"  - after : {a}")


def _to_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_memory_card(
    card: dict[str, Any] | None = None,
    fallback_id: str | None = None,
) -> dict[str, Any]:
    card = dict(card or {})
    explanation = card.get("explanation")
    if not isinstance(explanation, dict):
        explanation = {}

    normalized = {
        "id": str(card.get("id") or fallback_id or ""),
        "category": str(card.get("category") or "general"),
        "description": str(card.get("description") or card.get("content") or ""),
        "task_description": str(
            card.get("task_description") or card.get("context") or ""
        ),
        "task_description_summary": str(
            card.get("task_description_summary") or card.get("context_summary") or ""
        ),
        "strategy": str(card.get("strategy") or ""),
        "last_generation": _to_int(card.get("last_generation"), default=0),
        "programs": _to_list(card.get("programs")),
        "aliases": _to_list(card.get("aliases")),
        "keywords": _to_list(card.get("keywords")),
        "evolution_statistics": card.get("evolution_statistics")
        if isinstance(card.get("evolution_statistics"), dict)
        else {},
        "explanation": {
            "explanations": _to_list(explanation.get("explanations")),
            "summary": str(explanation.get("summary") or ""),
        },
        "works_with": _to_list(card.get("works_with")),
        "links": _to_list(card.get("links")),
        "usage": card.get("usage") if isinstance(card.get("usage"), dict) else {},
    }

    return normalized


def _memory_to_dict(
    mem, base_card: dict[str, Any] | None = None, memory_id: str | None = None
):
    mem_id = (
        _safe_get(mem, "id", None) or _safe_get(mem, "memory_id", None) or memory_id
    )
    card = normalize_memory_card(base_card, fallback_id=mem_id)
    if mem is None:
        return card

    card["id"] = str(mem_id or card["id"])
    card["category"] = str(
        card.get("category") or _safe_get(mem, "category", None) or "general"
    )
    card["description"] = str(card.get("description") or _safe_get(mem, "content", ""))
    card["task_description"] = str(
        card.get("task_description") or _safe_get(mem, "context", "")
    )
    card["strategy"] = str(card.get("strategy") or _safe_get(mem, "strategy", ""))
    card["keywords"] = _to_list(_safe_get(mem, "keywords", []) or [])

    if not card.get("links"):
        card["links"] = (
            _safe_get(mem, "links", None)
            or _safe_get(mem, "linked_memories", None)
            or _safe_get(mem, "linked_ids", None)
            or _safe_get(mem, "relations", None)
            or []
        )
    card["links"] = _to_list(card["links"])

    return card


def export_memories_jsonl(
    memory_system,
    memory_ids,
    out_path,
    card_overrides: dict[str, dict[str, Any]] | None = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card_overrides = card_overrides or {}
    unique_ids = list(dict.fromkeys(memory_ids))
    with out_path.open("w", encoding="utf-8") as f:
        for mid in unique_ids:
            mem = memory_system.read(mid)
            base_card = card_overrides.get(mid)
            if mem is None and base_card is None:
                continue
            record = _memory_to_dict(mem, base_card=base_card, memory_id=mid)
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
            category=category,  # required by you
            timestamp=None,  # blank
            keywords=keywords,
            context=context,
        )
        ids.append(mid)
    return ids


# -----------------------------
# Demo script
# -----------------------------
def main():
    # Configure OpenAI-compatible inference service (OpenRouter supported via base_url).
    api_key = config.OPENAI_API_KEY
    if not api_key and config.LLM_BASE_URL:
        api_key = "EMPTY"

    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY/OPENROUTER_API_KEY env var. "
            "Set one first, e.g.: export OPENAI_API_KEY='...'"
        )

    base_url = config.LLM_BASE_URL

    llm_service = OpenAIInferenceService(
        model_name=config.OPENROUTER_MODEL_NAME,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=0,
        reasoning=config.OPENROUTER_REASONING,
    )

    # Initialize the memory system 🚀
    memory_system = AgenticMemorySystem(
        model_name=config.AMEM_EMBEDDING_MODEL_NAME,  # Embedding model for ChromaDB
        llm_backend="custom",  # Use external inference service
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
        "Vectorized triangle calculation with precise tolerance (0.002*fitness); reduced false collinearity detection; improved gradient accuracy by 4x.",
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
    export_path = (
        Path(__file__).resolve().parent / "amem_exports" / "amem_memories.jsonl"
    )
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
        rcontent = (
            r.get("content") if isinstance(r, dict) else _safe_get(r, "content", "")
        )
        print(f"  [{i}] id={rid} | {rcontent}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
