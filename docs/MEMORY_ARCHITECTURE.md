# Memory System Architecture

The shared memory system (`gigaevo/memory/shared_memory/`) stores and retrieves evolutionary knowledge cards — ideas, strategies, and program metadata discovered during MAP-Elites runs. It supports local-only operation, API-backed persistence, agentic vector search (GAM), and LLM-based card deduplication.

## Module Overview

```
AmemGamMemory (orchestrator, ~736 lines)
  ├── config: MemoryConfig           — Pydantic config (replaces 18 kwargs)
  ├── card_store: CardStore          — Card dict + entity mappings + disk persistence
  ├── note_sync: NoteSync | None     — A-MEM / Chroma bridge
  ├── api_sync: ApiSync | None       — Remote concept API synchronization
  ├── gam: GamSearch | None          — GAM ResearchAgent lifecycle
  ├── dedup: CardDedup               — Vector scoring + LLM decision + merge
  └── llm_service / generator        — Injected LLM dependencies
```

## Files

| File | Lines | Responsibility |
|------|-------|----------------|
| `memory.py` | ~736 | Orchestrator: wires collaborators, coordinates save/search/rebuild |
| `memory_config.py` | ~110 | `MemoryConfig`, `GamConfig`, `ApiConfig` (Pydantic BaseModel) |
| `card_store.py` | ~130 | Card dict, entity mappings, JSON index persistence |
| `note_sync.py` | ~180 | Bridges cards to A-MEM vector store (Chroma) |
| `api_sync.py` | ~215 | Paginated fetch, full sync, search via concept API |
| `gam_search.py` | ~100 | GAM ResearchAgent build/invalidate lifecycle |
| `card_dedup.py` | ~400 | Vector scoring, LLM dedup decision, card merge computation |
| `agentic_runtime.py` | ~60 | `AgenticRuntime` dataclass + `load_agentic_runtime()` factory |

Supporting files (unchanged):

| File | Responsibility |
|------|----------------|
| `models.py` | Pydantic card models (`MemoryCard`, `ProgramCard`, `Explanation`) |
| `card_conversion.py` | Pure functions: normalize, convert, format cards |
| `card_update_dedup.py` | Pure functions: query building, weighted scoring, LLM parsing |
| `concept_api.py` | HTTP client for the remote Memory API |
| `utils.py` | Small helpers (`truncate_text`, `looks_like_uuid`) |

## Design Principles

1. **Single-responsibility collaborators.** Each class owns one concern. `CardStore` owns persistence, `NoteSync` owns the Chroma bridge, `CardDedup` owns scoring — none of them trigger rebuilds or make orchestration decisions.

2. **Orchestrator coordinates.** `AmemGamMemory` decides *when* to rebuild, *when* to persist, *when* to sync. Collaborators return results (booleans, tuples, lists) and the orchestrator acts on them.

3. **Pydantic config.** `MemoryConfig` replaces 18 scattered constructor kwargs. Sub-configs (`GamConfig`, `ApiConfig`) group related settings. `from_legacy_kwargs()` provides backward compatibility during migration.

4. **Dependency injection.** `AgenticRuntime` bundles the four agentic class dependencies (AgenticMemorySystem, MemoryNote, ResearchAgent, AMemGenerator). Tests pass `FakeAgenticRuntime` at construction; production auto-detects via `load_agentic_runtime()`.

5. **No circular imports.** Collaborator modules import from `models`, `card_conversion`, `card_update_dedup`, `concept_api` — never from `memory.py`.

## Data Flow

### save_card

```
save_card(card_dict)
  → normalize_memory_card(card_dict)
  → if existing card: _save_card_core → card_store + note_sync + API
  → if program card: _save_card_core (skip dedup)
  → if dedup enabled:
      dedup.score_candidates(card) → vector similarity scores
      dedup.format_for_llm(scored) → truncated payloads
      dedup.decide_action(card, candidates) → {action, updates}
        → "discard": return existing card_id
        → "update": dedup.compute_merges → _save_card_core for each merge
        → "add": fall through
  → _save_card_core(card)
      → card_store.ensure_id(card)
      → API save (if enabled)
      → card_store.cards[id] = card
      → note_sync.upsert_agentic(card)
      → dedup.invalidate_retrievers()
      → periodic rebuild()
```

### search

```
search(query, memory_state)
  → if API: _sync_from_api → api_sync.sync → rebuild if changed
  → if GAM agent: research_agent.research(query)
  → if API: _search_via_api → api_sync.search → synthesize results
  → fallback: _search_local_cards (keyword matching)
```

### rebuild

```
rebuild()
  → card_store.serialize_all() → card_store.persist()
  → note_sync.export_jsonl() (JSONL for GAM)
  → gam.build() → sets gam.agent
  → dedup.invalidate_retrievers()
```

## External Consumers

`MemorySelectorAgent` (`gigaevo/llm/agents/memory_selector.py`) uses `MemoryConfig` for local-only mode:

```python
from gigaevo.memory.shared_memory.memory_config import MemoryConfig, GamConfig

config = MemoryConfig(
    checkpoint_path=memory_dir,
    search_limit=5,
    gam=GamConfig(enable_bm25=True, allowed_tools=["vector_description"]),
)
memory = AmemGamMemory(config=config)
```

For API mode (platform backend), legacy kwargs are still used since the platform class has its own constructor.

## Testing

Tests use `inject_fakes_into_memory(mem)` from `tests/fakes/agentic_memory.py` to replace agentic classes with in-memory fakes (no Chroma, no embeddings, no LLM calls). After injection, tests also create `GamSearch` manually since the constructor can't create it without real dependencies:

```python
mem = AmemGamMemory(checkpoint_path=..., use_api=False, ...)
fake_sys = inject_fakes_into_memory(mem)
mem.generator = FakeAMemGenerator({"llm_service": MagicMock()})

# GamSearch not created during __init__ (deps unavailable before fakes)
if mem.gam is None:
    mem.gam = GamSearch(
        research_agent_cls=mem._ResearchAgentCls,
        generator=mem.generator,
        card_store=mem.card_store, ...
    )
```

Test helpers then patch `mem.gam.build` and `mem.dedup.build_retrievers` with fake implementations that use `FakeRetriever` (Jaccard similarity) instead of real Chroma vector search.

Tests access card data directly via `mem.card_store.cards`, `mem.card_store.entity_by_card_id`, etc. — no backward-compatibility properties on the orchestrator.

## Key Classes

### MemoryConfig

Pydantic `BaseModel` with `ConfigDict(extra="forbid")`. Accepts either structured config or `from_legacy_kwargs()` for backward compatibility:

```python
# Structured (preferred)
config = MemoryConfig(
    checkpoint_path=Path("./mem"),
    api=ApiConfig(base_url="http://...", namespace="ns"),
    gam=GamConfig(allowed_tools=["vector_description"]),
    dedup=CardUpdateDedupConfig(enabled=True),
)
mem = AmemGamMemory(config=config)

# Legacy kwargs (still supported)
mem = AmemGamMemory(
    checkpoint_path="./mem",
    use_api=True,
    base_url="http://...",
    ...
)
```

### CardStore

Owns `cards: dict[str, AnyCard]`, entity mappings, and `api_index.json` persistence. Thread-safe for single-threaded MAP-Elites (shared by reference to all collaborators).

### CardDedup

Stateless scoring + LLM decision engine. Returns results (scored candidates, LLM decisions, merged cards) — never writes to `CardStore` directly. The orchestrator applies writes.

Key pattern: `score_candidates(card)` uses its own lazy retriever cache (`_retrievers`), invalidated via `invalidate_retrievers()` after each card save.

### GamSearch

Manages the GAM `ResearchAgent` lifecycle. `build()` imports `amem_gam_retriever` helpers, builds page/memory stores from exported JSONL, constructs retrievers, and creates the agent. `invalidate()` clears the agent for rebuild.

### NoteSync

Bridges memory cards to the A-MEM vector store (Chroma). `upsert_fast()` skips enrichment for bulk sync; `upsert_agentic()` uses LLM content analysis. `export_jsonl()` writes card data for GAM store ingestion.
