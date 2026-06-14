# GigaEvo Memory System

Core components:
- **AmemGamMemory** (`shared_memory/memory.py`): Main orchestrator — saves cards, searches memory, manages lifecycle
- **CardStore** (`shared_memory/card_store.py`): In-memory card catalog with persistence to disk
- **NoteSync** (`shared_memory/note_sync.py`): Bridges cards ↔ local A-MEM vector store (Chroma)
- **GamSearch** (`shared_memory/gam_search.py`): Manages GAM `ResearchAgent` lifecycle (build/rebuild/clear)
- **AmemGamRetriever** (`shared_memory/amem_gam_retriever.py`): GAM store + retriever construction helpers
- **CardSearch** (`shared_memory/card_search.py`): Lexical search, ranking, and optional LLM synthesis
- **CardDedup** (`shared_memory/card_dedup.py`): LLM-scored duplicate detection before writes
- **CardConversion** (`shared_memory/card_conversion.py`): Card ↔ A-MEM note conversion + JSONL export
- **ApiSync** (`shared_memory/api_sync.py`): Optional API backend for remote concept sync

All modules log under a uniform `[Memory][<Component>]` prefix —
single-grep `\[Memory\]\[` covers the whole subsystem.

API-backed mode wraps local A-MEM + GAM retrieval with cloud persistence:
- **Source of truth**: Remote API concepts (`/v1/concepts`)
- **Local runtime**: Synchronized A-MEM notes + GAM retrievers in Chroma vectors
- **Local cache**: `api_index.json` + `amem_exports/` + `gam_shared/` directories

## One knob: `memory={none,reader,writer,full}` (read this before launching)

Memory is **one Hydra knob with four presets**. It assembles a single
`MemorySystem` node that owns *both* the read side (the `MemoryProvider` that
injects cards into the mutation context) and the write side (the `IdeaTracker`
that extracts/enriches cards with the `memory/llm` router). Two booleans inside
that node — `reader_enabled` and `writer_enabled` — are what each preset flips:

| Preset | reader | writer | What runs | Cost |
|---|---|---|---|---|
| `memory=none` | off | off | `NullMemoryProvider` + `NullPostRunHook` | none |
| `memory=reader` | on | off | injects cards from an existing bank; no extraction | read only |
| `memory=writer` | off | on | extracts/enriches cards into a bank for a *later* run; injects nothing | `memory/llm` spend |
| `memory=full` | on | on | reader + writer share **one** card bank | `memory/llm` spend |

The **writer** spends money on `memory/llm` (default `gemini`, swap with
`memory/llm=qwen_instruct`); the **reader** changes fitness. `full` shares a
single backend/reputation/dedup between the two — that sharing is a Python fact
inside `MemorySystem`, not a `${ref:memory.*}` YAML web.

- **Memory arm ON** (read + write): `pipeline=intra_extra_memory memory=full`
- **True no-memory baseline**: `pipeline=standard memory=none`
- **Seed a bank for later** (`memory=writer`): extracts/enriches but injects
  nothing. **Do not** use this as a "no-memory" run — it pays full
  `memory/llm` cost for cards nobody reads.
- **`memory=reader`** injects from a pre-populated bank without paying to write.

> ⚠️ **Picking `writer` or `full` turns the `memory/llm` writer on and bills it.**
> A no-memory baseline is `memory=none` + `pipeline=standard` — a single preset
> disables both sides, so the old "writer left on" footgun
> (`docs/audits/NOMEM_BASELINE_WRITER_LEFT_ON.md`) is gone: there is no second
> knob to forget.

### Guards (fail-fast)

`pipeline=intra_extra_memory` reads *and* writes; `memory=full` is the canonical
arm. The two sides are validated **asymmetrically** at startup:

- **Writer guard (raises):** `LiveMemoryRefreshHook.__init__` raises `TypeError`
  unless its tracker is a real `IncrementalPostRunHook` — so the writer-off presets
  (`memory=none`, `memory=reader`) fail fast under this pipeline; use
  `pipeline=standard` for those.
- **Reader guard (warns):** `IntraExtraMemoryPipelineBuilder.__init__` only
  **warns** on a `NullMemoryProvider` — it does *not* raise, because `memory=writer`
  is a legitimate write-cost-controlled baseline (cards written by the tracker,
  never injected). `memory=full` turns the read path back on.

For a true no-memory baseline use `pipeline=standard memory=none` (both sides off,
no guard fires).

### Verify the resolved arm from the log

The startup banner states the *resolved* wiring — check it, don't assume:

```
[Memory][Arm] provider=SelectorMemoryProvider tracker=IdeaTracker
              post_step_hook=LiveMemoryRefreshHook pipeline_builder=IntraExtraMemoryPipelineBuilder
```

A no-memory run shows `provider=NullMemoryProvider tracker=NullPostRunHook
pipeline_builder=IntraMemoryPipelineBuilder`. Any `IdeaTracker` in the banner means the
**writer is on and will spend** on `memory/llm`.

## Memory Flow for New Users

### Writing a Memory (Step 1)

```
your code calls:  memory.save_card({"description": "...", ...})
       ↓
AmemGamMemory checks:
  - Is dedup enabled? Run CardDedup (vector search + LLM scoring) to decide:
    ├─ "add" → save as new card
    ├─ "discard" → skip this card
    └─ "update" → merge into existing card
       ↓
   normalize card (use CardStore to assign/verify IDs)
       ↓
   (optional) write to API in legacy API-backed mode
       ↓
   store in CardStore (in-memory + persisted to disk)
       ↓
   sync with A-MEM (NoteSync exports JSONL → Chroma)
       ↓
periodic rebuild (every N writes, or after API sync):
  - re-export all cards to JSONL
  - rebuild GAM index (ResearchAgent for agentic retrieval)
```

### Searching Memory (Step 2)

```
your code calls:  memory.search(query)
       ↓
AmemGamMemory routes through fallback chain:
  1. GAM agentic search (if ResearchAgent available)
     └─ calls GAM ResearchAgent.research(query)
          └─ semantic retrieval via Chroma vectors
               └─ returns structured answer with memories
  
  2. [fallback] API full-text search (if API mode enabled)
     └─ calls API /v1/search endpoint
  
  3. [fallback] Local lexical search (always available)
     └─ token-overlap matching over CardStore.cards
       ↓
   optional: LLM synthesis (if enabled)
     └─ summarizes results into natural answer
       ↓
   return results to caller
```

### Complete Lifecycle in Code

```python
# Initialize with optional API backend
mem = AmemGamMemory(
    config=MemoryConfig(...),
    # Optional: set llm_service, generator, runtime for agentic features
)

# 1. Write
mem.save_card({
    "id": "idea_1",
    "description": "Simulated annealing for optimization",
    "category": "strategy"
})  # → dedup checked, stored locally, synced to A-MEM

# 2. Search
results = mem.search("how to escape local minima?")
# → tries GAM → API → local, returns ranked cards

# 3. Manage lifecycle
mem.rebuild()  # force export + retriever rebuild
mem.close()    # clean shutdown
```

### Data ownership model

- API is authoritative when API mode is enabled.
- Local state is a synchronized, query-optimized cache:
  - `api_index.json`: card ID <-> entity UUID mapping + known version IDs + normalized cards
  - `amem_exports/amem_memories.jsonl`: exported cards for GAM ingestion
  - `gam_shared/amem_store/...`: GAM page/index store
  - `chroma/...`: local vector index used by A-MEM/GAM components

## API Mapping

Cards are represented locally in a normalized schema (`shared_memory/models.py`) and mapped to API concept payloads.

### Local card fields

- `id`, `category`, `description`, `task_description`, `task_description_summary`, `strategy`
- `keywords`, `links`, `works_with`
- `explanation.summary`
- optional maps: `evolution_statistics`

### API write mapping

`save_card(...)` writes:
- `content`: normalized concept content (card fields)
- `meta.name`: derived from `id` + description/task text
- `meta.tags`: category + strategy + keywords
- `meta.when_to_use`: joined context/description/explanation summary/keywords
- `meta.namespace`, `meta.author`
- `channel` (default `latest`)

Writes use:
- `POST /v1/concepts` for new cards
- `PUT /v1/concepts/{entity_id}` for updates

Reads/search use:
- `GET /v1/search?entity_type=concept...`
- `GET /v1/concepts/{entity_id}?channel=...`
- `DELETE /v1/concepts/{entity_id}`

## Sync + Rebuild Lifecycle

### On initialization

- Loads local `api_index.json` if present.
- Initializes optional LLM/generator/A-MEM/GAM runtime.
- If `sync_on_init=true` and API mode is enabled, performs full sync of concept entities.

### Incremental sync during search

- `search(...)` calls `_sync_from_api(force_full=False)`.
- It fetches concept hits page-by-page (`sync_batch_size`) for the configured namespace.
- Version IDs are used to skip unchanged entities.
- Changed/new entities are fetched, converted back to cards, and upserted locally.
- Deleted remote entities are removed locally.

### Rebuild trigger

Rebuild regenerates export + GAM retrievers:
- automatic after sync if local state changed
- periodic after writes (`rebuild_interval`, default 30 saves)
- explicit via `memory.rebuild()`

## Retrieval Strategy and Fallbacks

`search(query, memory_state=None)` behavior:

1. If available, use GAM `ResearchAgent.research(...)` (agentic retrieval).
2. On GAM error/unavailability, use API full-text `/v1/search`.
3. In local-only mode (`api=None`), use local token-overlap search.
4. Optional LLM synthesis can post-process retrieved cards into a final answer.

If no OpenRouter key is provided:
- agentic retrieval/generator is disabled
- output falls back to plain card listing or API search responses

## Configuration

This package reads **no environment variables** — enforced by
`tests/memory/test_no_env_in_memory.py`. All wiring comes from Hydra:

- `memory={none,reader,writer,full}` selects the assembled `MemorySystem`
  (`config/memory/`) — one preset flips the `reader_enabled` / `writer_enabled`
  booleans that own the read-side provider and the write-side tracker.
- `config/memory/backend/local.yaml` builds the shared card-bank backend via
  `LocalMemoryBackendFactory`; its fields are the tuning surface.
- The memory LLM is the `config/memory/llm/` group (`gemini` default,
  `qwen_instruct`); `MemorySystem` threads it into the backend in Python.

Effective defaults shipped by `LocalMemoryBackendFactory`:
- `search_limit` (default `5`)
- `rebuild_interval` (default `30`)
- `enable_llm_synthesis` (default `false`)
- `enable_bm25` (default `false`)
- `embedding_model_name` (default `all-MiniLM-L6-v2`)

`MemoryConfig` (`shared_memory/memory_config.py`) is the validated runtime
config object the factory assembles; construct it directly only in tests.

## Quick Start

For end-to-end usage from an evolution run, see the root-level
[`README_memory.md`](../../README_memory.md) (single-pass live pipeline,
paper arm matrix).

## Local-only Mode

Local-only is the default and the only mode used by paper runs: construct
with `api=None` in `MemoryConfig` (what the `memory=reader` / `memory=full`
presets ship). Behavior:
- no API writes/reads/sync
- cards are kept locally
- retrieval is local (agentic if LLM/runtime available, otherwise lexical fallback)

The remote-API path (`api_sync.py`, `concept_api.py`) is legacy platform code,
no longer wired to a Hydra preset, and not exercised by current experiments.

## Troubleshooting

- `Cannot connect to Memory API at ...`: API service is not running or wrong `MEMORY_API_URL`.
- `OPENROUTER_API_KEY is not set...`: agentic retrieval disabled; fallback path still works.
- `Agentic runtime dependencies are unavailable...`: A-MEM/GAM import/init failed; fallback to API/local search.
- Empty results after writes: run `memory.rebuild()` to force export + retriever rebuild.

## Current Limitation

- Main API vector search endpoint (`/v1/search/vector`) is not used here.
- Vector/agentic retrieval is done locally through synchronized A-MEM/GAM indexes.
