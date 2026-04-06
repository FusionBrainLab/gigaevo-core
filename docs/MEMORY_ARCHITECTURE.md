# Memory System Architecture

The shared memory system (`gigaevo/memory/shared_memory/`) stores and retrieves evolutionary knowledge cards — ideas, strategies, and program metadata discovered during MAP-Elites runs. It supports local-only operation, API-backed persistence, agentic vector search (GAM), and LLM-based card deduplication.

---

## Module Map

```
AmemGamMemory (orchestrator, 481 lines)
  ├── config: MemoryConfig             — Pydantic config (replaces 18 kwargs)
  ├── card_store: CardStore            — Card dict + entity mappings + disk persistence
  ├── note_sync: NoteSync | None       — A-MEM / Chroma vector store bridge
  ├── api_sync: ApiSync | None         — Remote concept API synchronization
  ├── gam: GamSearch | None            — GAM ResearchAgent lifecycle
  ├── dedup: CardDedup                 — Vector scoring + LLM decision + merge
  ├── llm_service: LLMServiceProtocol  — Injected LLM (OpenAI inference)
  └── generator: GeneratorProtocol     — Injected A-MEM generator
```

### File Inventory

| File | Lines | Responsibility |
|------|------:|----------------|
| `memory.py` | 481 | Orchestrator: wires collaborators, coordinates save/search/rebuild/delete |
| `memory_config.py` | 58 | `MemoryConfig`, `GamConfig`, `ApiConfig` (Pydantic `BaseModel`) |
| `card_store.py` | 152 | Card dict, entity mappings, JSON index persistence |
| `note_sync.py` | 178 | Bridges cards to A-MEM vector store (Chroma) |
| `api_sync.py` | 207 | Paginated fetch, full sync, search via concept API |
| `gam_search.py` | 101 | GAM ResearchAgent build/invalidate lifecycle |
| `card_dedup.py` | 398 | Vector scoring, LLM dedup decision, card merge computation |
| `agentic_runtime.py` | 144 | `AgenticRuntime` Pydantic model + factory functions for LLM/storage init |
| `protocols.py` | 52 | Protocol definitions for DI (`LLMServiceProtocol`, `AgenticMemoryProtocol`, etc.) |

Supporting files (pure functions and helpers):

| File | Lines | Responsibility |
|------|------:|----------------|
| `models.py` | 88 | Pydantic card models (`MemoryCard`, `ProgramCard`, `Explanation`) |
| `card_conversion.py` | 550 | Pure functions: normalize, convert, format, search, synthesize cards |
| `card_update_dedup.py` | 582 | Pure functions: query building, weighted scoring, LLM parsing |
| `concept_api.py` | 151 | HTTP client for the remote Memory API (`_ConceptApiClient`) |
| `utils.py` | 74 | Small helpers (`truncate_text`, `looks_like_uuid`) |
| `amem_gam_retriever.py` | 269 | GAM store/retriever builders (imports Chroma, LlamaIndex) |
| `a_mem_memory_creation.py` | 301 | A-MEM note creation logic |
| `__init__.py` | 31 | Public re-exports: `AmemGamMemory`, card models, `normalize_memory_card` |

**Total**: 3,882 lines across 18 files (down from 1,280 lines in a single `memory.py`).

---

## Design Principles

### 1. Single-Responsibility Collaborators

Each class owns one concern. `CardStore` owns persistence, `NoteSync` owns the Chroma bridge, `CardDedup` owns scoring. None of them trigger rebuilds or make orchestration decisions — they return results (booleans, tuples, lists) and the orchestrator acts on them.

### 2. Orchestrator Coordinates

`AmemGamMemory` decides *when* to rebuild, *when* to persist, *when* to sync. It delegates *how* to the collaborators. The orchestrator is ~481 lines — every method fits on one screen.

### 3. Pydantic Config

`MemoryConfig` replaces 18 scattered constructor kwargs. Sub-configs (`GamConfig`, `ApiConfig`, `CardUpdateDedupConfig`) group related settings. All use `ConfigDict(extra="forbid")` for validation.

```python
config = MemoryConfig(
    checkpoint_path=Path("./mem"),
    api=ApiConfig(base_url="http://...", namespace="ns"),  # None = local-only
    gam=GamConfig(allowed_tools=["vector_description"]),
    dedup=CardUpdateDedupConfig(enabled=True),
)
mem = AmemGamMemory(config=config)
```

### 4. Protocol-Based Dependency Injection

External dependencies are typed via `typing.Protocol` (structural subtyping). Production code passes real classes; tests pass fakes with matching signatures. No monkey-patching needed.

### 5. kwargs-Only Constructors

All classes use `*` in their `__init__` signatures to force keyword arguments. This prevents positional-argument ordering bugs and makes call sites self-documenting.

### 6. No Circular Imports

Collaborator modules import from `models`, `card_conversion`, `card_update_dedup`, `concept_api` — never from `memory.py`. The import DAG is strictly acyclic:

```
memory.py
  ├── agentic_runtime.py  ← protocols.py
  ├── card_store.py       ← card_conversion.py ← models.py
  ├── note_sync.py        ← card_store.py, card_conversion.py
  ├── api_sync.py         ← card_store.py, note_sync.py, concept_api.py
  ├── gam_search.py       ← card_store.py
  ├── card_dedup.py       ← card_store.py, card_update_dedup.py
  └── card_conversion.py  ← models.py
```

---

## Class Reference

### MemoryConfig (`memory_config.py`)

```python
class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_path: Path
    search_limit: int = Field(default=5, gt=0)
    rebuild_interval: int = Field(default=10, gt=0)
    enable_llm_synthesis: bool = True
    enable_memory_evolution: bool = True
    enable_llm_card_enrichment: bool = True
    api: ApiConfig | None = None          # None = local-only mode
    gam: GamConfig = Field(default_factory=GamConfig)
    dedup: CardUpdateDedupConfig = Field(default_factory=CardUpdateDedupConfig)
```

**`GamConfig`**: `enable_bm25`, `allowed_tools`, `top_k_by_tool`, `pipeline_mode`.

**`ApiConfig`**: `base_url`, `namespace`, `author`, `channel`, `sync_batch_size`, `sync_on_init`.

### AmemGamMemory (`memory.py`)

The orchestrator. Constructor signature:

```python
class AmemGamMemory(GigaEvoMemoryBase):
    def __init__(
        self,
        *,
        config: MemoryConfig,
        runtime: AgenticRuntime | None = None,     # DI for agentic classes
        llm_service: LLMServiceProtocol | None = None,  # DI for LLM
        generator: GeneratorProtocol | None = None,      # DI for generator
    ) -> None:
```

When `runtime` is `None`, auto-detects via `load_agentic_runtime()`.
When `llm_service`/`generator` are `None`, auto-creates from environment config.

**Public API** (6 methods):

| Method | Purpose |
|--------|---------|
| `save_card(card)` | Normalize, dedup, save to store + A-MEM + API, periodic rebuild |
| `save(data, category)` | Convenience wrapper around `save_card` |
| `search(query, memory_state)` | GAM agent → API search → local keyword fallback |
| `get_card(card_id)` | Direct card lookup from store |
| `delete(memory_id)` | Remove from store + API + A-MEM, rebuild |
| `rebuild()` | Serialize → export JSONL → build GAM agent → invalidate dedup retrievers |
| `get_card_write_stats()` | Return write statistics dict |
| `close()` | Close API client |

Context manager support: `with AmemGamMemory(config=cfg) as mem: ...` calls `rebuild()` + `close()` on exit.

### CardStore (`card_store.py`)

Owns all card data and disk persistence:

```python
class CardStore:
    def __init__(self, *, index_file: Path):
        self.cards: dict[str, AnyCard] = {}
        self.entity_by_card_id: dict[str, str] = {}
        self.card_id_by_entity: dict[str, str] = {}
        self.entity_version: dict[str, str] = {}
        self.note_ids: set[str] = set()
        self.write_stats: dict[str, int] = { ... }
```

| Method | Purpose |
|--------|---------|
| `get(card_id)` | Lookup card by ID |
| `put(card_id, card)` | Store card |
| `remove(card_id)` | Pop card from dict |
| `ensure_id(card)` | Generate `mem-{uuid}` ID if missing |
| `serialize_all()` | `model_dump()` all cards |
| `persist(serialized)` | Atomic JSON write via temp file + `os.replace` |
| `link_entity(card_id, entity_id, version)` | Create bidirectional entity mapping |
| `unlink_entity(entity_id)` | Remove entity mapping, return card_id |
| `save_entity(card_id, entity_id, version)` | Link with stale-mapping cleanup |
| `clear_entity(card_id)` | Remove entity mapping for a card |
| `resolve_card_id(key)` | Resolve card_id or entity_id to a card_id in the store |

Persistence format (`api_index.json`):
```json
{
  "entity_by_card_id": { "card-1": "eid-abc" },
  "entity_version_by_entity": { "eid-abc": "v3" },
  "memory_cards": { "card-1": { ... card fields ... } }
}
```

### NoteSync (`note_sync.py`)

Bridges cards to the local A-MEM vector store (Chroma). Receives `memory_system` and `note_cls` at construction (injected, not imported).

```python
class NoteSync:
    def __init__(self, *, memory_system, note_cls, card_store: CardStore):
```

| Method | Purpose |
|--------|---------|
| `upsert_fast(card)` | Sync card into A-MEM/Chroma without LLM evolution (bulk sync path) |
| `upsert_agentic(card)` | Add/update via A-MEM's agentic add/update path (card save path) |
| `remove(card_id)` | Delete from A-MEM |
| `export_jsonl(out_path, serialized_cards)` | Export all cards as JSONL for GAM store ingestion |
| `build_note(card, existing)` | Create a MemoryNote from a card (preserving timestamps, etc.) |
| `fields_changed(existing, ...)` | Static method: compare note fields for change detection |

### ApiSync (`api_sync.py`)

Synchronizes cards between local CardStore and remote Memory API. Returns change flags — does NOT trigger rebuilds.

```python
class ApiSync:
    def __init__(
        self, *, client, card_store, note_sync, namespace, channel,
        sync_batch_size, search_limit,
    ):
```

| Method | Purpose |
|--------|---------|
| `sync(force_full)` | Paginated fetch from API, update local store. Returns `bool` (changed) |
| `search(query, memory_state)` | Search API, update local store. Returns `(cards, local_changed)` |
| `fetch_all_hits()` | Paginated fetch with namespace filtering |

Sync behavior:
- Incremental: skips entities with unchanged `version_id`
- Full: re-fetches all entities regardless of version
- Stale entity cleanup: removes local entities no longer present on remote
- Namespace filtering: only syncs entities matching configured namespace

### GamSearch (`gam_search.py`)

Manages GAM ResearchAgent lifecycle. The orchestrator calls `build()` during rebuild and reads `agent` for search dispatch.

```python
class GamSearch:
    def __init__(
        self, *, research_agent_cls, generator, card_store, checkpoint_dir,
        gam_store_dir, export_file, enable_bm25, allowed_gam_tools,
        gam_top_k_by_tool, gam_pipeline_mode,
    ):
        self.agent: Any = None
```

| Method | Purpose |
|--------|---------|
| `build()` | Import `amem_gam_retriever` helpers, build stores, create ResearchAgent |
| `invalidate()` | Clear agent reference (triggers rebuild on next `build()`) |

`build()` lazily imports `amem_gam_retriever` (which depends on Chroma/LlamaIndex) so the rest of the system works without these heavy dependencies.

### CardDedup (`card_dedup.py`)

Pure decision engine for card deduplication. Returns merge instructions — does NOT write cards.

```python
class CardDedup:
    def __init__(
        self, *, card_store, llm_service, config, allowed_gam_tools,
        gam_store_dir, export_file, checkpoint_dir,
    ):
```

| Method | Purpose |
|--------|---------|
| `score_candidates(card)` | Vector similarity scoring against existing cards |
| `format_for_llm(scored)` | Truncate and format candidates for LLM prompt |
| `decide_action(card, candidates)` | Ask LLM: `add`, `discard`, or `update` |
| `compute_merges(card, updates)` | Compute `(card_id, merged_card)` pairs from update actions |
| `invalidate_retrievers()` | Clear cached retrievers (after card store changes) |
| `build_retrievers()` | Build dedup retriever index from exported records |
| `resolve_retriever(tool_name)` | Lazy-build + resolve retriever by tool name |

Dedup retriever management is independent from GAM search retrievers — dedup uses its own cache (`_retrievers`), invalidated via `invalidate_retrievers()` after each card save.

### AgenticRuntime (`agentic_runtime.py`)

Bundles the four agentic class dependencies for DI:

```python
class AgenticRuntime(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    memory_system_cls: type[Any]   # AgenticMemorySystem
    memory_note_cls: type[Any]     # MemoryNote
    research_agent_cls: type[Any]  # ResearchAgent
    generator_cls: type[Any]       # AMemGenerator
```

Factory functions in the same module:

| Function | Purpose |
|----------|---------|
| `load_agentic_runtime()` | Import A-MEM + GAM deps. Returns `AgenticRuntime | None` |
| `init_llm_and_generator(...)` | Create LLM service + generator from env config |
| `init_agentic_storage(...)` | Create A-MEM system (Chroma vector store) |

### Protocols (`protocols.py`)

Structural types for external dependencies:

```python
class LLMServiceProtocol(Protocol):
    def generate(self, data: str) -> tuple[str, Any, int | None, float | None]: ...

class AgenticMemoryProtocol(Protocol):
    memories: dict[str, Any]
    retriever: Any
    def read(self, memory_id: str) -> MemoryNoteProtocol | None: ...
    def add_note(self, content: str, **kwargs: Any) -> str: ...
    def update(self, memory_id: str, **kwargs: Any) -> bool: ...
    def delete(self, memory_id: str) -> bool: ...
    def analyze_content(self, content: str) -> dict[str, Any]: ...
    def _document_for_note(self, note: MemoryNoteProtocol) -> str: ...

class ResearchAgentProtocol(Protocol):
    def research(self, request: str, memory_state: str | None = None) -> ResearchOutput: ...

class GeneratorProtocol(Protocol):
    def generate_single(self, prompt: str | None = None, **kwargs: Any) -> dict[str, Any]: ...
```

---

## Data Flow

### save_card

```
save_card(card_dict)
  → normalize_memory_card(card_dict)
  → if existing card: _save_and_persist(card) → _save_card_core → card_store + note_sync + API
  → if program card: _save_and_persist(card) (skip dedup)
  → if dedup enabled + LLM available:
      dedup.score_candidates(card) → vector similarity scores
      dedup.format_for_llm(scored) → truncated payloads
      dedup.decide_action(card, candidates) → {action, updates}
        → "discard": return existing card_id (rejected)
        → "update": _apply_update_actions → dedup.compute_merges → _save_card_core per merge
        → "add": fall through
  → _save_and_persist(card)
      → _save_card_core(card)
          → card_store.ensure_id(card)
          → LLM enrichment (keywords, context) if enabled
          → API save (if api != None) → card_store.save_entity(...)
          → card_store.cards[id] = card
          → note_sync.upsert_agentic(card) (if note_sync != None)
          → dedup.invalidate_retrievers()
          → periodic rebuild() if _iters_after_rebuild >= rebuild_interval
      → card_store.persist() (unless rebuild already persisted)
```

### search

```
search(query, memory_state)
  → if API: _sync_from_api → api_sync.sync → rebuild if changed
  → if research_agent: research_agent.research(query) → integrated_memory
      → on failure: fall through to next tier
  → if API: _search_via_api → api_sync.search → synthesize/format results
  → fallback: _search_local_cards → keyword matching over card_store.cards
```

Search tiers (in order of preference):
1. **GAM agentic search**: ResearchAgent with vector retrievers (most sophisticated)
2. **API search**: Concept API full-text search + LLM synthesis
3. **Local keyword search**: Jaccard-like keyword matching over in-memory cards

### rebuild

```
rebuild()
  → card_store.serialize_all() → card_store.persist()
  → note_sync.export_jsonl() (JSONL for GAM store)
  → gam.build() → sets gam.agent → assigns self.research_agent
  → dedup.invalidate_retrievers()
  → _iters_after_rebuild = 0
```

Triggered:
- After every `rebuild_interval` card saves (default: 10)
- After API sync detects changes
- After delete (if agentic deps available)
- On context manager exit (if cards were saved since last rebuild)

### delete

```
delete(memory_id)
  → if API: resolve entity_id → api.delete_concept → unlink entity
  → else: resolve card_id → clear entity mapping
  → card_store.cards.pop(card_id)
  → note_sync.remove(card_id)
  → rebuild() or card_store.persist()
```

---

## Constructor Wiring

The `__init__` of `AmemGamMemory` wires all collaborators in a specific order:

```
1. config, paths, index_file, export_file
2. API client (from config.api)
3. Agentic runtime (DI or auto-detect)
4. CardStore (loads existing index from disk)
5. LLM service + generator (DI or from environment)
6. A-MEM storage (from runtime + LLM service)
7. NoteSync (from memory_system + note_cls + card_store)
8. Normalized GAM settings
9. CardDedup (always created; config.enabled gates scoring)
10. GamSearch (from runtime + generator + card_store)
11. ApiSync (from api client + card_store + note_sync)
12. Initial GAM build (if export file exists)
13. Initial API sync (if sync_on_init)
```

Collaborators share the same `CardStore` instance by reference. Single-threaded (MAP-Elites does not use concurrent card access).

---

## External Consumers

### Hydra Integration

Train scripts never touch `AmemGamMemory` directly. The path is:

```
config/memory/local.yaml
  → memory_provider._target_ = gigaevo.memory.provider.SelectorMemoryProvider
      → lazy creates MemorySelectorAgent
          → reads config/memory_backend.yaml (runtime settings)
          → constructs MemoryConfig + AmemGamMemory(config=...) for local mode
          → or constructs platform backend with legacy kwargs for API mode
```

Config files:
- `config/memory/none.yaml` → `NullMemoryProvider` (no-op, default)
- `config/memory/local.yaml` → `SelectorMemoryProvider` (local backend)
- `config/memory/api.yaml` → `SelectorMemoryProvider` (API backend)

### MemorySelectorAgent (`gigaevo/llm/agents/memory_selector.py`)

The bridge between Hydra config and the memory backend. For local mode (the common case):

```python
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.memory_config import GamConfig, MemoryConfig

mem_config = MemoryConfig(
    checkpoint_path=memory_dir,
    search_limit=search_limit,
    rebuild_interval=rebuild_interval,
    enable_llm_synthesis=runtime_enable_llm_synthesis,
    enable_memory_evolution=runtime_enable_memory_evolution,
    enable_llm_card_enrichment=runtime_fill_missing_fields,
    gam=GamConfig(
        enable_bm25=enable_bm25,
        allowed_tools=allowed_gam_tools or [],
        top_k_by_tool=gam_top_k_by_tool or {},
        pipeline_mode=gam_pipeline_mode or "default",
    ),
)
memory = AmemGamMemory(config=mem_config)
```

For API mode, the platform backend (`gigaevo.memory_platform.AmemGamMemory`) still uses legacy kwargs — it has its own constructor.

### SelectorMemoryProvider (`gigaevo/memory/provider.py`)

Strategy object injected into the DAG pipeline via Hydra:
- `NullMemoryProvider`: no-op, returns empty selection
- `SelectorMemoryProvider`: delegates to `MemorySelectorAgent` (lazy init)
  - `select_cards(program, ...)` → `MemorySelection(cards, card_ids)`

---

## Testing Infrastructure

### Test Factories (`tests/fakes/agentic_memory.py`)

Two factory functions provide pre-wired test instances:

**`make_test_memory(tmp_path, **overrides)`** — Local-only mode. No agentic system, no LLM, no generator. Used for card persistence, API client, dedup, and basic save/search tests.

```python
mem = make_test_memory(tmp_path, rebuild_interval=3, search_limit=10)
```

**`make_test_memory_with_agentic(tmp_path, **overrides)`** — Full fake agentic infrastructure pre-wired via constructor DI. Returns `(memory, fake_system)`.

```python
mem, fake_sys = make_test_memory_with_agentic(tmp_path, enable_llm_card_enrichment=True)
# fake_sys is a FakeAgenticMemorySystem — inspect fake_sys.memories, fake_sys.retriever
```

Both accept the same overrides: `search_limit`, `rebuild_interval`, `enable_llm_synthesis`, `enable_memory_evolution`, `enable_llm_card_enrichment`, `card_update_dedup_config` (dict), `api` (ApiConfig), `gam` (GamConfig).

### Fake Classes

| Fake | Real Counterpart | Purpose |
|------|-----------------|---------|
| `FakeMemoryNote` | `MemoryNote` | In-memory note with all fields |
| `FakeAgenticMemorySystem` | `AgenticMemorySystem` | Dict-backed memory store + `FakeRetriever` |
| `FakeRetriever` | ChromaRetriever | Jaccard similarity keyword search |
| `FakeResearchAgent` | ResearchAgent | Searches fake retrievers, formats results |
| `FakeAMemGenerator` | AMemGenerator | Returns canned LLM responses |
| `FakeMemoryStore` | InMemoryMemoryStore | List-backed memory store |
| `FakePageStore` | InMemoryPageStore | List-backed page store |
| `FakeSearchResult` | GAM search result | `page_id`, `score`, `meta` |

The `_get_fake_runtime()` helper constructs an `AgenticRuntime` with all fake classes:
```python
AgenticRuntime(
    memory_system_cls=FakeAgenticMemorySystem,
    memory_note_cls=FakeMemoryNote,
    research_agent_cls=FakeResearchAgent,
    generator_cls=FakeAMemGenerator,
)
```

### Test Files

| File | Tests | Coverage |
|------|------:|----------|
| `test_memory_backend_fakes.py` | Fake infrastructure: upsert, remove, rebuild, enrichment, full lifecycle |
| `test_memory_backend_agentic.py` | Full agentic path: GAM search, dedup with retrievers |
| `test_api_sync.py` | API sync, dedup LLM decisions, request body verification |
| `test_api_client.py` | HTTP client, truncation, error handling |
| `test_roundtrip.py` | Save → persist → reload → verify round-trip integrity |
| `test_memory_selector_agent.py` | MemorySelectorAgent integration |
| `test_memory_backend.py` | Basic save/search/delete |
| `test_card_update_dedup.py` | Pure scoring functions |
| `test_card_conversion.py` | Pure conversion functions |
| `test_models.py` | Pydantic model validation |
| `test_memory_realistic_e2e.py` | Integration: real EvolutionEngine + FakeDagRunner + memory |

All tests use constructor-time dependency injection — no monkey-patching, no `inject_fakes_into_memory`.

---

## Error Handling

### Graceful Degradation

The system degrades gracefully when optional dependencies are unavailable:

| Missing | Effect |
|---------|--------|
| A-MEM/GAM imports | `runtime = None`, local keyword search only |
| OpenAI API key | `llm_service = None`, no synthesis, no dedup, no enrichment |
| Chroma vector store | `memory_system = None`, no note sync, no GAM |
| Concept API server | `api = None`, local-only mode |

Each init step is wrapped in try/except with `logger.warning` — the system always starts, with reduced capabilities.

### Atomic Persistence

`CardStore.persist()` writes to a temp file (`api_index.{pid}.tmp`) then calls `os.replace()` for atomic rename. This prevents corruption from interrupted writes.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Hydra Config Layer                        │
│  config/memory/{none,local,api}.yaml                        │
│     → NullMemoryProvider | SelectorMemoryProvider           │
└────────────────┬────────────────────────────────────────────┘
                 │ lazy creates
┌────────────────▼────────────────────────────────────────────┐
│              MemorySelectorAgent                             │
│  Reads config/memory_backend.yaml + env vars                │
│  Builds MemoryConfig, creates AmemGamMemory                 │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│              AmemGamMemory (orchestrator)                    │
│  save_card ─┬─> CardStore ──> persist (api_index.json)      │
│             ├─> NoteSync  ──> A-MEM (Chroma)                │
│             ├─> ApiSync   ──> Concept API (HTTP)            │
│             └─> CardDedup ──> vector scoring + LLM          │
│                                                              │
│  search ───┬─> GamSearch  ──> ResearchAgent (GAM)           │
│            ├─> ApiSync    ──> API full-text search           │
│            └─> local      ──> keyword matching               │
│                                                              │
│  rebuild ──┬─> CardStore.persist()                           │
│            ├─> NoteSync.export_jsonl()                       │
│            ├─> GamSearch.build()                             │
│            └─> CardDedup.invalidate_retrievers()             │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Direction

```
memory.py (orchestrator)
    │
    ├──▶ memory_config.py ──▶ card_update_dedup.py
    ├──▶ agentic_runtime.py ──▶ protocols.py ──▶ card_conversion.py
    ├──▶ card_store.py ──▶ card_conversion.py ──▶ models.py
    ├──▶ note_sync.py ──▶ card_store.py, card_conversion.py
    ├──▶ api_sync.py ──▶ card_store.py, note_sync.py, concept_api.py
    ├──▶ gam_search.py ──▶ card_store.py
    ├──▶ card_dedup.py ──▶ card_store.py, card_update_dedup.py
    └──▶ concept_api.py (HTTP client, no internal deps)
```

All arrows point downward — no cycles.
