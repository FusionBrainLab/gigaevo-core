# Evo Memory Agent (API-Backed Copy)

This package adapts `evo_memory_agent` to the main repository Memory API schema.

It wraps local A-MEM + GAM retrieval with API-backed persistence:
- Source of truth: API concepts (`/v1/concepts`)
- Local runtime: synchronized A-MEM notes + GAM retrievers
- Local mapping/index: `api_index.json` in the chosen checkpoint directory

## How It Works

`AmemGamMemory` (`shared_memory/memory.py`) is the main entrypoint.

### High-level flow

1. `save_card(...)` normalizes a memory card and persists it as a concept entity in the API (when `MEMORY_USE_API=true`).
2. Optional card update/dedup pipeline (config: `card_update_dedup` in `config/memory.yaml`) can run before write:
   - multi-query vector retrieval (`description`, `explanation summary`, and two combined fields)
   - weighted rerank
   - LLM decision: `add` / `discard` / `update`
3. The saved card is also upserted into local A-MEM runtime for retrieval.
4. `search(...)` first performs incremental sync from API, then tries GAM agentic retrieval.
5. If GAM is unavailable/fails, it falls back to API full-text search.
6. If API mode is disabled, it falls back to local lexical search over cached cards.

### Data ownership model

- API is authoritative when API mode is enabled.
- Local state is a synchronized, query-optimized cache:
  - `api_index.json`: card ID <-> entity UUID mapping + known version IDs + normalized cards
  - `amem_exports/amem_memories.jsonl`: exported cards for GAM ingestion
  - `gam_shared/amem_store/...`: GAM page/index store
  - `chroma/...`: local vector index used by A-MEM/GAM components

## API Mapping

Cards are represented locally in a normalized schema (examples in `new_mem_example.json`) and mapped to API concept payloads.

### Local card fields

- `id`, `category`, `description`, `task_description`, `task_description_summary`, `strategy`
- `keywords`, `links`, `works_with`
- `explanation.summary`
- optional maps: `evolution_statistics`, `usage`

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
- periodic after writes (`rebuild_interval`, default 10 saves)
- explicit via `memory.rebuild()`

## Retrieval Strategy and Fallbacks

`search(query, memory_state=None)` behavior:

1. If available, use GAM `ResearchAgent.research(...)` (agentic retrieval).
2. On GAM error/unavailability, use API full-text `/v1/search`.
3. In local-only mode (`MEMORY_USE_API=false`), use local token-overlap search.
4. Optional LLM synthesis can post-process retrieved cards into a final answer.

If no OpenRouter key is provided:
- agentic retrieval/generator is disabled
- output falls back to plain card listing or API search responses

## Configuration

Environment variables commonly used:

```bash
MEMORY_API_URL=http://localhost:8000
MEMORY_NAMESPACE=exp5
MEMORY_USE_API=true
OPENROUTER_API_KEY=...
OPENROUTER_MODEL_NAME=openai/gpt-4.1-mini
```

Important runtime flags in `AmemGamMemory(...)`:
- `search_limit` (default `5`)
- `rebuild_interval` (default `10`)
- `enable_llm_synthesis` (default `true`)
- `enable_bm25` (default `false`)
- `allowed_gam_tools` (default: all supported GAM tools)
- `gam_top_k_by_tool` (per-tool max retrieved hits, defaults to `5` each)
- `gam_pipeline_mode` (`default` | `experimental`, default `default`)
- `sync_batch_size` (default `100`)
- `sync_on_init` (default `true`)
- `channel` (default `latest`)
- `namespace`, `author`

## Quick Start

1. Start API stack from repo root:

```bash
make up
```

2. Write cards to API:

```bash
python gigaevo.memory/memory_write_example.py
```

3. Search:

```bash
python gigaevo.memory/memory_read_example.py
```

4. Save + search in one script:

```bash
python gigaevo.memory/shared_memory/memory_usage_example.py
```

## Local-only Mode

Set:

```bash
MEMORY_USE_API=false
```

Behavior in this mode:
- no API writes/reads/sync
- cards are kept locally
- retrieval is local (agentic if LLM/runtime available, otherwise lexical fallback)

## Troubleshooting

- `Cannot connect to Memory API at ...`: API service is not running or wrong `MEMORY_API_URL`.
- `OPENROUTER_API_KEY is not set...`: agentic retrieval disabled; fallback path still works.
- `Agentic runtime dependencies are unavailable...`: A-MEM/GAM import/init failed; fallback to API/local search.
- Empty results after writes: run `memory.rebuild()` to force export + retriever rebuild.

## Current Limitation

- Main API vector search endpoint (`/v1/search/vector`) is not used here.
- Vector/agentic retrieval is done locally through synchronized A-MEM/GAM indexes.
