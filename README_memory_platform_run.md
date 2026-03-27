# Running `gigaevo-core-internal` With `memory_platform` + `gigaevo-memory`

This guide explains how to run:

- `gigaevo-memory` as the backend API on `http://localhost:8000`
- `gigaevo-core-internal/run.py` with `ideas_tracker=true`
- the new `gigaevo.memory_platform` backend automatically selected when `api.use_api: true`

## What switches the backend

The backend choice is now tied to `use_api`:

- `api.use_api: false` -> legacy `gigaevo.memory.shared_memory.memory`
- `api.use_api: true` -> new `gigaevo.memory_platform`

This applies to:

- the ideas-tracker final memory write path
- the runtime memory selector path when `memory_enabled=true`

This selection is automatic now:

- if `api.use_api: true`, `gigaevo-core-internal` uses `gigaevo.memory_platform`
- if `api.use_api: false`, it uses the legacy local backend

There is no silent fallback from platform to legacy when `use_api=true`. If platform import/init fails, the run should fail visibly so the test result is trustworthy.

## Prerequisites

You need:

- Docker or another way to run PostgreSQL and Redis
- a Python 3.12 environment for `gigaevo-memory`
- your normal `gigaevo-core-internal` environment

`uv` is optional. If you do not have `uv`, use `pip` in a dedicated Python 3.12 environment.

## 1. Start PostgreSQL and Redis

`gigaevo-memory` requires the PostgreSQL `pgvector` extension. A plain `postgres:15` image is not enough.

```bash
docker run -d --name gigaevo-memory-postgres \
  -e POSTGRES_DB=gigaevo \
  -e POSTGRES_USER=gigaevo \
  -e POSTGRES_PASSWORD=gigaevo \
  -p 5432:5432 \
  pgvector/pgvector:pg15

docker run -d --name gigaevo-memory-redis \
  -p 6379:6379 \
  redis:7-alpine
```

If you already have PostgreSQL and Redis running locally, reuse them.

## 2. Create a Python 3.12 environment for `gigaevo-memory`

### Option A: `venv`

```bash
cd /home/petranokhin/projects/gigaevo_memory/gigaevo-memory

python3.12 -m venv .venv-memory
source .venv-memory/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ./api
python -m pip install sentence-transformers
```

### Option B: conda

```bash
conda create -n gigaevo-memory python=3.12 -y
conda activate gigaevo-memory

cd /home/petranokhin/projects/gigaevo_memory/gigaevo-memory

python -m pip install --upgrade pip
python -m pip install -e ./api
python -m pip install sentence-transformers
```

If you have `uv`, you can use it instead, but it is optional.

## 3. Export backend environment variables

The repo defaults point to Docker hostnames like `postgres` and `redis`. If you run the API directly on your machine, override them with `localhost`.

```bash
export POSTGRES_DSN='postgresql+asyncpg://gigaevo:gigaevo@localhost:5432/gigaevo'
export REDIS_URL='redis://localhost:6379/0'
export ENABLE_VECTOR_SEARCH=true
export EMBEDDING_PROVIDER=sentencetransformers
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Notes:

- `ENABLE_VECTOR_SEARCH=true` is recommended for a real end-to-end test because `memory_platform` uses backend vector search for GAM-style retrieval and dedup retrieval.
- `sentence-transformers` is required for that local embedding path.

## 4. Run `gigaevo-memory` migrations

```bash
cd /home/petranokhin/projects/gigaevo_memory/gigaevo-memory/api
alembic -c app/db/alembic.ini upgrade head
```

This must include the new migration for search documents:

- [002_memory_card_search_documents.py](/home/petranokhin/projects/gigaevo_memory/gigaevo-memory/api/app/db/migrations/versions/002_memory_card_search_documents.py)

If you see:

```text
FAILED: No 'script_location' key found in configuration.
```

it means you ran `alembic upgrade head` without the repo config file. Use `-c app/db/alembic.ini`.

If you see:

```text
extension "vector" is not available
```

you started PostgreSQL without `pgvector`. Replace it with the `pgvector/pgvector:pg15` image above.

## 5. Start the `gigaevo-memory` API

In the same Python 3.12 env:

```bash
cd /home/petranokhin/projects/gigaevo_memory/gigaevo-memory/api
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Leave this terminal running.

## 6. Verify the backend is healthy

In another terminal:

```bash
curl http://localhost:8000/health
```

Expected shape:

```json
{
  "status": "ok",
  "postgres": "ok",
  "redis": "ok"
}
```

`status: degraded` usually means PostgreSQL or Redis env is wrong.

## 7. Configure `gigaevo-core-internal`

Edit [config/memory.yaml](/home/petranokhin/projects/gigaevo_memory/gigaevo-core-internal/config/memory.yaml).

Use at least:

```yaml
api:
  base_url: http://localhost:8000
  namespace: your_test_namespace
  use_api: true
  channel: latest

ideas_tracker:
  memory_write_pipeline:
    enabled: true
```

Recommended for a full test:

- keep `api.use_api: true`
- use a fresh namespace for each test run
- keep `api.channel: latest` unless you intentionally need a different channel
- keep `ideas_tracker.memory_write_pipeline.enabled: true`
- run `memory_enabled=true` if you want to test runtime reads too

### Namespace and channel

When `api.use_api: true`, both writing and runtime retrieval use:

- `api.namespace`
- `api.channel`

Environment overrides:

- `MEMORY_API_URL` overrides `api.base_url`
- `MEMORY_NAMESPACE` overrides `api.namespace`
- `MEMORY_USE_API` overrides `api.use_api`

At the moment, read and write share the same namespace/channel config. There is no separate `read_namespace` vs `write_namespace`.

## 8. Run `gigaevo-core-internal`

Open a new terminal and activate your normal core environment.

```bash
conda activate <your-gigaevo-core-env>
cd /home/petranokhin/projects/gigaevo_memory/gigaevo-core-internal

export MEMORY_API_URL=http://localhost:8000
python run.py ideas_tracker=true memory_enabled=true
```

Meaning:

- `ideas_tracker=true` runs the ideas tracker and final memory write pipeline
- `memory_enabled=true` also exercises runtime memory-selector reads during the experiment

If you only want to test the final ideas-tracker write path, run:

```bash
python run.py ideas_tracker=true
```

## 9. What should happen

When `api.use_api: true`, `gigaevo-core-internal` should automatically use `gigaevo.memory_platform`.

You should see:

- the experiment runs normally
- ideas tracker runs at the end
- the memory write pipeline prints stats
- if `memory_enabled=true`, runtime memory selection should also initialize against the backend API

### What memory is used for retrieval

If:

- `memory_enabled=true`
- `api.use_api=true`

then runtime retrieval uses memory cards stored in `gigaevo-memory` for the configured:

- `namespace`
- `channel`

`memory_platform` reads those cards from the backend and rebuilds a local GAM page store from them for runtime use.

Important:

- the backend database is the source of truth
- the checkpoint directory is only local runtime state/cache
- the ideas-tracker final write happens after the main experiment, so memory written at the end of a run is generally available for future runs, not earlier retrieval inside that same run

## 10. Validate written data in the backend

List memory cards:

```bash
curl "http://localhost:8000/v1/memory-cards?limit=20&offset=0&channel=latest"
```

Run a search:

```bash
curl -X POST "http://localhost:8000/v1/search/unified" \
  -H "Content-Type: application/json" \
  -d '{
    "search_type": "bm25",
    "query": "your expected idea text",
    "entity_type": "memory_card",
    "namespace": "your_test_namespace",
    "document_kind": "full_card",
    "top_k": 5
  }'
```

If vector search is enabled, you can also test hybrid or vector through the API/UI later.

## 11. Checkpoint directory when `use_api=true`

If `api.use_api: true`, the persistent memory cards are saved in `gigaevo-memory` database, not in the checkpoint directory.

But the checkpoint directory is still used locally by `memory_platform` for runtime artifacts, such as:

- `platform_index.json`
- `gam_shared/platform_store`

So:

- remote persistent storage: backend database
- local runtime artifacts: checkpoint directory

You do not need to pass `checkpoint_dir` to persist memory in the backend.
You may still want to set it to isolate local runtime files per run.

## 12. Common failures

### `uv: command not found`

Not a problem. Use `pip` in a Python 3.12 env as shown above.

### `python3.12: command not found`

Use conda:

```bash
conda create -n gigaevo-memory python=3.12 -y
conda activate gigaevo-memory
```

### `ModuleNotFoundError: sentence_transformers`

Install:

```bash
python -m pip install sentence-transformers
```

### `/health` shows degraded postgres

Your `POSTGRES_DSN` is wrong, or PostgreSQL is not running.

### `/health` shows degraded redis

Your `REDIS_URL` is wrong, or Redis is not running.

### `run.py` still seems to use the old memory backend

Check:

- `api.use_api: true` in [config/memory.yaml](/home/petranokhin/projects/gigaevo_memory/gigaevo-core-internal/config/memory.yaml)
- `MEMORY_USE_API` is not overriding it to `false`
- `MEMORY_API_URL=http://localhost:8000` is set correctly

## 13. Minimal two-terminal workflow

### Terminal 1: backend

```bash
conda activate gigaevo-memory
cd /home/petranokhin/projects/gigaevo_memory/gigaevo-memory

export POSTGRES_DSN='postgresql+asyncpg://gigaevo:gigaevo@localhost:5432/gigaevo'
export REDIS_URL='redis://localhost:6379/0'
export ENABLE_VECTOR_SEARCH=true
export EMBEDDING_PROVIDER=sentencetransformers
export EMBEDDING_MODEL=all-MiniLM-L6-v2

cd api
alembic -c app/db/alembic.ini upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Terminal 2: core

```bash
conda activate <your-gigaevo-core-env>
cd /home/petranokhin/projects/gigaevo_memory/gigaevo-core-internal

export MEMORY_API_URL=http://localhost:8000
python run.py ideas_tracker=true memory_enabled=true
```

That is the intended end-to-end test path for the new backend.
