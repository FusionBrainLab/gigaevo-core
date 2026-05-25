# Memory + Ideas Tracker — Run Guide

Two scenarios are supported today: a **single-pass live pipeline**
(`intra_extra_memory`) that reads and writes memory in the same run, and
an older **two-pass build-then-use** flow that still works for any pipeline
that wires a `MemorySelectorAgent`.

## Required environment

Both flows need an OpenRouter key for the IdeaTracker analyzers
(`google/gemini-3-flash-preview`) and any agentic memory retrieval.
Without it the GAM/IdeaTracker calls 401 silently and zero cards are
written:

```bash
export OPENROUTER_API_KEY=sk-or-...
export HTTPS_PROXY=http://...           # if your egress is proxied
```

## Scenario A — Single-pass live intra/extra memory (recommended)

The `intra_extra_memory` pipeline runs the mutator with an intra-process
card store that the `LiveMemoryRefreshHook` keeps in sync with the
end-of-run write pipeline, so reading and writing share state via the
tracker's `_run_lock`.

```bash
OPENAI_API_KEY=sk-gigaevo python run.py \
  problem.name=heilbron \
  llm_base_url=http://INTERNAL_IP:4000 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  pipeline=intra_extra_memory \
  ideas_tracker=default \
  memory=local \
  num_parents=1 \
  redis.db=10
```

Hydra group co-overrides `ideas_tracker=default memory=local num_parents=1`
are **required** — `pipeline=intra_extra_memory` alone silently falls
back to `Null*` providers.

## Scenario B — Two-pass build cards, then read

Useful when you want a clean reusable card bank and a separate
evolution run that consumes it.

```bash
# 1. Build memory bank (no memory read in evolution)
python run.py problem.name=heilbron ideas_tracker=default \
  checkpoint_dir=outputs/memory_bank_01

# 2. Run with memory enabled, pointing at the same dir
python run.py problem.name=heilbron memory=local \
  checkpoint_dir=outputs/memory_bank_01
```

After step 1 the run folder contains `memory_write_stats.json` with
per-run `updated` / `rejected` counts.

## How `checkpoint_dir` is applied

- `memory=local` (or `memory=api`): used as `paths.checkpoint_dir` for the
  memory backend during the run (read/update of checkpointed memory state).
- `ideas_tracker=default` with `memory_write_enabled: true`: the same
  `checkpoint_dir` is used by the final write step to persist cards.

## Hydra groups

- Pipeline: [`config/pipeline/`](config/pipeline/) — `intra_extra_memory`, `standard`, ...
- Memory backend: [`config/memory/`](config/memory/) — `local`, `api`, `none`
- Ideas tracker: [`config/ideas_tracker/`](config/ideas_tracker/) — `default`, `fast`, `true` (alias), `none`

## Platform / API-backed memory

For the remote `gigaevo-memory` backend (Postgres + pgvector), see
[`README_memory_platform_run.md`](README_memory_platform_run.md).
