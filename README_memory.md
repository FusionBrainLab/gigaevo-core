# Memory + Ideas Tracker — Run Guide

Two scenarios are supported today: a **single-pass live pipeline**
(`intra_extra_memory`) that reads and writes memory in the same run, and
an older **two-pass build-then-use** flow that still works for any run
that wires a read-side provider (`memory=local`).

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
are **required**. Omitting them is loud, not silent: without
`ideas_tracker` the run fails at startup (the `LiveMemoryRefreshHook`
needs a tracker), and `memory=none` logs a
`[Memory][Arm] read path DISABLED` WARNING.

## Scenario B — Two-pass build cards, then read

Useful when you want a clean reusable card bank and a separate
evolution run that consumes it.

```bash
# 1. Build memory bank (no memory read in evolution)
python run.py problem.name=heilbron ideas_tracker=default \
  checkpoint_dir=outputs/memory_bank_01

# 2. Run with memory enabled, pointing at the same dir
#    (read path only: tracker + live-refresh hook off)
python run.py problem.name=heilbron pipeline=intra_extra_memory \
  memory=local ideas_tracker=none post_step_hook=null \
  checkpoint_dir=outputs/memory_bank_01
```

After step 1 the run folder contains `memory_write_stats.json` with
per-run `updated` / `rejected` counts.

## How `checkpoint_dir` is applied

- `memory=local` (or the legacy `memory=legacy_api`): used as
  `paths.checkpoint_dir` for the memory backend during the run (read/update
  of checkpointed memory state).
- `ideas_tracker=default` with `memory_write_enabled: true`: the same
  `checkpoint_dir` is used by the final write step to persist cards.

## Hydra groups

- Pipeline: [`config/pipeline/`](config/pipeline/) — `intra_extra_memory`, `standard`, ...
- Memory backend: [`config/memory/`](config/memory/) — `local`, `none`, `legacy_api`
- Ideas tracker: [`config/ideas_tracker/`](config/ideas_tracker/) — `default`, `fast`, `true` (alias), `none`

## Paper arm matrix

The three experiment arms differ only in Hydra group overrides. Every run
logs a startup `[Memory][Arm]` banner — verify the arm from the first
generation's log, not from `.hydra/config.yaml` archaeology.

| Arm | Overrides | Active components | Gen-1 log verification |
|---|---|---|---|
| Intra-only baseline | `pipeline=standard` | intra stage only; no tracker, no read provider | `[Memory][Arm]` banner shows `Null`/`None` provider and hook |
| Write-side-controlled baseline | `pipeline=intra_extra_memory ideas_tracker=default memory=none` | tracker writes cards; read path Null | banner + `[Memory][Arm] read path DISABLED (memory=none)` WARNING |
| Full memory | `pipeline=intra_extra_memory ideas_tracker=default memory=local` | tracker + read provider + auction → 1 card | banner + `[Memory][Exposure] FIRST_INJECTION` once the bank is non-empty |

The write-side-controlled baseline is state-pure: the tracker never touches
program metadata or Redis engine state. It is timing-dirty only under
wall-clock stoppers — fine with `max_mutants`.

## Platform / API-backed memory

For the remote `gigaevo-memory` backend (Postgres + pgvector), see
[`README_memory_platform_run.md`](README_memory_platform_run.md).
