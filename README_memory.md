# Memory + Ideas Tracker — Run Guide

Two scenarios are supported today: a **single-pass live pipeline**
(`intra_extra_memory`) that reads and writes memory in the same run, and
an older **two-pass build-then-use** flow that still works for any run
that wires a read-side provider (`memory=reader` or `memory=full`).

## Required environment

The default memory LLM (`memory/common/llm=gemini`,
`google/gemini-3-flash-preview`) and the agentic memory retrieval both
reach OpenRouter, so both flows need an OpenRouter key. Without it the
GAM and librarian calls 401 silently and zero cards are written:

```bash
export OPENROUTER_API_KEY=sk-or-...
export HTTPS_PROXY=http://...           # if your egress is proxied
```

## Scenario A — Single-pass live intra/extra memory (recommended)

The `intra_extra_memory` pipeline runs the mutator with an intra-process
card store that the `LiveMemoryRefreshHook` keeps in sync with the
end-of-run librarian write path, so reading and writing share state via
the tracker's `_run_lock`.

```bash
OPENAI_API_KEY=sk-gigaevo python run.py \
  problem.name=heilbron \
  llm_base_url=http://INTERNAL_IP:4000 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  pipeline=intra_extra_memory \
  memory=full \
  num_parents=1 \
  redis.db=10
```

Hydra group co-overrides `memory=full num_parents=1` are **required**.
`memory=full` turns *both* sides on (reader + writer share one card bank).
Under `pipeline=intra_extra_memory` the writer-off presets (`memory=none`,
`memory=reader`) **fail fast at startup** — the `LiveMemoryRefreshHook`
needs a real tracker.

## Scenario B — Two-pass build cards, then read

Useful when you want a clean reusable card bank and a separate
evolution run that consumes it.

```bash
# 1. Build memory bank (writer on, reader off — cards written, never injected)
python run.py problem.name=heilbron memory=writer \
  checkpoint_dir=outputs/memory_bank_01

# 2. Run with the read path only, pointing at the same dir
python run.py problem.name=heilbron memory=reader \
  checkpoint_dir=outputs/memory_bank_01
```

After step 1 the run folder contains `write_ledger.jsonl` — one
append-only row per ingest/eviction verdict (`added` / `updated` /
`merged` / `discarded` / `rejected_harm` / `evicted`).

## How `checkpoint_dir` is applied

- `memory=reader` (or `memory=full`): used as `paths.checkpoint_dir` for the
  memory backend during the run (read/update of checkpointed memory state).
- `memory=writer` (or `memory=full`) with `memory_write_enabled: true`: the same
  `checkpoint_dir` is used by the final write step to persist cards.

## Hydra groups

- Pipeline: [`config/pipeline/`](config/pipeline/) — `intra_extra_memory`, `standard`, ...
- Memory: [`config/memory/`](config/memory/) — `none`, `reader`, `writer`, `full`
- Memory LLM: [`config/memory/common/llm/`](config/memory/common/llm/) — `gemini` (default), `qwen_instruct`

## Paper arm matrix

The three experiment arms differ only in Hydra group overrides. Every run
logs a startup `[Memory][Arm]` banner — verify the arm from the first
generation's log, not from `.hydra/config.yaml` archaeology.

| Arm | Overrides | Active components | Gen-1 log verification |
|---|---|---|---|
| Intra-only baseline | `pipeline=standard memory=none` | intra stage only; no tracker, no read provider | `[Memory][Arm]` banner shows `Null`/`None` provider and hook |
| Write-side-controlled baseline | `pipeline=intra_extra_memory memory=writer` | tracker writes cards; read path Null | banner + `[Memory][Arm] read path DISABLED` WARNING |
| Full memory | `pipeline=intra_extra_memory memory=full` | tracker + read provider + auction → 1 card | banner + `[Memory][Exposure] FIRST_INJECTION` once the bank is non-empty |

The write-side-controlled baseline is state-pure: the tracker never touches
program metadata or Redis engine state. It is timing-dirty only under
wall-clock stoppers — fine with `max_mutants`.

## Platform / API-backed memory (removed)

The remote `gigaevo-memory` (Postgres + pgvector) backend was removed in the
one-knob config collapse; only the local backend remains. See
[`README_memory_platform_run.md`](README_memory_platform_run.md) for the
tombstone.
