# Memory — Run Guide

Canonical guide: [`docs/memory.md`](docs/memory.md) (arm matrix, config
reference, card anatomy, observability, workflows). Package internals:
[`gigaevo/memory/README.md`](gigaevo/memory/README.md).

## Quick launch

```bash
# Full memory (read + write, one shared bank):
python run.py problem.name=heilbron pipeline=intra_extra_memory memory=full

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=standard memory=none

# Two-pass: build a bank, then read it
python run.py problem.name=heilbron pipeline=standard memory=writer \
    checkpoint_dir=outputs/memory_bank_01
python run.py problem.name=heilbron pipeline=standard memory=reader \
    checkpoint_dir=outputs/memory_bank_01

# Static curated levers (no bank, no memory LLM):
python run.py problem.name=heilbron pipeline=intra_extra_memory memory=static \
    memory.provider.levers_file=/abs/path/levers.md post_step_hook=null
```

`memory={none,reader,writer,full,static}` is one Hydra knob; arms swap
`_target_`s, there are no enable flags. Under `pipeline=intra_extra_memory`
the writer-off arms (`memory=none`, `memory=reader`) fail fast at startup —
the `LiveMemoryRefreshHook` needs a real writer.

## Hydra groups

- Pipeline: [`config/pipeline/`](config/pipeline/) — `intra_extra_memory`, `standard`, ...
- Memory arms: [`config/memory/`](config/memory/) — `none`, `reader`, `writer`, `full`, `static`
- Components: [`config/memory/`](config/memory/) subgroups — `llm`, `reputation`,
  `auction`, `budget`, `excluder`, `evictor`

## Platform / API-backed memory (removed)

The remote `gigaevo-memory` (Postgres + pgvector) backend was removed; only
the local backend remains (`RemoteMemoryStore` is a skeleton awaiting the
remote port). See [`README_memory_platform_run.md`](README_memory_platform_run.md)
for the tombstone.
