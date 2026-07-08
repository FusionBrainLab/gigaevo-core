# Memory — Run Guide

Canonical guide: [`docs/memory.md`](docs/memory.md) (arm matrix, config
reference, card anatomy, observability, workflows). Package internals:
[`gigaevo/memory/README.md`](gigaevo/memory/README.md).

## Quick launch

```bash
# Full memory within the same run (read + live writes, one shared bank):
python run.py problem.name=heilbron pipeline=memory_guided memory=full memory/write=live

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=guided memory=none

# Two-pass: build a bank, then read it
python run.py problem.name=heilbron pipeline=guided memory=writer \
    checkpoint_dir=outputs/memory_bank_01
python run.py problem.name=heilbron pipeline=memory_guided memory=reader \
    checkpoint_dir=outputs/memory_bank_01

# Static curated levers (no bank, no memory LLM):
python run.py problem.name=heilbron pipeline=memory_guided memory=static \
    memory.provider.levers_file=/abs/path/levers.md
```

`memory={none,reader,writer,full,static}` is one Hydra knob; arms swap
`_target_`s, while `memory/write={none,end_of_run,live}` chooses write cadence.
`pipeline=guided` never reads external memory cards; `pipeline=memory_guided`
does.

The memory subsystem has its own LLM route. The default `memory/llm=gemini`
uses OpenRouter and reads `OPENROUTER_API_KEY`; in-cluster/local setups can use
`memory/llm=qwen_instruct`, which reads `OPENAI_API_KEY` and targets the
configured LiteLLM-compatible proxy.

## Hydra groups

- Pipeline: [`config/pipeline/`](config/pipeline/) — `guided`, `memory_guided`, ...
- Program format: [`config/program_format/`](config/program_format/) — `python_source`, `json_document`
- Memory arms: [`config/memory/`](config/memory/) — `none`, `reader`, `writer`, `full`, `static`
- Components: [`config/memory/`](config/memory/) subgroups — `llm`, `reputation`,
  `auction`, `budget`, `excluder`, `evictor`

## Platform / API-backed memory (removed)

The remote `gigaevo-memory` (Postgres + pgvector) backend was removed; only
the local backend remains (`RemoteMemoryStore` is a skeleton awaiting the
remote port). See [`README_memory_platform_run.md`](README_memory_platform_run.md)
for the tombstone.
