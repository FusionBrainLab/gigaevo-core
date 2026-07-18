# Memory — Run Guide

Canonical guide: [`docs/memory.md`](docs/memory.md) (arm matrix, config
reference, card anatomy, observability, workflows). Package internals:
[`gigaevo/memory/README.md`](gigaevo/memory/README.md).

## Quick launch

```bash
# Full memory within the same run (read + live writes, one shared bank):
python run.py problem.name=heilbron pipeline=memory_guided memory=v2

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=guided memory=none

# Two-pass: build a bank on one run, reuse it on the next. Both runs use
# memory=v2 and share checkpoint_dir; memory=v2 reads and refreshes the bank.
python run.py problem.name=heilbron pipeline=guided memory=v2 \
    checkpoint_dir=outputs/memory_bank_01
python run.py problem.name=heilbron pipeline=memory_guided memory=v2 \
    checkpoint_dir=outputs/memory_bank_01
```

`memory={none,v2}` is one Hydra knob; `memory=v2` swaps in the causal-bandit
provider + live writer, while `memory/write={none,end_of_run,live}` chooses
write cadence. `pipeline=guided` never reads external memory cards;
`pipeline=memory_guided` does.

The memory subsystem has its own LLM route. The default `memory/llm=gemini`
uses OpenRouter and reads `OPENROUTER_API_KEY`; in-cluster/local setups can use
`memory/llm=qwen_instruct`, which reads `OPENROUTER_API_KEY` and targets the
configured LiteLLM-compatible proxy.

## Hydra groups

- Pipeline: [`config/pipeline/`](config/pipeline/) — `guided`, `memory_guided`, ...
- Program format: [`config/program_format/`](config/program_format/) — `python_source`, `json_document`
- Memory arms: [`config/memory/`](config/memory/) — `none`, `v2`
- Components: [`config/memory/`](config/memory/) subgroups — `llm`, `write`,
  `excluder`, `evictor`, `applicability`, `context`

## Platform / API-backed memory (removed)

The remote `gigaevo-memory` (Postgres + pgvector) backend was removed, along
with its `RemoteMemoryStore` skeleton (`gigaevo/memory/storage/remote.py`); only
the local backend remains. See [`README_memory_platform_run.md`](README_memory_platform_run.md)
for the tombstone.
