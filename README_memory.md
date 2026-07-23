# Memory — Run Guide

Canonical guide: [`docs/memory.md`](docs/memory.md) (arm matrix, config
reference, card anatomy, observability, workflows). Package internals:
[`gigaevo/memory/README.md`](gigaevo/memory/README.md).

## Quick launch

```bash
# Full memory within the same run (read + live writes, one shared bank):
python run.py problem.name=heilbron pipeline=memory_guided memory=v2

# Fixed shared card set: read existing cards and append causal trial evidence,
# but never author insight or program cards.
python run.py problem.name=heilbron \
    pipeline=memory_guided \
    memory=v2_multitask \
    memory_bank_dir=/absolute/path/to/shared_memory_bank \
    memory.writer.authoring_enabled=false

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=guided memory=none

# Two-pass: populate the shared bank, then freeze its card set for the next run.
# Share only memory_bank_dir; leave checkpoint_dir run-specific.
python run.py problem.name=heilbron pipeline=memory_guided \
    memory=v2_multitask memory_bank_dir=/absolute/path/to/shared_memory_bank
python run.py problem.name=heilbron pipeline=memory_guided \
    memory=v2_multitask memory_bank_dir=/absolute/path/to/shared_memory_bank \
    memory.writer.authoring_enabled=false
```

`memory={none,v2}` is one Hydra knob; `memory=v2` swaps in the causal-bandit
provider + live writer, while `memory/write={none,end_of_run,live}` chooses
write cadence. `pipeline=guided` never reads external memory cards;
`pipeline=memory_guided` does.

For a fixed shared card set with feedback, keep `memory/write=live` and set
`memory.writer.authoring_enabled=false`. This skips the task-summary, card
author, equivalence, and program-exemplar paths, while the updater still
releases completed selection leases and appends deduplicated `use_trials` to
existing cards. The `v2_multitask` preset also selects `memory/evictor=none`, so
cards are not retired.

This is authoring-disabled, not filesystem read-only: `cards.json` and
`selection_leases.json` must remain writable. `memory/write=none` removes the
updater entirely, so it does not stamp shared trials or promptly release
completed-child leases. Start from a populated bank; an empty fixed bank leaves
the policy with no card actions.

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
