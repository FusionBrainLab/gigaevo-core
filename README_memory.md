# Memory + Ideas Tracker Run Guide

> **Status.** The memory provider is currently fixed to
> ``NullMemoryProvider`` on the typed CLI; enabling memory or the
> ideas-tracker PostRunHook requires editing the experiment file to
> construct the provider / hook and thread them through
> ``EvolutionContext`` and ``EvolutionEngine``. The two-step workflow
> below describes the runtime topology; treat the listed flags as the
> design surface, not as runnable CLI arguments.

## Quick start (2 runs)

This is the recommended order:

1. Run without memory, but with the ideas tracker, to write memory cards.
2. Run with memory enabled, using the same checkpoint folder as source.

Before step 1, ensure this is enabled in `config/memory.yaml` (the
backend config loaded by `gigaevo/memory/runtime_config.py`):

```yaml
ideas_tracker:
  memory_write_pipeline:
    enabled: true

card_update_dedup:
  enabled: true
```

### Step 1: build memory cards (no memory in evolution yet)

Construct an experiment file (e.g. `experiments/heilbron_ideas_writer.py`)
that attaches an `IdeaTracker` PostRunHook to the engine and sets
`checkpoint_dir` on the tracker constructor. Then:

```bash
python run.py experiments/heilbron_ideas_writer.py
```

After this step, the ideas-tracker run folder also includes
`memory_write_stats.json` with per-run write stats (including `updated`
and `rejected` counts).

### Step 2: run with memory enabled (read from that folder)

Construct an experiment file (e.g. `experiments/heilbron_memory_reader.py`)
that wires `SelectorMemoryProvider(checkpoint_dir="outputs/memory_bank_01", ...)`
into `EvolutionContext`. Then:

```bash
python run.py experiments/heilbron_memory_reader.py
```

## How checkpoint_dir is applied

- When the experiment wires `SelectorMemoryProvider`, `checkpoint_dir`
  is used as `paths.checkpoint_dir` for the memory GAM backend during
  the run (this is where it reads / updates checkpointed memory state).
- When the experiment wires an `IdeaTracker` with
  `memory_write_pipeline.enabled=true`, the same `checkpoint_dir` is
  used by the ideas-tracker final write step to store cards into the
  memory DB pipeline.
