# Memory + Ideas Tracker Run Guide

## Quick start (2 runs)

This is the recommended order:

1. Run without memory, but with ideas tracker, to write memory cards.
2. Run with memory enabled, using the same checkpoint folder as source.

Before step 1, ensure this is enabled in `config/memory.yaml`:

```yaml
ideas_tracker:
  memory_write_pipeline:
    enabled: true
```

### Step 1: build memory cards (no memory in evolution yet)

```bash
python run.py \
  problem.name=heilbron \
  ideas_tracker=true \
  checkpoint_dir=outputs/memory_bank_01
```

### Step 2: run with memory enabled (read from that folder)

```bash
python run.py \
  problem.name=heilbron \
  memory_enabled=true \
  checkpoint_dir=outputs/memory_bank_01
```

## How checkpoint_dir is applied

- If `memory_enabled=true`, `checkpoint_dir` is used as `paths.checkpoint_dir` for the memory GAM backend during the run (this is where it reads/updates checkpointed memory state).
- If `ideas_tracker=true` and `ideas_tracker.memory_write_pipeline.enabled=true`, the same `checkpoint_dir` is used by ideas tracker final write step to store cards into memory DB pipeline.


