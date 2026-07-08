# Memory-Guided Pipeline (`pipeline=memory_guided`)

`pipeline=memory_guided` is the guided mutation DAG plus external memory-card
retrieval. It reads cards; it does not decide write cadence.

Use it when the mutation suggester should see cross-run memory cards:

```bash
python run.py problem.name=heilbron pipeline=memory_guided memory=reader \
  checkpoint_dir=/data/banks/heilbron
```

For read + write against one shared bank:

```bash
python run.py problem.name=heilbron pipeline=memory_guided memory=full
```

For live mid-run writes:

```bash
python run.py problem.name=heilbron pipeline=memory_guided memory=full memory/write=live
```

## DAG Contract

`MemoryContextStage` selects cards through `memory.provider` and feeds them only
to `MutationSuggestionStage.memory_cards`. Raw memory cards do not go directly
into `MutationContextStage.memory`; the mutation agent sees structured
suggestions produced from the cards.

The per-parent history summary still feeds `MutationContextStage.memory`, same
as `pipeline=guided`.

## Advanced Controls

These are experiment and prompt-placement controls. Most users should leave
them at defaults.

| Setting | Default | Meaning |
|---|---:|---|
| `pipeline_builder.fresh_context_reorder` | `true` | Select cards using the fresh parent-history summary and current evolutionary stats from this DAG pass. `false` uses older metadata context. |
| `pipeline_builder.reverse_repack` | `false` | Reverse rendered card order so the strongest selected card is closest to the final instruction. |
| `pipeline_builder.no_card_control_probability` | `0.10` | Withhold selected cards from this fraction of otherwise card-eligible mutations to create fair no-card controls for reputation. |
| `pipeline_builder.memory_block_last` | `false` | Move the memory block later in the mutation context, closer to the final mutation instruction. |

## Read/Write Split

Reading cards:

```bash
pipeline=memory_guided memory=reader checkpoint_dir=/data/banks/task
```

Writing cards:

```bash
pipeline=guided memory=writer checkpoint_dir=/data/banks/task
```

Reading and writing:

```bash
pipeline=memory_guided memory=full checkpoint_dir=/data/banks/task
```

Live writing:

```bash
pipeline=memory_guided memory=full memory/write=live checkpoint_dir=/data/banks/task
```

