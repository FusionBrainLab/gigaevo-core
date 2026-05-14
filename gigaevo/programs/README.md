# Programs

`Program` is the unit of evolution: a Python function plus its lifecycle
state, metrics, lineage, and stage results. See `program.py` for the schema
and `program_state.py` for the state machine.

## Canonical vocabulary

Several counters used to travel under different names. The repository is
incrementally canonicalising them (issue #232). The names below are
authoritative — use them in new code, plots, manifests, and log lines.

### `iteration` (canonical)

The **program-ordinal counter** for an evolutionary run.

- Lives on `Program.iteration` (`gigaevo/programs/program.py`).
- Mirrored from `EngineMetrics.iteration`
  (`gigaevo/evolution/engine/metrics.py`), which the engine increments
  exactly once per successful `generate_mutations` call, before DAG
  evaluation.
- Initial (seed) programs have `iteration = 0`; each newly produced
  program gets the next value in monotone order.
- Single source of truth for "how far through the run are we?".

**Do not introduce new names** for this quantity. If a stage, dashboard,
log line, or stored event already uses an older name (`total_mutants`,
`step`, `s`, ...), prefer migrating it to `iteration` rather than adding
yet another alias.

### `lineage.generation` (distinct concept)

The **depth of a program in the parent graph**, not a global counter.

- Lives on `Program.lineage.generation` (`gigaevo/programs/program.py`).
- Seed programs are at the root depth; each mutation adds one to the
  maximum parent depth.
- Bounded by `GENESIS_GENERATION` (`gigaevo/programs/constants.py`).

`lineage.generation` and `iteration` are independent: two programs with
the same lineage depth are typically created at very different
iterations, and vice versa.

## Known follow-ups under #232

The rename is being landed in slices to keep each PR easy to review.
The following overloads of `total_mutants` / "generation" still live on
`main` and will be canonicalised in subsequent PRs:

- `EngineSnapshot.total_mutants`
  (`gigaevo/evolution/engine/snapshot.py`) — the persistence mirror of
  `EngineMetrics.iteration`. The on-disk Redis field name is held stable
  for now to avoid an artifact migration; the in-memory rename will
  land separately.
- `StopContext.total_mutants` (`gigaevo/evolution/engine/stopper.py`) —
  the stopper input wired from `EngineMetrics.iteration`. Renames with
  the snapshot field.
- Tracker `step` resolution
  (`gigaevo/utils/trackers/backends/{redis,tensorboard,wandb}.py`) —
  often resolves to `iteration` but is not labelled as such.
- Dataframe `s` field (`gigaevo/utils/dataframes.py`) — same value,
  shorter alias kept for storage compactness; to be renamed or
  documented as a derived alias.
- `CollectorStage` carries both `iteration` and `generation`
  (`gigaevo/programs/stages/collector.py`); the latter refers to
  `lineage.generation` and the column needs clearer naming.
- Multi-island global counter named `generation`
  (`gigaevo/evolution/strategies/multi_island.py`) — collides with
  `lineage.generation` and should move to `epoch` / `loop_iter` or
  similar.

When you touch any of the above, prefer fixing the name in the same PR
that introduces unrelated changes to the file — but do not bundle the
multi-island rename with the engine-side ones; the conceptual split is
load-bearing for the review.
