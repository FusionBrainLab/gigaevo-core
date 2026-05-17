# Dataplane Phase 2 Substrate

This document enumerates dataplane primitives that ship today with no
production consumer. They are intentionally retained: each one is a
typed substrate that a Phase 2 feature is scheduled to build on. The
audit lane that trimmed speculative code spared them so a future
contributor sees one place naming both the present shape and the future
destination, instead of deleting them as dead and re-introducing them
later under a different name.

If you find yourself about to delete something on this list, first
verify the Phase 2 destination still holds; if it does, leave the
substrate alone.

## LWW register

- `DataPlane.lwwr_set` / `DataPlane.lwwr_get`
- `gigaevo/dataplane/scripts/lwwr_set.lua`

Present: typed last-writer-wins register with HLC tiebreak, fully
covered by `tests/test_lwwr.py`.

Phase 2 destination: single-value config / aggregate fields under a
stream-shaped consumer where concurrent writers race on the same key
and the resolution rule is "latest HLC wins". The script is the
substrate; the consumer wiring is the Phase 2 work.

## Hybrid logical clock

- `HlcTimestamp` in `gigaevo/dataplane/models.py`

Present: `(physical_ns, counter)` pair with packed-hex encoding and
strict lexicographic ordering.

Phase 2 destination: event-stream timestamp carried on every emitted
event for ordering without a global clock; Phase 4's distributed HLC
adds a node-id field that fits in the reserved-zero trailing pad of
the current packed-hex layout, so the format is forward-compatible.

## Lattice vocabulary

- `BoolLattice`, `EpochLattice`, `GenerationLattice`, `ProductLattice`,
  `MonotoneLattice` in `gigaevo/dataplane/lattices.py`

Present: the algebraic vocabulary that `Versioned` and `Freshness`
already use semantically — `EpochLattice` and `GenerationLattice` are
the join axes the `(epoch, generation)` product lattice composes;
`MonotoneLattice` is the generic shape behind retrograde-rejecting
counters.

Phase 2 destination: `BoolLattice` powers the "any worker has seen this
aggregate" admission gate (an OR-join over a set of per-worker
witnesses); `ProductLattice` formalises the freshness join that today
is open-coded inside `Versioned.combine_max`.

## NewType IDs

- In `gigaevo/dataplane/ids.py`: `AggregateId`, `BanditArm`,
  `CausationId`, `CorrelationId`, `StreamName`, `ConsumerGroup`,
  `ConsumerName`, `NodeId`, `IdempotencyToken`, `EventId`, `EpochId`,
  `GenerationId`, `StepId`

Present: zero-runtime-cost `NewType` aliases over `str` / `int`. Today
none are consumed outside the dataplane itself.

Phase 2 destination: the vocabulary for event-sourced primitives —
`AggregateId` / `EventId` / `CausationId` / `CorrelationId` on the
event-emit path, `StreamName` / `ConsumerGroup` / `ConsumerName` on
the consumer-group path, `NodeId` for distributed HLC, `EpochId` /
`GenerationId` / `StepId` for the per-aggregate freshness witnesses,
`IdempotencyToken` for the FSM idempotency surface (already used
internally by the coordinator and re-exported so Phase 2 callers can
annotate the token they receive). `BanditArm` reserves the arm-id
shape for the Phase 2 bandit-router admission gate.
