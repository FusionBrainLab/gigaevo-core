# Memory-v2 lifecycle tutorial

This tutorial follows one memory card through the current evolutionary system:
authoring, semantic deduplication, randomized selection, credit, and causal
retirement.

The authoritative overview is [memory.md](memory.md). The mathematical details
are in [memory_v2_bayesian_system_report.md](memory_v2_bayesian_system_report.md).

## 1. Start a run

Select the production memory arm:

```text
memory=v2
```

Useful ablations are:

```text
memory/applicability=none   # no agentic RAG applicability signal
memory/evictor=none        # keep causal selection, disable retirement
memory/write=end_of_run    # no live mid-run card creation
```

The default writer refreshes content and retirement every five successful live
ingestor sweeps. It does not modify old ledgers.

## 2. Freeze a decision

Before a mutation, the provider freezes:

- run, environment, parent, iteration, and generation;
- bounded reward semantics and fitness direction;
- MAP-Elites island, parent cell, archive state, and behavior coordinates;
- immutable revisions of every eligible live card;
- lineage exclusions and current selection leases;
- optional RAG applicability labels.

The candidate universe is the whole eligible bank. RAG assesses applicability
inside that universe; it is not a second candidate lane.

The fitted posterior produces immediate utility, delayed lineage option value,
and invalidity risk for each card. The policy removes actions that fail the
safety chance constraint, samples from coherent posterior worlds, reserves an
exploration floor, and may propose a card.

It then randomizes the offer:

- treatment: render the proposed card into the mutator prompt;
- control: render no card, while retaining which action was proposed.

Both proposal and offer propensities are persisted. This makes the control
outcome usable causal evidence rather than an accidental empty retrieval.

## 3. Protect the selected revision

If a card is offered, `InFlightSelectionRegistry` leases its bank id before the
mutation proceeds. The renderer receives the same immutable revision used by
the decision.

The lease moves from the mutation attempt to the child when the child is
created. It is released only when the terminal outcome has been synchronized.
Periodic retirement checks the live id and any historical aliases, so an
in-flight treatment cannot disappear before credit lands.

## 4. Record immediate and delayed outcomes

A terminal row records:

- whether treatment was delivered;
- which delivered card ids the mutator explicitly cited as used;
- whether the terminal is a valid outcome, invalid outcome, ineligible row, or
  censoring event;
- bounded direction-normalized gain when observed;
- frozen nuisance predictions and propensities;
- child/base linkage and archive disposition.

Immediate utility is the child outcome relative to its base parent.

Delivery and use are deliberately separate. The immutable ledger retains every
randomized delivery/withholding outcome for offer-policy ITT/OPE and uptake
diagnostics. The card-usefulness posterior admits withheld controls and cited
deliveries only. If a delivered card is absent from grounded `card_ids_used`,
that outcome has zero effect on its reward, safety, lineage, and retirement
posterior.

With the default lineage depth `3`, the system also waits for a fixed number of
same-island mutation opportunities and records the best non-negative,
archive-accepted descendant lift. This trains a separate lineage head; it never
silently inflates immediate reward.

## 5. Author an insight

The writer extracts only strictly valid mutation records. Invalid programs are
not sent to the author, so the author prompt contains no fictitious validity
degree of freedom.

For one eligible parent-to-child mutation, the card author sees:

- parent and child source;
- a bounded unified diff;
- the mutator report and its explanation;
- parent/child fitness and whether higher or lower is better;
- signed gain, already oriented so positive means improvement;
- archive status.

The response schema allows only:

```text
DROP | NEW
```

`DROP` returns no card. `NEW` returns exactly one conditional hypothesis:

```text
When observable condition C holds, try action A because mechanism M.
```

The author does not see the card bank and therefore cannot decide equivalence.
Raw mutation notes are never banked as a fallback.

A complete author/retrieval/equivalence chain has a wall-clock timeout. A failed
mutation record or exemplar is retried only up to
`writer.max_ingest_attempts` (default `3`), so a persistent timeout,
schema/context error, or store failure cannot consume one LLM call forever.

## 6. Retrieve and deduplicate the authored action

Only after authoring does the librarian build:

```text
card_brief = description + explanation_summary
```

It retrieves embedding neighbors from the same card kind. Ordinary memory-v2
keeps only the same task; `memory=v2_multitask` searches the whole shared bank.
The default write embedding scope contains those same semantic fields.

The equivalence judge sees the candidate and offered neighbor ids. Its schema
allows only:

```text
NEW | EQUIVALENT
```

For an `insight`, `EQUIVALENT` requires both:

1. materially the same applicability condition;
2. materially the same intervention.

Shared topic, mechanism, objective, vocabulary, or expected effect is
insufficient. Broader/narrower conditions are not equivalent.

For a `program`, `EQUIVALENT` instead means the same load-bearing strategy
family under materially the same applicability condition: materially the same
representation or state, core procedure, decision logic, update or output
policy, and essential constraints. Seeds, constants, ordinary hyperparameters,
resource budgets, batching or scheduling details, and interchangeable
supporting plumbing do not split a family.

For `EQUIVALENT`, the existing treatment prose remains immutable. The gate pools
provenance, task-labelled gain events, and task/run-labelled randomized trials,
and records the incoming candidate description for audit. Cross-task program
fitness is never used to replace the canonical representative. There is no
`MERGE`, union prose, or later consolidation scheduler.

For `NEW`, the card is admitted. The optional novelty judge may reject a
prior-obvious insight before admission. Equivalence and novelty failures
currently fail open to `NEW`; this keeps evolution running but means experiment
analysis must measure residual duplicates.

## 7. Author program strategy families

The writer also selects a bounded number of strong programs and asks the program
author for one holistic strategy hypothesis. The response is again
`DROP | NEW`.

Program candidates pass through the same authored-action retrieval and
kind-aware equivalence protocol. Equivalent strategy families keep the better
concrete representative under the configured fitness direction. The default
bank cap is 32 same-task program families.

By default program source is not stored. There is no unused source hash.

## 8. Sweep causal retirement

After content and evidence synchronization, the causal evictor snapshots:

- the current card bank;
- the exact causal evidence version;
- all immediate and delayed pending counts.

For each card lineage it requires:

- minimum cited-treatment support;
- minimum pooled controls;
- support across distinct assessable `(island, MAP-Elites parent cell)` contexts;
- no pending immediate or lineage outcomes.

It fits the current hierarchical posterior over the current bank. Retirement
fails closed if either reward head:

- fails optimization;
- reaches a configured hyperparameter boundary;
- places too much residual-scale mass at a numerical boundary.

For each supported context it predicts two optimistic semantic states:

- RAG unassessed;
- RAG applicable.

The selection policy calls any positive effect helpful. Retirement instead asks
whether effect exceeds a low quantile (default 10th percentile) of the non-zero
absolute normalized gains in this task's randomized control arm. Controls keep
the scale independent of the card effect under judgment, and the boundary
follows realized mutation dynamics rather than configured metric width. A
supported neutral card can eventually retire, while an uncertain card retains
exploration value. Exact zeros are excluded so the quantile measures a
non-trivial step; it is trusted only with the configured number of measured
non-zero controls. Otherwise the boundary falls back to zero.

A supported context whose remaining feasible positive headroom does not exceed
that threshold cannot certify uselessness or count toward context support. Its
posterior is still evaluated: an optimistic keep-vote rescues the card, while a
non-viable verdict from that clipped context is ignored. Confidently harmful
cards can still retire when enough assessable contexts exist.

The default normalized residual-scale floor is `0.01`. A lower-noise task can
push posterior mass onto that boundary; retirement then fails-keep and warns
with the startup-validated
`memory.posterior_config.reward_residual_sd_bounds` knob.

For every context/state pair, the evictor computes a Wilson upper bound over the
posterior Monte Carlo estimate of:

```text
P(safe and practically useful)
```

Only when every upper bound is at or below the configured viability probability
does it emit a one-use verdict.

## 9. Revalidate and delete

The verdict contains the exact:

- evidence version;
- immutable card revision.

Inside the card-store update, the admission gate:

1. consumes the verdict;
2. rechecks the ledger version and card revision;
3. rejects deletion if the live id or a historical alias is leased;
4. rejects deletion if supported foreign-task gain evidence is net helpful;
5. deletes the card;
6. records the exact retired id and write-ledger reason;
7. tombstones the id and exact authored payload for the rest of the run.

A stale verdict is never reused as an admission filter.

The causal SQLite ledger and JSON bank do not share a transaction. A terminal
could theoretically commit in the small interval after the final evidence
version read and before deletion. All earlier checks fail closed; removing that
last interval would require one shared transactional store.

## 10. Interpret retained cards correctly

A card remaining in the bank does not necessarily mean it is good:

- it may be uncertain;
- it may lack cited-treatment/control support;
- all observations may come from one MAP-Elites cell;
- a lineage outcome may still be pending;
- posterior numerical checks may have failed;
- it may be useful in another task;
- it may have stopped being cited before use support accumulated.

Likewise, censored outcomes do not train the reward or safety head. This is
conservative—cards are retained—but assumes censoring is conditionally
non-informative.

Retirement is re-evaluated after new evidence. “One-use verdict” means one
verdict cannot be consumed twice; it does not mean the card is tested only once
over its lifetime.

## 11. Inspect a run

Start with:

```text
<checkpoint>/cards.json
<checkpoint>/write_ledger.jsonl
<checkpoint>/memory_v2_selection_evidence.sqlite3
<checkpoint>/memory_events.jsonl
<checkpoint>/ope_summary.json
```

Questions to answer:

| Question | Evidence |
|---|---|
| Was the whole bank eligible? | decision candidate snapshots |
| What did RAG assess? | applicability assessment and candidate labels |
| Which card was proposed/offered? | decision record and propensities |
| Did the child receive and cite it? | immutable render metadata, frozen mutation assignment, and terminal |
| Was reward immediate or delayed? | terminal versus lineage observations |
| Did authoring drop, add, or deduplicate? | write ledger and authored-candidate description |
| Why was a card retained? | support/pending/numerical retirement diagnostics |
| Which ids were removed? | `MEMORY_V2_WRITER_SYNC.retired_card_ids` and `evicted` rows |
| Was retrieval useful overall? | offer-policy ITT/OPE, uptake rate, and cited-use posterior summaries |

Do not infer RAG or memory value from card count, embedding distance, or selected
examples alone. The randomized ledger and its treatment/control statistics are
the source of truth.
