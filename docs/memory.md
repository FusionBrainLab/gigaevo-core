# Evolutionary memory

The supported memory modes are:

- `memory=none`: no external-memory read or write path.
- `memory=v2`: live card authoring plus randomized, contextual Bayesian
  selection and causal retirement.

`memory=v2` is the production design. It does not use the removed v1
reputation/auction/reconcile/consolidation stack.

## System shape

One shared `LocalMemoryStore` owns the durable card bank and its process-local
vector index. The read and write paths use that same store instance.

```text
READ
whole live bank
  -> lineage exclusion
  -> optional agentic applicability assessment
  -> contextual posterior and chance-constrained probability matching
  -> randomized offer/control action
  -> immutable render + selection lease
  -> terminal and delayed-lineage outcomes in the causal ledger

WRITE
strictly valid mutation outcomes
  -> author DROP or at most one conditional hypothesis
  -> retrieve same-kind neighbors in the configured task or whole-bank scope
  -> strict NEW or EQUIVALENT judgment
  -> admit new treatment or pool evidence into the exact equivalent
  -> author bounded program-strategy exemplars
  -> synchronize causal evidence and leases
  -> periodic conservative causal retirement
```

The reader never assigns efficacy from prose, embedding similarity, retrieval,
or the mutator explanation. Those signals generate or describe actions.
The full randomized offer ledger supports offer-policy ITT/OPE. Card usefulness
is learned only from withheld controls and delivered cards explicitly named in
the mutator's grounded `card_ids_used` output. A delivered-but-uncited card
remains an offer/uptake observation but contributes nothing to reward,
invalidity, delayed-lineage, or retirement inference.

## Card model

There are two card kinds:

- `insight`: a mutation hypothesis distilled from one parent-to-child outcome.
- `program`: a holistic strategy family distilled from a strong concrete
  program. Equivalent families keep the best representative.

Every authored card contains:

- `description`: one conditional treatment hypothesis;
- `explanation_summary`: its proposed mechanism;
- task identity and task-description context;
- provenance program ids;
- causal/historical lineage metadata where applicable.

The author is instructed to use the semantic form:

> When observable condition C holds, try action A because mechanism M.

Cards are hypotheses, not proven truths. There are no free-form keywords,
content-union merges, or program-source hashes. Program source is retained only
when `program_exemplars.store_code=true`; the default is metadata plus the
authored strategy.

The write-neighbor query is `card_brief(description + explanation_summary)`.
The default `desc_expl` embedding scope indexes the same semantic fields.
Embedding proximity only proposes neighbors. The LLM equivalence judge requires
the same intervention and applicability condition for insight cards, or the
same strategy-family applicability condition, load-bearing representation or
state, core procedure, decision logic, update or output policy, and essential
constraints for program cards while ignoring incidental implementation
variants.

## Librarian protocol

### Insight authoring

The author sees:

- parent and child programs;
- their bounded unified diff;
- the mutator change report and explanation;
- parent fitness, child fitness, and fitness direction;
- direction-normalized signed gain;
- archive disposition.

Invalid children never enter this path: strict metric validity is an extractor
precondition, so validity is not a dead prompt field. The response schema itself
permits only `DROP | NEW`. `DROP` has no card; `NEW` has exactly one card.

Transient or deterministic authoring failures never bank raw mutation notes.
Each mutation record and exemplar is attempted at most
`writer.max_ingest_attempts` times (default `3`) before it is dropped for the
remainder of the run. `writer.ingest_call_timeout_s` bounds the complete
author/retrieval/equivalence chain for one ingest.

### Equivalence

Only after a candidate is authored does the librarian retrieve neighbors. The
equivalence response schema permits only:

- `NEW`: no offered neighbor is equivalent under the card-kind rule;
- `EQUIVALENT`: one offered id is the same insight intervention or program
  strategy family.

An equivalent candidate does not rewrite or broaden the banked treatment.
Provenance and founding evidence are pooled into the existing card. In
`memory=v2_multitask`, this lookup spans the whole bank, so equivalent actions
authored by different tasks share one canonical card. Gain events retain their
task context and randomized use trials retain their task/run labels. A program
representative is replaced using fitness only for same-task equivalents; raw
fitness values never compete across tasks. The write ledger retains the
discarded candidate description so equivalence mistakes can be audited.

Retrieval or equivalence failure currently fails open to `NEW`; the optional
novelty judge also fails open. This preserves evolution when the memory LLM is
unavailable, but an online dedup miss has no later exhaustive consolidation
pass. Experiments must therefore report residual semantic duplicates.

### Retirement tombstones

Causally retired ids and exact normalized authored payloads are tombstoned for
the rest of the process. This prevents deterministic retire/re-author loops
without a durable compatibility registry. A genuinely different formulation is
still eligible for ordinary authoring and causal evaluation.

## Causal selection

`CausalBanditMemoryProvider` considers the whole eligible bank. Agentic RAG, when
enabled, labels semantic applicability over that same universe; it does not own
a separate priority lane.

The policy:

1. freezes the pre-treatment evolutionary context and immutable card revisions;
2. fits immediate-utility, delayed-lineage-utility, and invalidity heads;
3. excludes actions that fail the configured chance constraint;
4. samples among the remaining actions from posterior worlds;
5. applies explicit proposal exploration;
6. randomizes whether the proposed action is actually offered;
7. persists proposal/offer/joint propensities and frozen nuisance predictions.

The control action is deliberate randomized evidence, not a retrieval failure.
One mutation attempt owns one decision and one terminal. A selection lease
protects every offered live card until the corresponding attempt/child is
resolved.

At child birth, the system intersects `card_ids_used` with the immutable
prompt-time slate and freezes the result on the mutation assignment. Missing,
malformed, or hallucinated ids receive no use credit. The terminal carries that
frozen set into the causal ledger; later metadata changes cannot alter it.

Lineage credit is separate from immediate credit. The default depth is `3`;
only non-negative archive-accepted best-descendant lift after the configured
same-island opportunity budget trains the delayed head.

## Causal retirement

The default `memory/evictor=causal` arm runs during the live writer cadence.
`memory/evictor=none` is the supported retirement ablation.

A card can be proposed for deletion only when all of the following hold:

- its lineage has at least the configured cited-treatment support;
- the ledger has the configured pooled randomized controls;
- support spans the configured number of assessable discrete
  `(MAP-Elites island, parent cell)` contexts;
- no immediate or delayed-lineage outcome is pending;
- both reward heads optimized successfully;
- neither reward head is at a hyperparameter boundary;
- neither reward head has excessive residual-scale boundary mass;
- prediction and deterministic safety integration remain numerically certified;
- under both `UNASSESSED` and optimistic `APPLICABLE` RAG states, the Wilson
  Monte Carlo upper bound for
  `P(safe and practically useful)` is below the retirement threshold in every
  supported context.

Selection defines helpfulness relative to zero. Retirement instead uses
`practical_effect_quantile`: a low quantile (default `0.10`) of non-zero
absolute normalized gains in the randomized control arm. Using controls keeps
the practical scale independent of the card effect being judged, and follows
realized task dynamics rather than the problem author's chosen metric bounds.
The quantile is used only after `min_global_control` measured, non-zero control
magnitudes exist. Exact zeros are intentionally excluded, so this is the scale
of a non-trivial realized step rather than a zero-inflated quantile; with
insufficient support the boundary falls back to zero, preserving harm
retirement while deferring neutral-card retirement.

A context whose remaining feasible positive headroom does not clear the
boundary cannot certify deletion and does not count toward
`min_distinct_contexts`, but its posterior is still evaluated and any optimistic
keep-vote rescues the card. Sparse support, uncertainty, and numerical-boundary
mass all fail-keep.

The default normalized residual lower bound is `0.01`. On tasks whose realized
noise lies below that floor, residual-boundary diagnostics disable retirement
and warn with the startup-validated
`memory.posterior_config.reward_residual_sd_bounds` knob. This is a loud,
operator-configurable fail-keep envelope; selection and writing continue.

The evictor creates a one-use verdict containing:

- the exact immutable card revision;
- the exact causal evidence version.

`CardAdmissionGate` immediately consumes that verdict inside the card-store
update, rechecks leases (including historical aliases), and applies the
foreign-task positive-evidence veto before deletion. A changed card or evidence
version rescues the card.

Production causal retirement requires `allow_cross_task=false`. Cross-task
delivery remains disabled until retirement evidence is identified separately
per source task; the admission veto is only a defensive final check.
With that setting, an empty query or card `task_key` is warned once and refused
rather than treated as a wildcard.
Consequently, legacy cards with an empty `task_key` are not candidates. Restamp
them with the correct task key, or set `allow_cross_task=true` only for a run
whose legacy bank is known to contain one compatible task.

The SQLite evidence ledger and JSON card bank are separate stores, so there is a
small residual interval between the final ledger-version read and the bank
delete that cannot be made atomic without a shared transaction. The design is
fail-closed before that interval and records exact retired ids for audit.

Repeated sweeps are sequential posterior checks, not independent fixed-sample
tests. Deterministic evidence-version RNG and the conservative Monte Carlo bound
avoid stochastic verdict churn, but model misspecification remains the main
sequential-testing risk.

Censored terminals are excluded from reward and safety fitting. This assumes
censoring is non-informative conditional on recorded context. A card that causes
hangs recorded only as censoring will therefore be retained rather than
declared safe or harmful.

Delivered-but-uncited terminals are also excluded from every usefulness head.
Cards that stop being cited before reaching minimum use support therefore
fail-keep. This is intentional: retirement never fabricates causal support from
exposure or staleness.

## Configuration

The canonical graph is `config/memory/v2.yaml`. Important component groups:

| Group | Values | Purpose |
|---|---|---|
| `memory/llm` | `gemini`, `qwen_instruct`, `gpt54_mini` | shared memory LLM router |
| `memory/applicability` | `agentic`, `none` | semantic RAG assessment or ablation |
| `memory/context` | `global` | task and MAP-Elites decision context |
| `memory/excluder` | `lineage`, `none` | prevent immediate lineage reuse |
| `memory/evictor` | `causal`, `none` | causal retirement or explicit ablation |
| `memory/no_card_evidence` | `none` | explicit absence of heuristic no-card evidence |
| `memory/write` | `live`, `end_of_run`, `none` | writer cadence |

The resolver-safe causal evictor couples its viability probability to
`SafetyConstraint.alpha` through `${ref:memory.safety::alpha}`. A production
composition test instantiates the complete node; merely checking `_target_` is
not sufficient.

## Persistence and observability

Reusable bank artifacts live under `memory_bank_dir` (which defaults to
`checkpoint_dir`):

| Artifact | Meaning |
|---|---|
| `cards.json` | authoritative card bank, including compact shared-card usefulness trials |
| `cards.json.lock` | short cross-process transaction for atomic card persistence |
| `cards.json.authoring.lock` | cross-process semantic retrieve → judge → admit transaction |
| `selection_leases.json` | in-flight card reservations for processes sharing the bank |

Parallel runs should share only `memory_bank_dir`; each run keeps its own
`checkpoint_dir` (the default Hydra layout already does this). Card persistence,
semantic admission, usefulness-trial updates, and selection leases are guarded
across processes. The vector index remains process-local and refreshes from the
authoritative bank after another process writes.

Run-local artifacts live under `checkpoint_dir`:

| Artifact | Meaning |
|---|---|
| `write_ledger.jsonl` | content/equivalence/rejection/retirement audit rows |
| `memory_v2_selection_evidence.sqlite3` | immutable decisions, terminals, mutation edges, and lineage outcomes |
| `memory_events.jsonl` | structured runtime events |
| `ope_summary.json` | probe-ITT/DR-AIPW policy summary when enough evidence exists |

Relevant outcomes include `added`, `updated`, `discarded`,
`rejected_retired`, `rejected_novelty`, `rejected_capacity`, `retired`, and
`evicted`. `discarded` is the only unledgered benign no-op.

`MEMORY_V2_WRITER_SYNC.retired_card_ids` is the exact retirement signal.
Assignment/outcome/applicability events and the causal SQLite rows are the
primary source for experiment analysis.

No migration, restamping, or stamping of old ledgers is performed.

## Validation

At minimum, changes to this subsystem should run:

```bash
pytest -q tests/llm tests/memory tests/memory_v2
ruff check .
ruff format --check .
python -m mypy gigaevo/ --ignore-missing-imports --no-error-summary
```

See [MEMORY_LIFECYCLE_TUTORIAL.md](MEMORY_LIFECYCLE_TUTORIAL.md) for a concrete
decision-to-retirement walkthrough and
[memory_v2_bayesian_system_report.md](memory_v2_bayesian_system_report.md) for
the statistical model.
