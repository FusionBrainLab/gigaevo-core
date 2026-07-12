# Memory System

Cross-run memory for the evolutionary loop. A **write system** distills each
run's mutation diffs and top exemplars into reusable cards; a **read system**
researches that bank and injects the most promising cards into mutation
prompts, tracking each card's realized fitness gain (reputation).

Package-internals map: [`gigaevo/memory/README.md`](../gigaevo/memory/README.md).
The pipeline that consumes memory cards:
[`MEMORY_GUIDED_PIPELINE.md`](MEMORY_GUIDED_PIPELINE.md). For a diagram-heavy
end-to-end explanation of card birth, read selection, no-card baselines,
posteriors, decay, and eviction, see
[`MEMORY_LIFECYCLE_TUTORIAL.md`](MEMORY_LIFECYCLE_TUTORIAL.md).

## The 30-second version

```bash
# Memory ON within the same run (read + live writes against one bank):
python run.py problem.name=heilbron pipeline=memory_guided memory=full memory/write=live

# Read from a pre-built bank:
python run.py problem.name=heilbron pipeline=memory_guided memory=reader \
  checkpoint_dir=/data/banks/heilbron

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=guided memory=none
```

`memory={none,reader,writer,full,static}` chooses the read/write components.
`memory/write={none,end_of_run,live}` chooses write cadence. Pipelines read
`${ref:memory.provider}` only when `pipeline=memory_guided`; engines read
`${ref:memory.write.post_run_hook}` as their finalizer.

| Arm | provider (read side) | writer (write side) | What runs |
|---|---|---|---|
| `memory=none` | `NullMemoryProvider` | `NullPostRunHook` | nothing |
| `memory=reader` | `ReaderMemoryProvider` | `NullPostRunHook` | injects from a pre-built bank; no extraction |
| `memory=writer` | `NullMemoryProvider` | `MemoryWriter` | authors a bank for a *later* run; injects nothing |
| `memory=full` | `ReaderMemoryProvider` | `MemoryWriter` | reader + writer share one bank under `checkpoint_dir`; use `memory/write=live` for same-run memory effects |
| `memory=static` | `StaticLeverMemoryProvider` | `NullPostRunHook` | fixed curated lever blocks; no bank, no embedder, no memory LLM |

> ⚠️ **`writer` and `full` bill the memory LLM** (`memory/llm=gemini` by
> default). A no-memory baseline is `pipeline=guided memory=none` — both
> sides off in one preset. Do not use `memory=writer` as a "no-memory" run: it
> pays full write-side cost for cards nobody reads (it is the deliberate
> "seed a bank" move).

### Pipeline compatibility

External-memory read is a pipeline property:

- `pipeline=guided` never reads external memory cards.
- `pipeline=memory_guided` reads external memory cards and requires
  `memory={reader,full,static}`.

External-memory write is a memory property:

- `memory/write=none` disables live and final writer hooks.
- `memory/write=end_of_run` writes once when the run stops.
- `memory/write=live` also installs `LiveMemoryRefreshHook` for mid-run writer
  sweeps and requires `memory={writer,full}`.

With the default per-run `checkpoint_dir`, `pipeline=memory_guided memory=full`
must use `memory/write=live`: otherwise the reader looks at an empty run-local
bank until the finalizer writes after the run is already over. End-of-run write
mode is for bank-building runs or for runs that read an explicit pre-built
`checkpoint_dir`.

## How memory flows through a run

**Read (per mutation).** `MemoryContextStage` calls
`provider.select_cards(program, ...)`. Under `memory={reader,full}` that runs
the `MemoryReader` stack:

```
research   LangGraph agent over the store: plan → retrieve → reflect
           (the planner sees a digest of the newest `digest_max_cards`
           banked cards — default 50, one description line each; the
           shortlister pre-benches warm cards the bootstrap reputation
           prices as guaranteed losers, so they stop occupying slots)
reputation bootstrap-EV re-pricing (mean + low quantile of a staleness-
           weighted bootstrap over each card's raw oriented gain deltas)
           over the cell-local BD-proximity posterior
auction    Thompson gate vs a no-card baseline arm; known cards bid the
           mean of one weighted bootstrap resample of direct deltas, zero
           atoms for invalid/unused exposures, plus a neutral pseudo-event;
           genuinely cold cards use the cold scale; gated by bid > 0, a
           round-quantile reserve, and a Sidak-adjusted no-card baseline gate
budget     cap winners to reader.max_cards
probe      cold-exploration lane after budget: a card whose staleness-scaled
           effective support is strictly below the evidence floor is probe-
           eligible and may fill an empty selection or (rarely) displace the
           weakest budgeted card when the budget is full
render     mutator-facing prompt block incl. efficacy endorsement
```

Every stage fails to an empty selection — a memory outage never sinks a
mutation. Winning blocks land in `MutationSuggestionStage.memory_cards`; the
final mutator sees the resulting structured suggestions, not raw external-card
text. The child program is stamped with frozen card-attribution metadata (see
[Tracking](#tracking-did-memory-actually-flow)).

**Write (per increment / end of run).** `MemoryWriter` runs:

```
extract    eligible parent→child records (strict metric validity)
reconcile  librarian LLM turns each diff into NEW / DUPLICATE / MERGE cards;
           a byte-identical description never mints a second id (exact
           text-twin dedup resolves to a provenance bump on the banked twin)
admit      novelty gate (optional, off by default) — reject NEW idea cards
           inside the mutator's prior
exemplars  author program cards for top-fitness programs (best_programs_percent);
           a harm-evicted exemplar is tombstoned and never re-authored this run
restamp    base-relative fitness gain events across the full lineage pool;
           credit that no longer resolves to a banked card is dropped
evict      configured eviction sweep — catastrophic-birth, confidently-harmful,
           and policy-non-viable cards leave the bank; deleted ids are
           tombstoned against re-admission for the rest of the run
consolidate inline same-batch folding of each increment's freshly added cards,
           plus a throttled background pass over the whole bank
```

When `writer.novelty_admission_gate` is on (**off by default** — the A/B
isolating its effect is still running; enable with
`memory.writer.novelty_admission_gate=true`), a
`NoveltyAdmissionAgent` scores each freshly-authored idea card on one axis —
*would a strong optimizer LLM already reach for this lever unprompted on this
task?* — and drops the card if so, before it enters the bank. It is a subtraction
gate for the prior-known majority (generic metaheuristic boilerplate the mutator
emits cold), not a quality or correctness check; a sound-but-obvious card is
rejected, a non-obvious card is kept even if wrong. It fails open (judge error →
admit) and never touches the reconcile-failed verbatim path. Insight cards only —
program exemplars carry concrete code+fitness and already dedup by exact code
identity, so prose-novelty is the wrong axis for them.

With `memory/write=live`, `LiveMemoryRefreshHook` additionally triggers bounded
writer sweeps every `refresh_every` post-step invocations, so cards written
mid-run become readable mid-run. Without it, writer-enabled runs write once at
run completion.

## Configuration

`config/memory/full.yaml` is the canonical arm — a flat `${ref:}` graph where
each component is declared once and shared by reference (the same
instantiate-once mechanism algorithm configs use for `behavior_space`):

```yaml
defaults:
  - llm: gemini              # memory LLM router (research + librarian agents)
  - read_policy: adaptive    # owns reputation + auction + budget + excluder + shortlister + probe
  - evictor: recommended     # birth-failure + harm + policy-non-viable eviction
  - write: end_of_run        # shipped cadence: author once at shutdown (override to live)

store:      # LocalMemoryStore = card bank + vector index + research agent
  _target_: gigaevo.memory.storage.local.LocalMemoryStore
  config:
    _target_: gigaevo.memory.storage.config.StoreConfig
    path: <checkpoint_dir>
    research: { _target_: gigaevo.memory.storage.config.ResearchConfig, default_top_k: 10, max_cards: 10 }
  llm: ${ref:memory.llm}

reader:
  _target_: gigaevo.memory.read.reader.MemoryReader
  # read_policy supplies `shortlister`; the adaptive policy uses
  # BootstrapFusedRankingShortlister with digest_max_cards=50 and
  # rep_floor_quantile=0.4.
  reputation: ${ref:memory.reputation}
  auctioneer: ${ref:memory.auction}
  budgeter: ${ref:memory.budget}
  context_model: ${ref:memory.context_model}
  candidate_projector:
    _target_: gigaevo.memory.read.projection.AuctionCandidateProjector
    prior: ${ref:memory.prior}
    context_model: ${ref:memory.context_model}
    no_card_evidence: ${ref:memory.no_card_evidence}
  probe_policy: ${ref:memory.probe_policy}
  renderer: { _target_: gigaevo.memory.read.render.EfficacyCardRenderer }
  max_cards: 1               # injection budget — cards the mutator sees

provider:   # ← pipelines consume this
  _target_: gigaevo.memory.provider.ReaderMemoryProvider
  reader: ${ref:memory.reader}
  excluder: ${ref:memory.excluder}

writer:     # ← engines consume this as post_run_hook
  _target_: gigaevo.memory.write.writer.MemoryWriter
  llm: ${ref:memory.llm}
  evictor: ${ref:memory.evictor}
  ...
```

Choose a whole read stack first (`memory/read_policy=portable`) and tune a leaf
only when needed (`memory.auction.ev_floor_quantile=0.5` for bootstrap policies,
`memory.reader.max_cards=2`). Raw `memory/reputation`, `memory/auction`, and
`memory/budget` leaves are still available for ablations, but the public API is
`memory/read_policy`.

### Component groups

| Group | Options | Notes |
|---|---|---|
| `memory/llm` | `gemini` (default), `qwen_instruct` | one router shared by the research + librarian agents |
| `memory/read_policy` | `adaptive` (default), `portable`, `median_ev_legacy`, `probability_legacy`, `contextual_bootstrap_decay`, `portable_bootstrap_decay`, `decay_median_ev_legacy` | whole read-stack presets. `adaptive` = contextual bootstrap-EV over BD-proximity + EB cold priors + persisted no-card evidence + explicit cold probes + bootstrap auction + top-bid budget + warm-card bench + lineage excluder; this is the recommended default. `portable` = same adaptive stack in global context, no `behavior_space` dependency; use for multi-island/no-BD algorithms. Decay variants are explicit experiments. Legacy variants preserve old median-EV / probability-only baselines with fixed priors and no cold probes |
| `memory/context` | `bd_cell` (default under `adaptive`), `global` | shared context policy used by reader, shortlist bench, no-card evidence, and writer baseline fitting. `bd_cell` recomputes cells under the live behavior space with global fallback; `global` ignores behavior space |
| `memory/prior` | `empirical_bayes` (adaptive default), `fixed_3_3` | cold-card prior policy. EB learns a config-driven cohort ladder (`levels`, default global → kind → kind+category, then context / context+kind / context+kind+category once the context model resolves a non-global bucket) from banked first non-founding exposures, shrunk toward a neutral `seed_prior: [1,1]` under a `k_max=6` concentration cap; a level whose counts match the previous one is skipped so the same evidence never compounds shrinkage. Fixed is for legacy/reproduction |
| `memory/no_card_evidence` | `json` (adaptive default), `none` | writer-published no-card controls/natural empty outcomes consumed by the reader's no-card abstention gate. This is distinct from the randomized no-card control rate |
| `memory/probe` | `cold_budget` (adaptive default), `none` | explicit cold-card exploration lane, run after the auction and budget. A card is probe-eligible iff its bid reports a non-empty support kind and its staleness-scaled effective support is strictly below the evidence floor (`probe_until_effective_events`, wired to `${memory.evidence.min_effective_events}`) — the same measure eviction uses, so probe (strict `<`) and eviction/adjudication (`>=`) partition card-space with no gap. With an empty selection it fills one slot with the best cold candidate at rate 0.50; with warm winners it adds a probe at rate 0.03 and only displaces the weakest budgeted card when the budget is already full (otherwise the probe joins the proven winner). At most one probe card per decision; empty support kinds are never probe-eligible (fail-safe) |
| `memory/reputation` | `bootstrap_bd`, `bootstrap_global`, `bootstrap_bd_decay`, `bootstrap_global_decay`, `bd_proximity`, `beta_binomial`, `bd_proximity_decay` | expert leaves used by read policies. Prefer selecting `memory/read_policy` unless running an ablation. `bootstrap_bd` wraps `bd_proximity` and re-prices each card's gain summary on the mean + low quantile of a weighted bootstrap over raw oriented deltas; staleness enters as per-event weight `w = 2^(-s/H)`. `bd_proximity` needs a single shared `behavior_space`; use `bootstrap_global`/`portable` otherwise |
| `memory/auction` | `thompson_bootstrap` (default), `thompson_ev`, `thompson` | `thompson_bootstrap` bids the mean of one staleness-weighted bootstrap resample of the card's EV support + a neutral pseudo-event; genuinely cold cards bid posterior × cold scale. It is gated by `bid > 0` plus an inclusive `ev_floor_quantile` reserve over the round's own bids (self-normalizing, no Beta assumption) and a Sidak-adjusted no-card baseline gate (`gate_quantile = baseline_quantile^(1/eligible_count)` against the persisted no-card evidence arm); `thompson_ev` bids expected value (θ × gain magnitude); `thompson` bids probability only. `thompson_bootstrap_novelty` (add with `+memory/auction=thompson_bootstrap_novelty` — the `+` is required because read_policy owns the group) is `thompson_bootstrap` with a novelty tax: each bid is scaled by `(1 + use_count)^-novelty_power` (use_count = the card's non-founding gain events, a deterministic injection count) before the reserve is computed, so repeat winners make room for fresh cards while injection volume is preserved (the quantile floor re-normalizes over the taxed bids) |
| `memory/budget` | `top_bid` (default), `top_theta` | pair `top_bid` with the EV bidders (`thompson_bootstrap`, `thompson_ev`) and `top_theta` with `thompson` |
| `memory/excluder` | `lineage` (default), `none` | `lineage` excludes cards already applied on the parent's lineage before research |
| `memory/evictor` | `recommended` (default), `harm`, `none` | `recommended` composes catastrophic birth-failure deletion, later-use harm eviction, and policy-non-viable active-bank cleanup after the reputation's effective evidence floor; contextual reputations provide explicit supported contexts for that cleanup; `harm` keeps only the later-use harm sweep |

Top-level `memory.baseline_prior` is the shared fallback no-card arm prior
(`Beta(3,3)` by default). Auctions, persisted no-card evidence, legacy fixed
cold priors, and reputation fallback cold priors all reference it. Adaptive EB
cold-card priors intentionally keep their own weaker `seed_prior: [1,1]` so
bank evidence can move cold-card beliefs quickly.

`memory/llm` is independent of the main mutation `llm`. The default
`memory/llm=gemini` calls OpenRouter and reads `OPENROUTER_API_KEY`.
`memory/llm=qwen_instruct` reads `OPENAI_API_KEY` and targets the configured
LiteLLM-compatible proxy; pair it with a larger thinking model on the main
`llm=single` route when mutation and card work should use different models.

### The read funnel — three distinct widths

A card travels through three narrowing stages, each with its own knob:

1. `memory.store.config.research.default_top_k` — **retrieval fan-out**: how
   many nearest cards *each* scoped vector query pulls from Chroma. The research
   agent may issue several queries across several iterations; their hits
   aggregate (deduped by card id) into one candidate pool.
2. `memory.store.config.research.max_cards` — **recall width (the shortlist)**:
   how many of that candidate pool the reflector may select. This shortlist is
   the population the auction ranks.
3. `memory.reader.max_cards` — **injection budget**: how many auction winners
   the budgeter actually renders into the mutation prompt.

So `default_top_k` and `research.max_cards` feed the auction; `reader.max_cards`
gates the auction's output. The shipped `memory=full`/`memory=reader` arms set
`default_top_k: 10` and `research.max_cards: 10` (a 10-wide shortlist) with
`reader.max_cards: 1` (one injected winner). The bare `ResearchConfig` Pydantic
defaults are 3/3 — the arms override them.

### Store and embedding knobs (`memory.store.config`)

Embedding is config, not code. `embed.embed_scopes` maps a scope name to the
card text fields concatenated into that scope's vector collection (defaults:
`description`, `desc_expl`, `desc_task`); `embed.nearest_scope` (default
`desc_expl`) backs the write path's nearest-card lookups (reconcile-agent
context, consolidation candidates — all pure top-k; there is no distance
threshold anywhere in the write path — program exemplars dedup by exact
normalized-code identity, not by embedding distance);
`embed.embedding_model` defaults to `Snowflake/snowflake-arctic-embed-m-v1.5`.
`embed.query_prefix` is the instruction prepended to every retrieval *query*
before it is embedded (never to the indexed card documents) — the asymmetric
query prompt arctic-embed-m-v1.5 was trained with, defaulting to
`"Represent this sentence for searching relevant passages: "`; set it to `""`
for a symmetric embedder that takes no query instruction.
`research.{max_iters,default_top_k,max_cards,query_scopes}` bound the research
loop.

The index fingerprints its embedding config (`embedding_model` + each scope's
field set) beside the Chroma data. Reopening a checkpoint dir under a *changed*
embedder fails loudly — the old vectors are incompatible with the new one, and
Chroma keys collections by name, so they would otherwise linger and silently
corrupt retrieval. **Changing the embedder means a fresh `checkpoint_dir`.**
(`query_prefix`/`nearest_scope` tune only queries, not stored vectors, so
they are not fingerprinted and can change freely against an existing bank.)

### `memory=static` — curated lever baseline

Serves a fixed `---`-separated block file into the same prompt slot the
dynamic system feeds, identically for every child — no bank, no embedder, no
memory LLM:

```bash
python run.py ... pipeline=memory_guided memory=static \
    memory.provider.levers_file=/abs/path/levers.md
```

A missing, empty, or wrong-block-count levers file fails the launch
(`memory.provider.expected_blocks`, default 6) rather than silently running a
degraded arm.

## Cards and gain events

One `Card` model (`gigaevo/memory/cards.py`), `kind ∈ {insight, program}`:

- **insight** — a distilled, transferable optimization lever: `description`
  (the mechanism), `explanation_summary` (one-line *why*, indexed as its own
  retrieval scope), `task_description_summary`, `keywords`, `category`.
- **program** — a top-fitness exemplar: the same prose fields plus
  `program_id`, `code`, and `fitness` (kind-gated by a validator).

When a card is injected and the child evaluates, the writer stamps a gain
event on the card: the base-relative fitness delta plus the decision context
(parent metrics). Reputation re-prices those events by weighted bootstrap over
the raw oriented deltas (mean + low quantile; staleness fades old evidence via
the bank-cycle resample weight toward neutral zero), and the auction bids the
mean of one such resample against a no-card Thompson baseline arm. Genuinely
cold cards use the round's borrowed gain scale for their first probe, so a
known card holds its slot only while its own gain distribution keeps beating
"inject nothing" (a fat left tail bids negative and abstains on the sign gate).
The downside Beta-Binomial
posterior is still what the harm gate reads; confidently-harmful posteriors
get the card evicted. Harm counting
is a **strict sign test** — any event with gain below zero counts as harm.
This is a deliberate departure from the old MAD noise band: per our analysis
a per-card band could not be designed soundly for these gain distributions,
so tiny negative deltas do count against a card and the noise guard is the
counting posterior itself — `harm_min_events: 3` before a card can be judged
harmful at all by default, via `memory.evidence.min_effective_events`, plus the
optimistic `harm_quantile` read of P(not harmful).
Measured on run data, that guard holds the sequential false-harm rate to
~0.77% at the observed median of ~2 uses per card; revisit the calibration if
cards start accumulating more than ~5 uses.

A freshly-authored **insight** card is born with a *founding* gain event: the
true signed fitness delta of the parent→child mutation it was distilled from
(negated for minimize objectives, so positive always means improvement). The
founding event is origin/admission evidence only: it does not enter the
use-attributed downside posterior, confidence flag, renderer endorsement, or EV
bid. Catastrophic founding losses are deleted by the recommended write-side
birth-failure evictor before they can behave like ordinary cold cards. Mild
founding-only cards remain statistically cold until a later child actually uses
them.

The founding event is preserved across the periodic restamp that recomputes
every use-attribution event from the program pool (it can never be re-derived:
the founding child predates the card), and it rides card merges onto the
survivor. It rides **NEW admits only**: a DUPLICATE or MERGE ruling at ingest
drops the incoming founding event, because the delta was measured for that
child against its parent — foreign evidence for a pre-existing lever.
Harm-eviction remains later-use-only; catastrophic origin failures are handled
by the separate birth-failure policy. The recommended evictor also removes
policy-non-viable cards after enough effective support when their active value
estimate is at or below `memory.neutral_gain` and their direct
baseline-adjusted evidence never beat that neutral point. The default evidence
floor is the readable shared knob `memory.evidence.min_effective_events: 3`,
mirrored into `memory.eviction_safety.min_effective_events`. That same floor
partitions the card lifecycle: the read-side cold-probe lane keeps a card
probe-eligible while its staleness-scaled effective support is strictly below
the floor, and only at or above the floor is the card adjudicable by auction
merit and evictable by `PolicyNonViableEvictor`. Both lanes compute effective
support with identical arithmetic (sum of `max(0, w)` over finite event weights,
times the staleness weight), probe using strict `<` and eviction using `>=`, so
support exactly at the floor lands in the eviction/adjudication lane — no card
falls between the lanes into a stuck zombie state where diluted unused/invalid
exposure could neither probe nor be evicted. Contextual scorers
supply explicit supported evidence contexts for harm and policy cleanup; if a
custom contextual scorer cannot provide them, the default skips contextless
cleanup. Program exemplars keep the zero-evidence-at-birth path.

## Tracking: did memory actually flow?

Parent-stage transient metadata written by `MemoryContextStage`:

| Key | Meaning |
|---|---|
| `memory_candidate_slate` | the full auction slate (winners and losers) |
| `memory_selected_idea_ids` | cards selected/rendered for the current parent-stage invocation; overwritten on requeue |

Child-birth metadata frozen by the mutation path
(`gigaevo/evolution/mutation/constants.py`):

| Key | Meaning |
|---|---|
| `memory_injected_idea_ids` | sorted union of cards actually rendered into the child prompt |
| `memory_used` | bool: whether any external card was injected |
| `memory_base_selected_idea_ids` / `memory_base_metrics` / `memory_base_id` | base-parent snapshot used for gain attribution |
| `memory_no_card_control` | selected cards were intentionally withheld for a randomized no-card control |
| `memory_lineage_applied_ids` | lineage-accumulated applied card ids (feeds the `lineage` excluder) |

The mutator's self-report of which shown cards it actually applied lives in
`mutation_output.card_ids_used`, not in `memory_used`.

## Observability

Per run, under `checkpoint_dir`:

| File | Contents |
|---|---|
| `memory_events.jsonl` | canonical event stream: read decisions, research steps, auction slates, budget caps, store writes/syncs, gain restamps, eviction sweeps, consolidation passes |
| `write_ledger.jsonl` | append-only admission/eviction verdicts (outcomes: `added`, `updated`, `merged`, `rejected_harm`, `rejected_novelty`, `evicted`; a benign no-op ingest — `DISCARDED` — writes no row) |
| `cards.json` | the bank itself |

First stop when debugging empty selections, repeated winners, or evictions:

```bash
python tools/memory_event_report.py <run-dir>    # events + ledger + bank summary
python tools/analyze_bandit_health.py <run-dir>  # auction/posterior health + figures
```

All memory logs carry a `[Memory][<Component>]` prefix.

## Two-pass workflow: build a bank, then A/B it

```bash
# Phase A — seed a bank (writer only, nothing injected):
python run.py problem.name=my_task pipeline=guided memory=writer \
    checkpoint_dir=/data/banks/my_task ...

# Phase B — treatment reads the Phase A bank; control runs memory=none:
python run.py problem.name=my_task pipeline=memory_guided memory=reader \
    checkpoint_dir=/data/banks/my_task ...
python run.py problem.name=my_task pipeline=guided memory=none ...
```

A fresh store over an existing checkpoint dir cold-loads the bank from disk at
construction — this cross-run handoff is covered end-to-end by
`tests/memory/test_e2e.py`.
