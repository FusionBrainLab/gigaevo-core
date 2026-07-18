# Memory System

> **⚠️ v1 — superseded.** This guide documents the retired v1 read stack
> (`MemoryReader`, the reputation/auction/probe pipeline, and the
> `memory={reader,writer,full,static}` arms), all removed in the memory-v1
> removal. The live system is the contextual Bayesian **memory v2**
> (`memory={none,v2}`); its canonical reference is
> [`memory_v2_bayesian_system_report.md`](memory_v2_bayesian_system_report.md).
> Sections below are kept for historical context and are **not** an accurate
> description of the current code. Full v1→v2 rewrite of this guide is pending.

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

The default singleton-parent causal Bayesian iteration is documented separately
in [`memory_v2.md`](memory_v2.md). It preserves the v1 card/write machinery but
does not reuse the v1 reputation or bootstrap-auction evidence model.

## The 30-second version

```bash
# Default: memory v2, live writes, one parent, 70% delivery / 30% control:
python run.py problem.name=heilbron

# Balanced validation of the same system:
python run.py problem.name=heilbron \
  memory.posterior_config.reference_offer_probability=0.50 \
  memory.policy_config.offer_probability=0.50

# Read from a pre-built bank:
python run.py problem.name=heilbron pipeline=memory_guided memory=reader \
  checkpoint_dir=/data/banks/heilbron

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=guided memory=none
```

`memory=v2` is the base-experiment default. The v1 presets
`memory={none,reader,writer,full,static}` remain explicit choices.
`memory/write={none,end_of_run,live}` chooses write cadence. Pipelines read
`${ref:memory.provider}` only in memory-guided pipelines; engines read
`${ref:memory.write.post_run_hook}` as their finalizer.

| Arm | provider (read side) | writer (write side) | What runs |
|---|---|---|---|
| `memory=v2` | `CausalBanditMemoryProvider` | `MemoryWriter` with causal updater | agentic retrieval then contextual Bayesian selection plus live writing; one card, one parent, 70/30 delivery/control |
| `memory=none` | `NullMemoryProvider` | `NullPostRunHook` | nothing |
| `memory=reader` | `ReaderMemoryProvider` | `NullPostRunHook` | injects from a pre-built bank; no extraction |
| `memory=writer` | `NullMemoryProvider` | `MemoryWriter` | authors a bank for a *later* run; injects nothing |
| `memory=full` | `ReaderMemoryProvider` | `MemoryWriter` | reader + writer share one bank under `checkpoint_dir`; ships with `memory/write=live` (same-run memory effects) — override to `end_of_run` only to seed a bank |
| `memory=static` | `StaticLeverMemoryProvider` | `NullPostRunHook` | fixed curated lever blocks; no bank, no embedder, no memory LLM |

> ⚠️ **`v2`, `writer`, and `full` bill the memory LLM** (`memory/llm=gemini` by
> default). A no-memory baseline is `pipeline=guided memory=none` — both
> sides off in one preset. Do not use `memory=writer` as a "no-memory" run: it
> pays full write-side cost for cards nobody reads (it is the deliberate
> "seed a bank" move).

### Pipeline compatibility

External-memory read is a pipeline property:

- `pipeline=guided` never reads external memory cards.
- `pipeline={memory_guided,memory_guided_noise}` reads external memory cards and
  requires `memory={v2,reader,full,static}`.

External-memory write is a memory property:

- `memory/write=none` disables live and final writer hooks.
- `memory/write=end_of_run` writes once when the run stops.
- `memory/write=live` also installs `LiveMemoryRefreshHook` for mid-run writer
  sweeps and requires `memory={writer,full}`.

`memory=full` ships with `memory/write=live` for exactly this reason: with the
default per-run `checkpoint_dir`, an end-of-run cadence would leave the reader
looking at an empty run-local bank until the finalizer writes after the run is
already over — the compose guard rejects that combination. Override to
`memory/write=end_of_run` only for bank-building runs or runs that read an
explicit pre-built `checkpoint_dir`.

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
auction    Thompson gate vs a no-card baseline arm; each card draws one u;
           known cards bid the empirical u-quantile of one weighted bootstrap
           batch over direct deltas, zero atoms for invalid/unused exposures,
           plus a neutral pseudo-event, while genuinely cold cards bid
           Beta.ppf(u) times the cold scale; the same Beta.ppf(u) serves the
           no-card gate; gated by bid > 0, a
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
author     inspect parent, child, diff, mutator explanation, signed gain,
           validity, and archive status; emit DROP or at most one NEW
           conditional hypothesis
dedup      retrieve same-kind/same-task neighbors using the authored
           description + mechanism, then classify strict NEW / EQUIVALENT;
           equivalence requires the same action and applicability condition
           and pools evidence without rewriting the banked treatment
admit      novelty gate (optional, off by default) — reject NEW idea cards
           inside the mutator's prior
exemplars  author one holistic strategy hypothesis for each selected strong
           program; equivalent strategy families retain their best concrete
           representative
evidence   synchronize completed causal outcomes and selection leases
retire     periodically remove only supported card lineages whose optimistic
           posterior probability of safe, helpful utility is low in every
           observed context; pending or stale verdicts fail closed
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
  - write: live              # shipped cadence: same-run read+write via LiveMemoryRefreshHook (override to end_of_run to seed a bank for a later reader)

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
only when needed (`memory.auction.ev_risk_alpha=0.1` to tighten the default risk
reserve, `memory.reader.max_cards=2`). Raw `memory/reputation`, `memory/auction`, and
`memory/budget` leaves are still available for ablations, but the public API is
`memory/read_policy`.

### Component groups

| Group | Options | Notes |
|---|---|---|
| `memory/llm` | `gemini` (default), `qwen_instruct` | one router shared by the research + librarian agents |
| `memory/read_policy` | `adaptive` (default), `portable`, `median_ev_legacy`, `probability_legacy`, `contextual_bootstrap_decay`, `portable_bootstrap_decay`, `decay_median_ev_legacy` | whole read-stack presets. `adaptive` = contextual bootstrap-EV over BD-proximity + EB cold priors + persisted no-card evidence + explicit cold probes + bootstrap auction + top-bid budget + warm-card bench + lineage excluder; its relative warm-card floor exempts cards below the shared effective-evidence boundary so they retain auction/probe access. This is the recommended default. `portable` = same adaptive stack in global context, no `behavior_space` dependency; use for multi-island/no-BD algorithms. Decay variants are explicit experiments. Legacy variants preserve old median-EV / probability-only baselines with fixed priors and no cold probes |
| `memory/context` | `bd_cell` (default under `adaptive`), `global` | shared context policy used by reader, shortlist bench, no-card evidence, and writer baseline fitting. `bd_cell` recomputes cells under the live behavior space with global fallback; `global` ignores behavior space |
| `memory/prior` | `empirical_bayes` (adaptive default), `fixed_3_3` | cold-card prior policy and warm Beta-Binomial base. EB learns a config-driven global/kind/category/task/context cohort ladder from each card's temporally first non-founding exposure, counting only strictly positive causal gain as help; zero, negative, and invalid outcomes enter the failure/complement mass. Context levels are native-task-only because BD metrics cannot be compared across tasks; global levels pool only these hard help/no-help signs. Counts shrink toward `seed_prior: [1,1]` under a `k_max=6` cap, and duplicate counts never compound shrinkage. Fixed is for legacy/reproduction; its warm posterior now coherently starts at Beta(3,3), not Beta(1,1) |
| `memory/no_card_evidence` | `json` (adaptive default), `none` | writer-published no-card controls/natural empty outcomes consumed by the reader's no-card abstention gate. This is distinct from the randomized no-card control rate |
| `memory/probe` | `cold_budget` (adaptive default), `none` | explicit cold-card exploration lane, run after the auction and budget. A card is probe-eligible iff its bid reports a non-empty support kind and its staleness-scaled effective support is strictly below the evidence floor (`probe_until_effective_events`, wired to `${memory.evidence.min_effective_events}`) — the same measure eviction uses, so probe (strict `<`) and eviction/adjudication (`>=`) partition card-space with no gap. With an empty selection it fills one slot with the best cold candidate at rate 0.50; with warm winners it adds a probe at rate 0.03 and only displaces the weakest budgeted card when the budget is already full (otherwise the probe joins the proven winner). At most one probe card per decision; empty support kinds are never probe-eligible (fail-safe) |
| `memory/reputation` | `bootstrap_bd`, `bootstrap_global`, `bootstrap_bd_decay`, `bootstrap_global_decay`, `bd_proximity`, `beta_binomial`, `bd_proximity_decay` | expert leaves used by read policies. Prefer selecting `memory/read_policy` unless running an ablation. `bootstrap_bd` wraps `bd_proximity` and re-prices each card's gain summary on the mean + low quantile of a weighted bootstrap over raw oriented deltas; event `i` gets staleness weight `w_i = 2^(-s_i/H)` from its own stamp. `bd_proximity` needs a single shared `behavior_space`; use `bootstrap_global`/`portable` otherwise |
| `memory/auction` | `thompson_bootstrap` (default), `thompson_ev`, `thompson` | `thompson_bootstrap` draws one `u` per card: a known card bids the empirical `u`-quantile of one staleness-weighted bootstrap-EV batch over its support + a neutral pseudo-event, while a genuinely cold card bids `Beta.ppf(u) × cold scale`; that same `Beta.ppf(u)` is its no-card gate theta. It is gated by a per-card EV reserve — the default `ev_reserve_mode=risk` admits a card iff `P(EV>0) >= 1 - ev_risk_alpha` (0.8 at the default `ev_risk_alpha=0.2`), read off a card-local bootstrap vector so admission is IIA-clean (independent of the rest of the slate); the legacy `ev_reserve_mode=quantile` instead requires `bid > 0` plus an inclusive `ev_floor_quantile` reserve over the round's own bids (self-normalizing, no Beta assumption) — and a Sidak-adjusted no-card baseline gate (`gate_quantile = baseline_quantile^(1/eligible_count)` against the persisted no-card evidence arm); `thompson_ev` likewise shares one `u` between its `θ × gain magnitude` bid and theta gate; `thompson` bids probability only. `thompson_bootstrap_novelty` (add with `+memory/auction=thompson_bootstrap_novelty` — the `+` is required because read_policy owns the group) is `thompson_bootstrap` with a novelty tax: each bid is scaled by `(1 + use_count)^-novelty_power` (use_count = the card's non-founding gain events, a deterministic injection count) before the reserve is computed, so repeat winners make room for fresh cards while injection volume is preserved (the quantile floor re-normalizes over the taxed bids) |
| `memory/budget` | `top_bid` (default), `top_theta` | pair `top_bid` with the EV bidders (`thompson_bootstrap`, `thompson_ev`) and `top_theta` with `thompson` |
| `memory/excluder` | `lineage` (default), `none` | `lineage` excludes cards already applied on the parent's lineage before research |
| `memory/evictor` | `recommended` (default), `harm`, `none` | `recommended` composes catastrophic birth-failure deletion, later-use harm eviction, and policy-non-viable active-bank cleanup after the reputation's effective evidence floor; contextual reputations provide explicit supported contexts for that cleanup; `harm` keeps only the later-use harm sweep |

`PendingDiscountedBootstrapAuctioneer` is a code-level opt-in only; no shipped
preset selects it. It scales each bid by
`(1 + pending_count)^-pending_power`, where `pending_count` is the lease
registry's pre-selection snapshot of uncredited in-flight exposures. The
snapshot is taken before the current winners are attached, so a card never
taxes its own current selection. `pending_counts=None` projects zero exposure,
and the auctioneer's default `pending_power=0` is bid-for-bid identical to the
base bootstrap auction without consuming another RNG draw.

EB cold priors can optionally consume an evicted-card evidence source to correct
the upward survivorship bias caused when harm-evicted cards drop out of cohort
help rates; the default is snapshot-only, no shipped preset wires the source,
and inclusion-propensity/IPS weighting remains a follow-up.

The bootstrap auction's per-card reserve is `memory.auction.ev_reserve_mode`.
The shared `thompson_bootstrap` preset sets it to `risk` with `ev_risk_alpha=0.2`,
so it is the default for every consumer of that auction — `memory=full`
(`adaptive`) and `memory=reader` (`portable`) alike, plus the `*_bootstrap_decay`
variants: a card is admitted when the fraction of its own bootstrap-EV samples
above zero is at least `1 - alpha` (P(EV>0) >= 0.8). That probability is read off a **card-local**
bootstrap vector seeded only from the card id (mirroring the per-card stats
bootstrap), decoupled from the shared-RNG bid draw. Two consequences: admission
is independent of which other cards share the round or where the card sits in the
draw order (IIA-clean), and turning the reserve on leaves the shared round RNG —
every bid plus the baseline draw — byte-identical. The legacy
`ev_reserve_mode=quantile` (retained by the `BootstrapThompsonAuctioneer` class
default and the `thompson_bootstrap_novelty` preset — the `*_legacy` read
policies use a different auctioneer with no EV reserve at all) instead prices an
inclusive `ev_floor_quantile` reserve over the round's own bids, which is
round-relative. `AuctionBid` records `ev_reserve_mode`,
`ev_positive_probability`, `ev_risk_alpha`, and `rejected_by_ev_floor`; those
fields also appear inside `MEMORY_AUCTION_RUN.bids`.

Top-level `memory.baseline_prior` is the shared fallback no-card arm prior
(`Beta(3,3)` by default). Auctions, persisted no-card evidence, legacy fixed
cold priors, and reputation fallback cold priors all reference it. Adaptive EB
cold-card priors intentionally keep their own weaker `seed_prior: [1,1]` so
bank evidence can move cold-card beliefs quickly.

Every shipped reputation preset also passes the selected card prior `(a0,b0)`
into its warm downside posterior: `a = a0 + (n-k_harm)` and
`b = b0 + k_harm`. The first valid event therefore updates the same Bayesian
world the zero-event projector exposed instead of resetting to Beta(1,1).
Explicit posterior-decay presets shrink evidence back toward that same `(a0,b0)`;
only direct/unwired `prior=None` construction retains the historical Beta(1,1)
warm base and decay target.

The reputation field `harm_model` defaults to `soft_count`, preserving those
updates exactly. The opt-in `mixture` model treats each finite event's latent
harm sign as an independent Bernoulli draw with probability `p_i=harm_mass_i`.
For legacy posterior parameters `alpha=a0+n-k_harm`, `beta=b0+k_harm`,
`N=alpha+beta`, and
`sigma_K^2=sum(weight_i^2 * p_i * (1-p_i))`, it returns a Beta with the same
mean and matched total pseudo-count
`S=alpha*beta*(N+1)/(alpha*beta+N*sigma_K^2)-1`:
`a_mix=(alpha/N)*S`, `b_mix=(beta/N)*S`. Thus uncertain signs widen the Beta
without moving its mean. Invalid/unused failures are deterministic harm and add
no sign variance; exact events also have `sigma_K^2=0`, so both cases
short-circuit to byte-identical legacy parameters. Foreign cross-task evidence
remains a hard-sign fold after the native mixture. No shipped preset enables
this field. Set it on each posterior-owning reputation, never on a bootstrap or
decay decorator. A direct `beta_binomial`/`bd_proximity` preset accepts
`+memory.reputation.harm_model=mixture`. The default adaptive `bootstrap_bd`
stack uses
`+memory.reputation.inner.harm_model=mixture` and, if cold-cell fallback should
use it too, `+memory.reputation.inner.fallback.harm_model=mixture`; the fallback
intentionally owns its setting independently.

Staleness is per event, not per card. For decision task `T`, let `S_T` be the
native bank event stamps and let `N_T` be the task population used by the
existing bank-cycle denominator. Then
`H = N_T * half_life_cycles`,
`s_i = |{t in S_T : t > stamp_i}|`, and `w_i = 2^(-s_i/H)`; unstamped events
have `w_i=1`. Foreign bank traffic is excluded from `S_T` and `N_T`, so it
cannot age native evidence. Bootstrap EV and the shared probe/eviction support
use `credit_i * w_i`; a new event therefore does not restore the weight of old
history. Default non-decay reputations keep the Beta posterior credit-only.
Only `*_decay` presets also apply `credit_i * w_i` to each posterior soft count,
invalid/unused failure mass, and foreign sign count before rebuilding the
posterior around `(a0,b0)`.

Bootstrap uncertainty uses that same fused event-weight vector rather than the
raw row count. After appending the unit neutral pseudo-event, let the exact
multinomial sampling weights be `q_j`. Each bootstrap replicate draws
`max(1, round(n_eff))` atoms, where
`n_eff = (sum(q_j) ** 2) / sum(q_j ** 2)` (Kish effective N). Zero-weight
events contribute to neither sum and therefore cannot make a stale history look
more precise. `n_bootstrap` remains the number of independent replicates, while
the block's `effective_events` remains `sum(credit_i * w_i)`; neither is replaced
by Kish N. When every event weight is `1.0`, the neutral weight is also `1.0`,
so `n_eff` equals the historical atom count exactly and seeded sampling consumes
the identical RNG stream.

`memory/llm` is independent of the main mutation `llm`. The default
`memory/llm=gemini` calls OpenRouter and reads `OPENROUTER_API_KEY`.
`memory/llm=qwen_instruct` targets `LOCAL_LLM_PROXY` and uses
`LITELLM_MASTER_KEY` when the proxy has a non-default key; pair it with a larger thinking model on the main
`llm=local_proxy` route when mutation and card work should use different models.

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
`desc_expl`) backs the write path's authored-action neighbor lookup. Embedding
similarity generates candidates only; the LLM equivalence judge requires the
same intervention and applicability condition. There is no distance threshold
or periodic exhaustive consolidation;
`embed.embedding_model` defaults to `Snowflake/snowflake-arctic-embed-m-v1.5`.
`embed.query_prefix` is the instruction prepended to every retrieval *query*
before it is embedded (never to the indexed card documents) — the asymmetric
query prompt arctic-embed-m-v1.5 was trained with, defaulting to
`"Represent this sentence for searching relevant passages: "`; set it to `""`
for a symmetric embedder that takes no query instruction.
`research.{max_iters,default_top_k,max_cards,query_scopes}` bound the research
loop.

The card bank is the source of truth. Each process keeps an in-memory Chroma
index, rebuilds it from `cards.json` at store startup, and rebuilds it again
after observing a cross-process bank change. No vectors or embedding
fingerprints are persisted, so existing `chroma/` directories are ignored dead
data. A process always embeds its locked bank view with its configured model and
scopes; changing embedding settings takes effect on its next startup rebuild.

Shared-bank deletions consult `selection_leases.json`, an atomic flock-guarded
owner registry under `checkpoint_dir`. A selected card remains protected across
processes until its attempt or credited child releases it. Same-host crashed
owners expire by PID liveness plus the `/proc/<pid>/stat` `pid_start` identity,
with `pid_start=0` conservatively falling back to PID-only checks; the two-hour
deadline is only a fallback for owners on other hosts. Unreadable or corrupt
lease state is preserved and fails closed bank-wide: deletion checks skip
eviction, and acquisitions roll back locally and raise instead of returning an
unpublished lease. Recovery is to remove `selection_leases.json` (and its
`.lock` sibling) after runs are quiescent, or accept that live owners republish
their full state on their next lease mutation. There is one accepted narrow
race: another process can delete a card between selection revalidation and the
durable lease write; that render can lose one credit event, which the stats
reconciliation drop path logs.

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

Each card stores its authoring `task_key`; every gain event's decision context
stores the task key under which its measurement ran.

- **insight** — a distilled, transferable optimization lever: `description`
  (conditional action + mechanism), `explanation_summary` (one-line *why*,
  indexed as its own retrieval scope), `task_description_summary`, `category`.
- **program** — a top-fitness exemplar: the same prose fields plus
  `program_id`, `code`, and `fitness` (kind-gated by a validator).

When a card is injected and the child evaluates, the writer stamps a gain
event on the card: the base-relative fitness delta plus the decision context
(parent metrics). Reputation re-prices those events by weighted bootstrap over
the raw oriented deltas (mean + low quantile; each event's own bank-cycle age
fades it toward neutral zero), and the auction draws one
uniform posterior world per card. A warm card bids that world's empirical
quantile from one bootstrap-EV batch; its Beta gate theta is the same uniform
quantile of its help posterior. Genuinely cold cards reuse that theta times the
round's borrowed gain scale for their first probe, so a
known card holds its slot only while its own gain distribution keeps beating
"inject nothing" (a fat left tail bids negative and abstains on the sign gate).
The downside Beta-Binomial
posterior is still what the harm gate reads; confidently-harmful posteriors
get the card evicted, subject to the shared aged-support floor and the
positive-EV veto detailed below. Exact events (`gain_se=0`, including the default point
estimator) keep the **strict sign test**. A positive measured se contributes
its Gaussian below-zero tail mass; a degraded paired measurement stores
`gain_se=None` and contributes the uninformative wide-limit mass
`Phi(0)=0.5`, never a definite sign. With the optional mixture harm model,
those fractional probabilities widen the downside Beta rather than acting as a
deterministic fractional count; the legacy/default soft-count model is
unchanged. For the global no-card median, measured
paired se also includes `std(no_card_deltas, ddof=1) / sqrt(n)` in quadrature
when at least two controls exist; exact point estimates stay exactly zero-se.
This remains a deliberate departure from the old MAD noise band: per our analysis
a per-card band could not be designed soundly for these gain distributions,
so tiny exact negative deltas do count against a card and the noise guard is
the counting posterior itself — `harm_min_events: 3` before a card can be
judged harmful at all by default, via `memory.evidence.min_effective_events`,
plus the optimistic `harm_quantile` read of P(not harmful), set to a conservative
0.95 for the irreversible run-long tombstone (the sweep re-checks after every
event, so a laxer bar invites a multiple-looks false tombstone on marginal
3-loss evidence). Measured on run data at the prior 0.80 bar, that guard held the
sequential false-harm rate to ~0.77% at the observed median of ~2 uses per card,
and 0.95 is strictly tighter; revisit the calibration if cards start accumulating
more than ~5 uses.

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
by the separate birth-failure policy. Harm eviction fires only once a card clears
the same shared aged-support floor the probe lane uses (below), and even then it
spares any card whose optimistic bootstrap EV band still clears the neutral point
(`IntroGain_bootstrap_ev_hi80 > memory.neutral_gain`): the sign-based harm gate is
magnitude-blind, so a fat-tailed winner can read confidently harmful while its
expected value is positive, and the irreversible tombstone must not delete it.
The recommended evictor also removes
policy-non-viable cards after enough effective support when their active value
estimate is at or below `memory.neutral_gain` and their direct
baseline-adjusted evidence never beat that neutral point. The default evidence
floor is the readable shared knob `memory.evidence.min_effective_events: 3`,
mirrored into `memory.eviction_safety.min_effective_events`. That same floor
partitions the card lifecycle: the read-side cold-probe lane keeps a card
probe-eligible while its staleness-scaled effective support is strictly below
the floor, and only at or above the floor is the card adjudicable by auction
merit and evictable by the harm sweep and `PolicyNonViableEvictor`. Both lanes compute effective
support with identical arithmetic (`sum(max(0, credit_i * w_i))` over finite
per-event terms), probe using strict `<` and eviction using `>=`, so
support exactly at the floor lands in the eviction/adjudication lane — no card
falls between the lanes into a stuck zombie state where diluted unused/invalid
exposure could neither probe nor be evicted. Contextual scorers
supply explicit supported evidence contexts for harm and policy cleanup; if a
custom contextual scorer cannot provide them, the default skips contextless
cleanup. Program exemplars keep the zero-evidence-at-birth path.

Across tasks, only each used event's hard help/no-help sign is shared: foreign
gain magnitude, uncertainty, and raw metrics never enter another task's
statistics or deletion decisions. Majority-helpful foreign evidence at the
shared floor vetoes harm eviction, exemplar pruning, and twin retirement.

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
| `memory_events.jsonl` | canonical event stream: read decisions, applicability research, policy actions, store writes/syncs, and writer synchronization including retired card ids |
| `write_ledger.jsonl` | append-only content and retirement verdicts (`added`, `updated`, `rejected_retired`, `rejected_novelty`, `evicted`; `discarded` is an unledgered no-op) |
| `ope_summary.json` | auto-computed DR-AIPW probe-ITT effect (`tau_dr`, CI, IPS cross-check) of the card policy over the ledger, plus reconciliation health (orphans/dupes). Refreshed by the writer after each increment, so it lands in-progress, not only at completion; `status: insufficient_data` until a reconciled treated/control probe outcome exists. Also emitted as a `MEMORY_OPE_SUMMARY` event line |
| `cards.json` | the bank itself |
| `selection_leases.json` | live cross-process owners and their leased card ids; created on first lease |

First stop when debugging empty selections, repeated winners, or evictions:

```bash
python tools/memory_card_health.py <run-dir>     # card bank structural/integrity snapshot
python -m gigaevo.memory.ope.reconcile <run-dir> # off-policy probe-ITT (tau) + A/A + reconciliation, on demand
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
