# Memory System

Cross-run memory for the evolutionary loop. A **write system** distills each
run's mutation diffs and top exemplars into reusable cards; a **read system**
researches that bank and injects the most promising cards into mutation
prompts, tracking each card's realized fitness gain (reputation).

Package-internals map: [`gigaevo/memory/README.md`](../gigaevo/memory/README.md).
The pipeline that consumes memory live: [`INTRA_EXTRA_MEMORY.md`](INTRA_EXTRA_MEMORY.md).

## The 30-second version

```bash
# Memory ON (read + write against one shared bank):
python run.py problem.name=heilbron pipeline=intra_extra_memory memory=full

# True no-memory baseline:
python run.py problem.name=heilbron pipeline=standard memory=none
```

`memory={none,reader,writer,full,static}` is one Hydra knob. Every arm defines
the same two consumer-facing nodes — pipelines read `${ref:memory.provider}`,
engines read `${ref:memory.writer}` as their `post_run_hook` — and the arms
differ only in which `_target_`s those nodes carry (Null variants for disabled
sides). There is no enable flag and no Python assembler; assembly is the YAML
`${ref:}` graph itself.

| Arm | provider (read side) | writer (write side) | What runs |
|---|---|---|---|
| `memory=none` | `NullMemoryProvider` | `NullPostRunHook` | nothing |
| `memory=reader` | `ReaderMemoryProvider` | `NullPostRunHook` | injects from a pre-built bank; no extraction |
| `memory=writer` | `NullMemoryProvider` | `MemoryWriter` | authors a bank for a *later* run; injects nothing |
| `memory=full` | `ReaderMemoryProvider` | `MemoryWriter` | reader + writer share one bank under `checkpoint_dir` |
| `memory=static` | `StaticLeverMemoryProvider` | `NullPostRunHook` | fixed curated lever blocks; no bank, no embedder, no memory LLM |

> ⚠️ **`writer` and `full` bill the memory LLM** (`memory/llm=gemini` by
> default). A no-memory baseline is `pipeline=standard memory=none` — both
> sides off in one preset. Do not use `memory=writer` as a "no-memory" run: it
> pays full write-side cost for cards nobody reads (it is the deliberate
> "seed a bank" move).

### Pipeline compatibility

`pipeline=intra_extra_memory` writes cards live through its `post_step_hook`
(`LiveMemoryRefreshHook`), which raises `TypeError` at startup unless the arm
mounts a real writer. Under that pipeline: use `memory=full` or
`memory=writer`; for `memory=static` add `post_step_hook=null` (there is no
bank to refresh); `memory=reader` and `memory=none` pair with
`pipeline=standard` instead.

## How memory flows through a run

**Read (per mutation).** `MemoryContextStage` calls
`provider.select_cards(program, ...)`. Under `memory={reader,full}` that runs
the `MemoryReader` stack:

```
research   LangGraph agent over the store: plan → retrieve → reflect
reputation Beta-Binomial posterior over each card's recorded gain events
auction    Thompson sampling of card bids against a no-card baseline arm
budget     cap winners to reader.max_cards
render     mutator-facing prompt block incl. efficacy endorsement
```

Every stage fails to an empty selection — a memory outage never sinks a
mutation. Winning blocks land in the mutation prompt and the child program is
stamped with the selected card ids (see [Tracking](#tracking-did-memory-actually-flow)).

**Write (per increment / end of run).** `MemoryWriter` runs:

```
extract    eligible parent→child records (strict metric validity)
reconcile  librarian LLM turns each diff into NEW / DUPLICATE / MERGE cards
admit      novelty gate (optional) — reject NEW idea cards inside the mutator's prior
exemplars  author program cards for top-fitness programs (best_programs_percent)
restamp    base-relative fitness gain events across the full lineage pool
evict      harm sweep — confidently-harmful cards leave the bank
consolidate throttled background near-duplicate folding
```

When `writer.novelty_admission_gate` is on (the default in `memory=full`), a
`NoveltyAdmissionAgent` scores each freshly-authored idea card on one axis —
*would a strong optimizer LLM already reach for this lever unprompted on this
task?* — and drops the card if so, before it enters the bank. It is a subtraction
gate for the prior-known majority (generic metaheuristic boilerplate the mutator
emits cold), not a quality or correctness check; a sound-but-obvious card is
rejected, a non-obvious card is kept even if wrong. It fails open (judge error →
admit) and never touches the reconcile-failed verbatim path. Insight cards only —
program exemplars carry concrete code+fitness and already dedup by exact code
identity, so prose-novelty is the wrong axis for them.

Under `pipeline=intra_extra_memory` the `LiveMemoryRefreshHook` additionally
triggers bounded writer sweeps every `refresh_every` post-step invocations, so
cards written mid-run become readable mid-run. All other pipelines write once,
at run completion.

## Configuration

`config/memory/full.yaml` is the canonical arm — a flat `${ref:}` graph where
each component is declared once and shared by reference (the same
instantiate-once mechanism algorithm configs use for `behavior_space`):

```yaml
defaults:
  - llm: gemini              # memory LLM router (research + librarian agents)
  - reputation: beta_binomial
  - auction: thompson_ev
  - budget: top_bid
  - excluder: none
  - evictor: harm            # scored by ${ref:memory.reputation}

store:      # LocalMemoryStore = card bank + vector index + research agent
  _target_: gigaevo.memory.storage.local.LocalMemoryStore
  config:
    _target_: gigaevo.memory.storage.config.StoreConfig
    path: <checkpoint_dir>
    research: { _target_: gigaevo.memory.storage.config.ResearchConfig, default_top_k: 10, max_cards: 10 }
  llm: ${ref:memory.llm}

reader:
  _target_: gigaevo.memory.read.reader.MemoryReader
  shortlister: { _target_: gigaevo.memory.read.shortlist.ResearchShortlister, store: ${ref:memory.store} }
  reputation: ${ref:memory.reputation}
  auctioneer: ${ref:memory.auction}
  budgeter: ${ref:memory.budget}
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

Swap a component group (`memory/auction=thompson`,
`memory/reputation=bd_proximity`, `memory/llm=qwen_instruct`) or tune a leaf
(`memory.auction.ev_floor=0.01`, `memory.reader.max_cards=2`).

### Component groups

| Group | Options | Notes |
|---|---|---|
| `memory/llm` | `gemini` (default), `qwen_instruct` | one router shared by the research + librarian agents |
| `memory/reputation` | `beta_binomial` (default), `bd_proximity` | `bd_proximity` weights gain events by behavior-descriptor proximity to the current parent |
| `memory/auction` | `thompson_ev` (default), `thompson` | `thompson_ev` bids expected value (θ × gain magnitude); `thompson` bids probability only |
| `memory/budget` | `top_bid` (default), `top_theta` | pair `top_bid` with `thompson_ev` and `top_theta` with `thompson` |
| `memory/excluder` | `none` (default), `lineage` | `lineage` excludes cards already applied on the parent's lineage before research |
| `memory/evictor` | `harm` (default), `none` | harm evictor is scored by the shared reputation instance |

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
python run.py ... pipeline=intra_extra_memory memory=static \
    memory.provider.levers_file=/abs/path/levers.md post_step_hook=null
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
(parent metrics). Reputation reduces those events to a Beta-Binomial
injection posterior; the auction samples that posterior against a no-card
baseline arm, so a card holds its slot only while it keeps beating "inject
nothing". Confidently-harmful posteriors get the card evicted.

A freshly-authored **insight** card is born with a *founding* gain event: the
true signed fitness delta of the parent→child mutation it was distilled from
(negated for minimize objectives, so positive always means improvement). This
seeds the card's bid on its own evidence from the first sweep instead of
starting cold — and a card distilled from a regression bids *low*. The founding
event is preserved across the periodic restamp that recomputes every
use-attribution event from the program pool (it can never be re-derived: the
founding child predates the card), and it rides card merges onto the survivor.
It shapes the **auction bid only** — harm-eviction is usage-based, so a card is
never evicted on the origin delta it was distilled from before use-attribution
has judged it. Program exemplars keep the zero-evidence-at-birth path.

## Tracking: did memory actually flow?

Child-program metadata stamped by the mutation path
(`gigaevo/evolution/mutation/constants.py`):

| Key | Meaning |
|---|---|
| `memory_selected_idea_ids` | cards the reader selected for this mutation |
| `memory_injected_idea_ids` | cards actually rendered into the prompt |
| `memory_used` | mutator self-report of which cards it applied |
| `memory_candidate_slate` | the full auction slate (winners and losers) |
| `memory_base_selected_idea_ids` / `memory_base_metrics` / `memory_base_id` | base-parent snapshot used for gain attribution |
| `memory_lineage_applied_ids` | lineage-accumulated card ids (feeds the `lineage` excluder) |

## Observability

Per run, under `checkpoint_dir`:

| File | Contents |
|---|---|
| `memory_events.jsonl` | canonical event stream: read decisions, research steps, auction slates, budget caps, store writes/syncs, gain restamps, eviction sweeps, consolidation passes |
| `write_ledger.jsonl` | append-only admission/eviction verdicts |
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
python run.py problem.name=my_task pipeline=standard memory=writer \
    checkpoint_dir=/data/banks/my_task ...

# Phase B — treatment reads the Phase A bank; control runs memory=none:
python run.py problem.name=my_task pipeline=standard memory=reader \
    checkpoint_dir=/data/banks/my_task ...
python run.py problem.name=my_task pipeline=standard memory=none ...
```

A fresh store over an existing checkpoint dir cold-loads the bank from disk at
construction — this cross-run handoff is covered end-to-end by
`tests/memory/test_e2e.py`.
