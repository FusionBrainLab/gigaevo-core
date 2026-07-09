# `gigaevo/memory` — architecture

Cross-run memory for the evolutionary loop: a **write system** distills each
run's mutation diffs and top exemplars into reusable cards, and a **read
system** researches that bank and injects the most promising cards into
mutation prompts, tracking each card's realized fitness gain (reputation).

User-facing guide (Hydra knobs, launch recipes, observability):
[`docs/memory.md`](../../docs/memory.md).
Diagram-heavy lifecycle tutorial (card birth/read/write/evidence/eviction):
[`docs/MEMORY_LIFECYCLE_TUTORIAL.md`](../../docs/MEMORY_LIFECYCLE_TUTORIAL.md).

## Layers

```
cards  →  events  →  storage  →  { read │ write }  →  provider / live_memory_hook
```

Strictly ordered — a layer imports only layers to its left; `read/` and
`write/` never import each other (eviction consumes a `CardScorer` Protocol it
declares, `read/reputation` implements it, config wires them). Only
`storage/index.py` touches Chroma; LLM handles live only in
`storage/research.py` and the write-side authoring modules
(`librarian.py`, `consolidation.py`, `writer.py`).
`tests/memory/test_layering.py` enforces all three rules over the AST.

Everything is Pydantic (frozen, `extra="forbid"`, no aliases); malformed
persisted data raises instead of being coerced.

## Module map

| Module | What it owns |
|---|---|
| `cards.py` | The one `Card` model (`kind ∈ {insight, program}`; program fields kind-gated), `ContextualGain` + `EvidenceAttribution` (origin/use/exposure provenance), `DecisionContext`, `CardStatsBlock`, `card_brief()` |
| `events.py` | Typed `MemoryEvent`s (canonical monitoring events + per-run `memory_events.jsonl` sink), `memory_event_context` correlation (decision/program/parent ids) |
| `storage/base.py` | `MemoryStore` ABC: `save/get/delete/snapshot/apply_merges`, `nearest()`, `research()`, `rebuild/close/is_ready` |
| `storage/bank.py` | `CardBank`: thread-safe in-proc dict + atomic `cards.json` persist; cold-loaded from disk once at construction |
| `storage/index.py` | `VectorIndex` (Chroma, one collection per embed scope) |
| `storage/research.py` | LangGraph research agent: plan → retrieve → reflect, ≤`max_iters`, fail-to-empty; prompts under `gigaevo/prompts/retrieval_{planner,reflection}/` |
| `storage/config.py` | `StoreConfig` / `EmbedConfig` / `ResearchConfig` — embedding is config, not code (`embed_scopes` maps scope → card text fields) |
| `storage/local.py` | `LocalMemoryStore` = CardBank ∘ VectorIndex ∘ ResearchAgent |
| `storage/remote.py` | `RemoteMemoryStore` skeleton (httpx; retrieval raises until the remote port lands) |
| `storage/state.py` | `StoreState` transition table (shared `validate_transition` pattern) |
| `storage/hf_cache.py` | HuggingFace cache/timeout shims applied before the embedder downloads |
| `context/models.py` | Shared context models (`GlobalMemoryContext`, `BDCellMemoryContext`) used by reader, shortlist bench, no-card evidence, and write-side baseline fitting |
| `context/no_card.py` | Persisted no-card evidence (`JsonNoCardEvidenceStore`) and static/null provider for legacy policies |
| `read/reader.py` | `MemoryReader` facade: shortlist → reputation → auction → budget → render; every stage a Protocol; fails to empty selection |
| `read/interfaces.py` | Shared Protocols for read components (`ReputationModel`, `Shortlister`, `Auctioneer`, `Budgeter`, renderer, no-card baseline) |
| `read/shortlist.py` | `ResearchShortlister`: mutation-grounded query → `store.research()`; passes a digest of the newest `digest_max_cards` banked cards (default 50, one description line each) as the planner's context |
| `read/fused.py` | `FusedRankingShortlister` (fused re-rank of the inner shortlist: `w_sem`/`w_rep`/`w_nov` + self-normalizing `rep_floor_quantile` gate) and `BootstrapFusedRankingShortlister` (default: warm-only bench over the bootstrap-repriced `ev_lo`/`ev_mean`; benched cards merge into `exclude_ids` before research, cold cards stay explorable) |
| `read/projection.py` | `AuctionCandidateProjector`: attaches context key, EB/static cold prior, no-card evidence, EV support, and staleness to each auction candidate |
| `read/prior.py` | `FixedMemoryPrior` and `EmpiricalBayesMemoryPrior` for cold-card/no-card priors |
| `read/probe.py` | `ColdProbePolicy`: explicit small cold-card exploration lane after auction/budget; `NoColdProbePolicy` for legacy arms |
| `read/reputation.py` | `BetaBinomialReputation` (+ `BDProximityReputation` cell-local variant); posterior/efficacy math over gain events (`block_from_events`); `BootstrapReputation` (default, wraps either): re-prices the gain summary on the mean + low quantile of a staleness-weighted bootstrap over raw oriented deltas, delegating the downside posterior to the inner |
| `read/auction.py` | `ThompsonAuctioneer` / `EVThompsonAuctioneer` / `BootstrapThompsonAuctioneer` (default: known cards bid one weighted bootstrap resample mean + neutral pseudo-event; true cold cards bid posterior × cold scale; sign gate + round-quantile EV reserve) + `TopThetaBudgeter` / `TopBidBudgeter` |
| `read/bootstrap.py` | `bootstrap_ev_samples` (weighted resample of known-card EV support + neutral pseudo-event) + `stable_rng` (sha256-seeded, replayable) — shared by reputation and auction |
| `read/staleness.py` | `bank_cycle_weight`: the ONE staleness mechanism, `w = 2^(-s/H)` (s = bank gain events newer than the card's latest, H = bank size × `half_life_cycles`) |
| `read/decay.py` | `DecayingReputation` (`memory/reputation=bootstrap_*_decay` or `bd_proximity_decay`): explicit posterior/count decay experiment; default bootstrap policies decay EV evidence but not Beta counts |
| `read/exclusion.py` | `NullExcluder`, `LineageExcluder` (filter-first lineage gate) |
| `read/render.py` | `EfficacyCardRenderer` — card → mutator-facing block incl. efficacy endorsement |
| `write/writer.py` | `MemoryWriter` (`IncrementalPostRunHook`): extract → reconcile → author exemplars → restamp gains → harm-evict, one lock |
| `write/librarian.py` | LLM card authoring: reconcile diffs into NEW/DUPLICATE/MERGE cards (byte-identical descriptions dedup to a provenance bump, never a second id), author exemplar prose; optional novelty-admission gate on NEW idea cards |
| `gigaevo/llm/agents/admission_novelty.py` (outside this package) | `NoveltyAdmissionAgent`: keep/reject an idea card on novelty vs the mutator's prior (`writer.novelty_admission_gate`, off by default pending its A/B) |
| `write/admission.py` | `CardAdmissionGate` (sole harm gate; harm deletions tombstone the id against re-admission for the run) + `WriteLedger` (`write_ledger.jsonl`) |
| `write/stats.py` | `CardStatsUpdater`: base-relative gain attribution + bank-wide restamp (credit that no longer resolves to a banked card is dropped); seeds each new insight card's `founding` gain event (signed parent→child delta) and preserves it across restamps |
| `write/merge.py` | `DedupPolicy` + card merge semantics |
| `write/consolidation.py` | Near-duplicate consolidation: inline same-batch pass over each increment's freshly-ADDED cards (`consolidate_written`) + throttled background pass over the whole bank |
| `write/eviction.py` | `CardScorer` / `CardValueScorer` Protocols, `BirthFailureEvictor` (catastrophic origin failure), `HarmEvictor` (later-use harm), `PolicyNonViableEvictor` (active-policy non-viable cards), `CompositeEvictor`, `NullEvictor` |
| `write/extraction.py` | `ProgramRecordExtractor` — eligible records via strict `MetricsContext` validity |
| `provider.py` | `MemoryProvider` ABC + `NullMemoryProvider` / `ReaderMemoryProvider` / `StaticLeverMemoryProvider` |
| `live_memory_hook.py` | `LiveMemoryRefreshHook` — periodic in-run writer sweeps (`post_step_hook`) |

## Assembly — YAML `${ref:}` graph, no assembler

There is no `MemorySystem`, no factory glue, no enable flags. Each
`config/memory/{none,reader,writer,full,static}.yaml` arm declares the same
reader/writer component graph and swaps `_target_`s (Null variants for disabled
sides); components are declared once and shared by reference:

- pipelines consume `${ref:memory.provider}`
- engines consume `${ref:memory.write.post_run_hook}` as their `post_run_hook`
  (`memory/write=none` resolves to `NullPostRunHook`, writer-enabled modes
  resolve to `${ref:memory.writer}`)
- live writer mode also installs `LiveMemoryRefreshHook` as the global
  `post_step_hook`

See [`docs/memory.md`](../../docs/memory.md) for the arm matrix, component
groups, and launch recipes.

## Persistence layout (per run, under `checkpoint_dir`)

| File | Writer | Contents |
|---|---|---|
| `cards.json` | `CardBank` | The bank: `{"cards": {id: card}}`, atomic rewrite per save |
| `chroma/` | `VectorIndex` | Vector collections, one per embed scope; rebuilt from the bank on demand. Holds `embed_fingerprint.json` — reopening under a changed `embedding_model` raises rather than rank against stale vectors |
| `write_ledger.jsonl` | `WriteLedger` | Append-only admission/eviction verdicts; outcomes: `added` / `updated` / `merged` / `rejected_harm` / `rejected_novelty` / `evicted` (a benign no-op ingest — `DISCARDED` — writes no row) |
| `memory_events.jsonl` | `events.py` sink | Every memory event, one JSON row each |

The bank is the source of truth; index writes are best-effort and heal on
rebuild. A fresh store over an existing `checkpoint_dir` picks the bank up
from disk (this is how a `memory=reader` run consumes a bank a prior
`memory=writer` run built — covered end-to-end by `tests/memory/test_e2e.py`).

## Observability

All modules log under `[Memory][<Component>]`; single-grep `\[Memory\]\[`
covers the subsystem. `memory_events.jsonl` is the primary trace
(read decisions, research steps, auction slates, budget caps, store
writes/syncs, gain restamps, eviction sweeps, consolidation passes).
First stop for debugging:

```bash
python tools/memory_event_report.py <run-dir>   # summarizes events + ledger + bank
python tools/analyze_bandit_health.py <run-dir> # auction/posterior health
```

## Configuration boundaries

All wiring comes through Hydra — no env-var cascade. The only `os.environ`
interaction is `storage/hf_cache.py` (HF cache dir + hub download timeouts,
applied before the embedding model downloads).
