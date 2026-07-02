# `gigaevo/memory` — architecture

Cross-run memory for the evolutionary loop: a **write system** distills each
run's mutation diffs and top exemplars into reusable cards, and a **read
system** researches that bank and injects the most promising cards into
mutation prompts, tracking each card's realized fitness gain (reputation).

User-facing guide (Hydra knobs, launch recipes, observability):
[`docs/memory.md`](../../docs/memory.md).

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
| `cards.py` | The one `Card` model (`kind ∈ {insight, program}`; program fields kind-gated), `ContextualGain` + `DecisionContext` (gain events), `CardStatsBlock`, `card_brief()` |
| `events.py` | Typed `MemoryEvent`s (canonical monitoring events + per-run `memory_events.jsonl` sink), `memory_event_context` correlation (decision/program/parent ids) |
| `storage/base.py` | `MemoryStore` ABC: `save/get/delete/snapshot/apply_merges`, `nearest()`, `research()`, `rebuild/close/is_ready` |
| `storage/bank.py` | `CardBank`: in-proc dict + atomic `cards.json` persist; mtime watermark for external-writer visibility |
| `storage/index.py` | `VectorIndex` (Chroma, one collection per embed scope) |
| `storage/research.py` | LangGraph research agent: plan → retrieve → reflect, ≤`max_iters`, fail-to-empty; prompts under `gigaevo/prompts/retrieval_{planner,reflection}/` |
| `storage/config.py` | `StoreConfig` / `EmbedConfig` / `ResearchConfig` — embedding is config, not code (`embed_scopes` maps scope → card text fields) |
| `storage/local.py` | `LocalMemoryStore` = CardBank ∘ VectorIndex ∘ ResearchAgent |
| `storage/remote.py` | `RemoteMemoryStore` skeleton (httpx; retrieval raises until the remote port lands) |
| `storage/state.py` | `StoreState` transition table (shared `validate_transition` pattern) |
| `storage/hf_cache.py` | HuggingFace cache/timeout shims applied before the embedder downloads |
| `read/reader.py` | `MemoryReader` facade: shortlist → reputation → auction → budget → render; every stage a Protocol; fails to empty selection |
| `read/shortlist.py` | `ResearchShortlister`: mutation-grounded query → `store.research()` |
| `read/reputation.py` | `BetaBinomialReputation` (+ `BDProximityReputation` cell-local variant); posterior/efficacy math over gain events (`block_from_events`) |
| `read/auction.py` | `ThompsonAuctioneer` / `EVThompsonAuctioneer` + `TopThetaBudgeter` / `TopBidBudgeter` |
| `read/exclusion.py` | `NullExcluder`, `LineageExcluder` (filter-first lineage gate) |
| `read/render.py` | `EfficacyCardRenderer` — card → mutator-facing block incl. efficacy endorsement |
| `write/writer.py` | `MemoryWriter` (`IncrementalPostRunHook`): extract → reconcile → author exemplars → restamp gains → harm-evict, one lock |
| `write/librarian.py` | LLM card authoring: reconcile diffs into NEW/DUPLICATE/MERGE cards, author exemplar prose |
| `write/admission.py` | `CardAdmissionGate` (sole harm gate) + `WriteLedger` (`write_ledger.jsonl`) |
| `write/stats.py` | `CardStatsUpdater`: base-relative gain attribution + bank-wide restamp |
| `write/merge.py` | `DedupPolicy` + card merge semantics |
| `write/consolidation.py` | Throttled background near-duplicate consolidation pass |
| `write/eviction.py` | `CardScorer` Protocol, `HarmEvictor`, `NullEvictor` |
| `write/extraction.py` | `ProgramRecordExtractor` — eligible records via strict `MetricsContext` validity |
| `provider.py` | `MemoryProvider` ABC + `NullMemoryProvider` / `ReaderMemoryProvider` / `StaticLeverMemoryProvider` |
| `live_memory_hook.py` | `LiveMemoryRefreshHook` — periodic in-run writer sweeps (`post_step_hook`) |

## Assembly — YAML `${ref:}` graph, no assembler

There is no `MemorySystem`, no factory glue, no enable flags. Each
`config/memory/{none,reader,writer,full,static}.yaml` arm declares the same
two consumer-facing nodes and swaps `_target_`s (Null variants for disabled
sides); components are declared once and shared by reference:

- pipelines consume `${ref:memory.provider}`
- engines consume `${ref:memory.writer}` as their `post_run_hook`

See [`docs/memory.md`](../../docs/memory.md) for the arm matrix, component
groups, and launch recipes.

## Persistence layout (per run, under `checkpoint_dir`)

| File | Writer | Contents |
|---|---|---|
| `cards.json` | `CardBank` | The bank: `{"cards": {id: card}}`, atomic rewrite per save |
| `chroma/` | `VectorIndex` | Vector collections, one per embed scope; rebuilt from the bank on demand |
| `write_ledger.jsonl` | `WriteLedger` | Append-only admission/eviction verdicts |
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
