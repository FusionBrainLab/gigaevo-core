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
`write/` never import each other (the write-side eviction surface — the
`Evictor` Protocol + `NullEvictor` — is self-contained in `write/`). Only
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
| `storage/base.py` | `MemoryStore` ABC: `save/get/delete/snapshot/merge_retire`, `nearest()`, `research()`, `rebuild/close/is_ready` |
| `storage/bank.py` | `CardBank`: thread-safe in-proc dict + atomic `cards.json` persist; cold-loaded from disk once at construction |
| `storage/index.py` | `VectorIndex` (in-memory Chroma, one collection per embed scope) |
| `storage/research.py` | LangGraph research agent: plan → retrieve → reflect, ≤`max_iters`, fail-to-empty; prompts under `gigaevo/prompts/retrieval_{planner,reflection}/` |
| `storage/config.py` | `StoreConfig` / `EmbedConfig` / `ResearchConfig` — embedding is config, not code (`embed_scopes` maps scope → card text fields) |
| `storage/local.py` | `LocalMemoryStore` = CardBank ∘ VectorIndex ∘ ResearchAgent |
| `storage/remote.py` | `RemoteMemoryStore` skeleton (httpx; retrieval raises until the remote port lands) |
| `storage/state.py` | `StoreState` transition table (shared `validate_transition` pattern) |
| `storage/hf_cache.py` | HuggingFace cache/timeout shims applied before the embedder downloads |
| `context/models.py` | Shared context models (`GlobalMemoryContext`, `BDCellMemoryContext`) used by reader, shortlist bench, no-card evidence, and write-side baseline fitting |
| `context/no_card.py` | Persisted no-card evidence (`JsonNoCardEvidenceStore`) and static/null provider for legacy policies |
| `read/reader.py` | `MemorySelection` (the read-selection payload: cards, ids, decision id, assignment) + the policy-version digest helpers (`extend_policy_version`, canonical component digest). `memory_v2` assembles the read policy that produces a `MemorySelection` |
| `read/interfaces.py` | Shared read-side Protocols: `Shortlister`, `PolicyDigestProvider` + the `policy_digest` fallback |
| `read/shortlist.py` | `ResearchShortlister`: mutation-grounded query → `store.research()`; passes a digest of the newest banked cards (one description line each) as the planner's context |
| `read/exclusion.py` | `CardExcluder` Protocol, `NullExcluder`, `LineageExcluder` (filter-first lineage gate) |
| `write/writer.py` | `MemoryWriter` (`IncrementalPostRunHook`): extract → reconcile → author exemplars → restamp gains → configured eviction sweep (`NullEvictor` by default), one lock |
| `write/crediting.py` | `EffectEstimator` seam: one `InjectionOutcome` → a `Measurement` (base-relative gain attribution) |
| `write/librarian.py` | LLM card authoring: reconcile diffs into NEW/DUPLICATE/MERGE cards (byte-identical descriptions dedup to a provenance bump, never a second id), author exemplar prose; optional novelty-admission gate on NEW idea cards |
| `gigaevo/llm/agents/admission_novelty.py` (outside this package) | `NoveltyAdmissionAgent`: keep/reject an idea card on novelty vs the mutator's prior (`writer.novelty_admission_gate`, off by default pending its A/B) |
| `write/admission.py` | Transactional admit/merge/provenance and fresh guarded retirement; harm deletions tombstone ids + `WriteLedger` |
| `write/stats.py` | `CardStatsUpdater`: base-relative gain attribution + bank-wide restamp (credit that no longer resolves to a banked card is dropped); seeds each new insight card's `founding` gain event (signed parent→child delta) and preserves it across restamps |
| `write/merge.py` | `DedupPolicy` + card merge semantics |
| `write/consolidation.py` | Near-duplicate consolidation: inline same-batch pass over each increment's freshly-ADDED cards (`consolidate_written`) + throttled background pass over the whole bank |
| `write/eviction.py` | `Evictor` Protocol, `NullEvictor` (the `memory=v2` default), and the shared hard-sign foreign-retention veto |
| `write/extraction.py` | `ProgramRecordExtractor` — eligible records via strict `MetricsContext` validity |
| `provider.py` | `MemoryProvider` ABC + `NullMemoryProvider` / `LeasedMemoryProvider` |
| `live_memory_hook.py` | `LiveMemoryRefreshHook` — periodic in-run writer sweeps (`post_step_hook`) |

## Assembly — YAML `${ref:}` graph, no assembler

There is no `MemorySystem`, no factory glue, no enable flags. Each
`config/memory/{none,v2}.yaml` arm declares the same component graph and swaps
`_target_`s (Null variants for the disabled read/write side); components are
declared once and shared by reference:

- pipelines consume `${ref:memory.provider}`
- engines consume `${ref:memory.write.post_run_hook}` as their `post_run_hook`
  (`memory/write=none` resolves to `NullPostRunHook`, writer-enabled modes
  resolve to `${ref:memory.writer}`)
- live writer mode also installs `LiveMemoryRefreshHook` as the global
  `post_step_hook`

See [`docs/memory.md`](../../docs/memory.md) for the arm matrix, component
groups, and launch recipes.

## Storage layout (per run, under `checkpoint_dir`)

| File | Writer | Contents |
|---|---|---|
| `cards.json` | `CardBank` | The bank: `{"cards": {id: card}}`, atomic rewrite per save |
| `write_ledger.jsonl` | `WriteLedger` | Append-only admission/eviction verdicts; outcomes: `added` / `updated` / `merged` / `rejected_harm` / `rejected_novelty` / `evicted` (a benign no-op ingest — `DISCARDED` — writes no row) |
| `memory_events.jsonl` | `events.py` sink | Every memory event, one JSON row each |

The bank is the source of truth. `VectorIndex` is process-local and in-memory;
it rebuilds from `cards.json` at startup and after cross-process bank refreshes,
while incremental index writes remain best-effort. Existing persisted `chroma/`
directories are ignored. A fresh store over an existing `checkpoint_dir` picks
the bank up from disk, so a later run consumes a bank a prior run built.

## Observability

All modules log under `[Memory][<Component>]`; single-grep `\[Memory\]\[`
covers the subsystem. `memory_events.jsonl` is the primary trace
(read decisions, research steps, store writes/syncs, gain restamps,
eviction sweeps, consolidation passes).
First stop for debugging:

```bash
python tools/memory_card_health.py <run-dir>    # card bank structural/integrity snapshot
```

## Configuration boundaries

All wiring comes through Hydra — no env-var cascade. The only `os.environ`
interaction is `storage/hf_cache.py` (HF cache dir + hub download timeouts,
applied before the embedding model downloads).
