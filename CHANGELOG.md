# CHANGELOG

<!-- version list -->

## v2.6.1 (2026-06-13)

### Bug Fixes

- **memory**: Read-path defaults + slate hygiene (review iter 2, set C)
  ([#277](https://github.com/KhrulkovV/gigaevo-core-internal/pull/277),
  [`8bf566b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8bf566baccfa0c0e9969fcceb981d9feba47502e))

- **memory**: Review iteration 2 — tracker correctness, write-path hardening, read-path defaults
  ([#277](https://github.com/KhrulkovV/gigaevo-core-internal/pull/277),
  [`8bf566b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8bf566baccfa0c0e9969fcceb981d9feba47502e))

- **memory**: Tracker correctness sweep (review-2 commit A)
  ([#277](https://github.com/KhrulkovV/gigaevo-core-internal/pull/277),
  [`8bf566b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8bf566baccfa0c0e9969fcceb981d9feba47502e))

- **memory**: Write-path hardening (review-2 commit B)
  ([#277](https://github.com/KhrulkovV/gigaevo-core-internal/pull/277),
  [`8bf566b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8bf566baccfa0c0e9969fcceb981d9feba47502e))

- **test**: Mock SentenceTransformer in fast-analyzer factory test
  ([#277](https://github.com/KhrulkovV/gigaevo-core-internal/pull/277),
  [`8bf566b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8bf566baccfa0c0e9969fcceb981d9feba47502e))


## v2.6.0 (2026-06-13)

### Bug Fixes

- **memory**: Review iteration 1 — paper-readiness fixes + canonical-dedup wiring
  ([#276](https://github.com/KhrulkovV/gigaevo-core-internal/pull/276),
  [`b40d28c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b40d28c6d193754735ef698e56c8e2e5ff5b161e))

- **memory**: Review-1 paper-readiness fixes — honest ledger ids, hardened reload/posterior,
  factory-aligned defaults, serialized tracker sweeps
  ([#276](https://github.com/KhrulkovV/gigaevo-core-internal/pull/276),
  [`b40d28c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b40d28c6d193754735ef698e56c8e2e5ff5b161e))

### Features

- **memory**: Wire canonical-key dedup producer into enrich_with_verification
  ([#276](https://github.com/KhrulkovV/gigaevo-core-internal/pull/276),
  [`b40d28c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b40d28c6d193754735ef698e56c8e2e5ff5b161e))


## v2.5.0 (2026-06-13)

### Features

- **memory**: Paper-readiness hardening — exposure observability, fail-loud LLM paths, config wiring
  ([#275](https://github.com/KhrulkovV/gigaevo-core-internal/pull/275),
  [`490452f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/490452f965e576eaf49090668a8da4e13ee43891))


## v2.4.0 (2026-06-12)

### Bug Fixes

- **ci**: Scope dry_run resolver stubs + drop tests/problems; CI on py3.12
  ([#274](https://github.com/KhrulkovV/gigaevo-core-internal/pull/274),
  [`3917da2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3917da25e8a19109a55e15ed6b42771404910699))

- **ci**: Un-stick RedisInstanceLock renewal cancel + survive future hangs
  ([#274](https://github.com/KhrulkovV/gigaevo-core-internal/pull/274),
  [`3917da2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3917da25e8a19109a55e15ed6b42771404910699))

- **deps**: Minimal install works end-to-end + promote lightweight matplotlib
  ([`5d3ab2b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5d3ab2b4b2fe1d64e3e7136a6c16c5ae25e61a5c))

- **memory**: Bounded parallel refresh + structured-output cutover + observability polish
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **memory**: Direction- and sentinel-correct stats tracking
  ([#272](https://github.com/KhrulkovV/gigaevo-core-internal/pull/272),
  [`e07ec8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e07ec8a949a03e92baf9d2e228f231b583a95d05))

- **memory**: Final-review pass — CLI direction flag + sentinel-ceiling test
  ([#272](https://github.com/KhrulkovV/gigaevo-core-internal/pull/272),
  [`e07ec8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e07ec8a949a03e92baf9d2e228f231b583a95d05))

- **memory**: Ledger rows for sweep evictions + legacy-backend dropped-component warning
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Resolve mypy type errors flagged by pre-push gate
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Review iteration 1 — write-side evictor/dedup plumbing, sweep_harmful facade,
  traceback logging ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Route HF-cache env diagnostics through loguru
  ([#273](https://github.com/KhrulkovV/gigaevo-core-internal/pull/273),
  [`f4dd916`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f4dd9160f01711045f0ed6c18bfb250e33f9e766))

- **memory**: Route IdeaTracker program-card posteriors through reputation singleton
  ([#270](https://github.com/KhrulkovV/gigaevo-core-internal/pull/270),
  [`2812ad7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2812ad76c9de65f7eb70ae338e3ef67cb71a7c5f))

- **metrics**: Recompute valid-frontier on every drain in iteration order
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **metrics_tracker**: Use program.iteration as step for valid/program/<key>
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **monitoring**: Extend frontier line + annotate jumps; prune buffer on clear_series
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **storage**: Address PR #267 review iteration 1
  ([`d381257`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d3812572990adf14bc1b8b08b01db7c0df48b8b0))

- **storage**: Review iteration 2 — disk-aware run.py messaging + status-set crash reconciliation
  ([`a6dc24f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a6dc24f0b81892afbade34a8a67d627a47c3bf01))

- **tabular**: Add fit_predict code skeleton to task-description CONTRACT
  ([#265](https://github.com/KhrulkovV/gigaevo-core-internal/pull/265),
  [`2728702`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/272870281133c135dbb1820e78b091db34f4719e))

- **tabular**: Align seed programs with the task-description protocol
  ([#264](https://github.com/KhrulkovV/gigaevo-core-internal/pull/264),
  [`4a5e614`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4a5e6142ce25862899e724c4879c827060c80e39))

- **tabular**: Align seeds with task-description protocol + semantic task descriptions
  ([#264](https://github.com/KhrulkovV/gigaevo-core-internal/pull/264),
  [`4a5e614`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4a5e6142ce25862899e724c4879c827060c80e39))

- **tabular**: Prevent programs from poisoning the worker dataset cache
  ([#268](https://github.com/KhrulkovV/gigaevo-core-internal/pull/268),
  [`d783f11`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d783f11a8e73b894e2eebb5b372338f4e978e92a))

- **types**: Satisfy mypy in memory_selector + CMA pipeline
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

### Build System

- **release**: Also strip compiled extensions and graphify-out
  ([`dcac51b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dcac51bfbb27917e4e80d706632c7bd1c1bbee1f))

- **release**: Teach sync_public.sh glob/regex prefixes + size gate
  ([`4855cff`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4855cff7861b056ec66c21ed61325ada096c1585))

### Chores

- **config**: Drop stale top-level top_p / top_k Hydra constants
  ([#257](https://github.com/KhrulkovV/gigaevo-core-internal/pull/257),
  [`4508e20`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4508e209b9f08b3cfa463e9138c46443f1af2f0c))

- **gitignore**: Untrack ~2.8GB bench fixtures + add runtime artifact rules
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **lint**: Satisfy ruff check + format for pre-push gate
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **lint**: Sort post-Agg imports in live_frontier_compare
  ([`b2db237`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b2db237770fb05771966dc972306245b4d5267af))

- **memory**: Prefix all memory logs [Memory][Locus]; untrack catboost
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

### Documentation

- **memory**: Field descriptions on every Pydantic model in the card paths
  ([#271](https://github.com/KhrulkovV/gigaevo-core-internal/pull/271),
  [`2ed48b7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2ed48b706e0605647cee96b4bb20dd74726e093e))

- **memory**: Refresh stale memory READMEs for release
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **tabular**: Canonical README for tabular suite + USAGE pointer
  ([`4b2d380`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4b2d380754f40d95f1350cff8c57dc5f2163034a))

### Features

- **cli**: -r accepts disk storage paths alongside prefix@db specs
  ([`1d62f22`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1d62f22eb7fad9f80442c6adc49c9b0d8e105e66))

- **core**: Curated framework fixes + tabular-regression QD support
  ([`52ebc6b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/52ebc6b107b730da785972309094019fdcce0e23))

- **evolution**: Content-addressable stage-output store for debug observability
  ([#266](https://github.com/KhrulkovV/gigaevo-core-internal/pull/266),
  [`50a3b10`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/50a3b10e75d868b325797d5e287c2b489e16152d))

- **llm,run**: Global LLM-concurrency cap + always-finalize hooks
  ([`0190f0e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0190f0e73cc7292e608b763c0234400915544e85))

- **memory**: Beta-Binomial confidence gate + Thompson auction for shared-bank cards
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Modular memory core, Hydra config groups, canonical event logging
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Single-owner EfficacyScorer for card- and idea-side gain stats
  ([#272](https://github.com/KhrulkovV/gigaevo-core-internal/pull/272),
  [`e07ec8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e07ec8a949a03e92baf9d2e228f231b583a95d05))

- **memory**: Surface per-card efficacy to mutator; drop dead usage blob
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Write-side integrity gate for card bank (GAM Phase 1)
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **metrics**: Storage-agnostic metrics history — disk JSONL backend + reader protocol
  ([`a650473`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a65047351b82acd0b16768f20b35de5d8d7a47f9))

- **monitoring**: Dynamic ETA ticker — periodic log driven by observed throughput
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **prompts**: Land v4 packed-grammar mutation prompt + strip CONTEXT scaffolding
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **stages**: Salvage best-so-far on cancellation for Optuna and CMA stages
  ([`4b1ce31`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4b1ce31ce416ed9cd81e627ce8366914e2266ca1))

- **storage**: Disk-backed ProgramStorage + ArchiveStorage backend
  ([`45b38a7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/45b38a7e48b445b2cc27e29eeb04c5e6300bdf8d))

- **tabular**: 10 dataset dirs (task_description + symlinks) + description generator
  ([`28bd9b2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/28bd9b20a14d25e3f4b59f8b7f46f238f0e9ef20))

- **tabular**: 5 seed programs per task type (reg/binclass/multiclass)
  ([`848274e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/848274e69f1b674161b935e5ee15788f32b13653))

- **tabular**: Build(name) validator + scorer with CV/holdout + task-typed keys
  ([`c738e77`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c738e7775f0269b780474333b44220b596e35769))

- **tabular**: Data loader with ordinal-encoded categoricals + column descriptions
  ([`21ad0f4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/21ad0f45cecb4923685cdaacd574a47598c52f2b))

- **tabular**: Generalised robustness BD probe (local Lipschitz + OOD slope)
  ([`b46f508`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b46f5087a49379e4c541f26be97f913107546850))

- **tabular**: Per-task-type metrics.yaml with key-set parity tests
  ([`3d938fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3d938fef87a5c0fb4d10d701581d0b64586e3909))

- **tabular**: Scale-free metric helpers (regression + classification)
  ([`b0450b0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b0450b07d04a869f08dc724c80b47e543d991298))

- **tabular**: Semantic column meaning + kaggler strategy hints in task descriptions
  ([#264](https://github.com/KhrulkovV/gigaevo-core-internal/pull/264),
  [`4a5e614`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4a5e6142ce25862899e724c4879c827060c80e39))

- **tabular**: Validator shim (sys.path[0]) + standalone test scorer
  ([`252ff3a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/252ff3a84d7d425326ec9ad5e47c3e42b3d7573c))

### Performance Improvements

- **exec**: Cap intra-worker thread libs to (cpu//8) to stop oversubscription
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **memory**: Bound live-refresh sweep and parallelise classifier
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

### Refactoring

- **config**: Storage backend as first-class Hydra group
  ([`afe050d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/afe050d017390e83dfa0abfc248212536fa47525))

- **memory**: Cut over to LangChain structured output + v3 selector prompts
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **memory**: DecisionMetrics/AuditMetrics split + CardStatsStamper; delete TieredAdmitter
  ([#272](https://github.com/KhrulkovV/gigaevo-core-internal/pull/272),
  [`e07ec8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e07ec8a949a03e92baf9d2e228f231b583a95d05))

- **memory**: Finish typed-card boundaries — vendor meta envelope, typed Idea statistics, canonical
  mutation-output key ([#271](https://github.com/KhrulkovV/gigaevo-core-internal/pull/271),
  [`2ed48b7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2ed48b706e0605647cee96b4bb20dd74726e093e))

- **memory**: GenerationBucketer + typed EfficacyEvent rows
  ([#272](https://github.com/KhrulkovV/gigaevo-core-internal/pull/272),
  [`e07ec8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e07ec8a949a03e92baf9d2e228f231b583a95d05))

- **memory**: Make LangChainLLMService structured-output method per-deployment
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **memory**: Remove dead usage-tracking store (converge on IntroGain)
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **memory**: Self-review pass — typed auction slate, CardAlias, IdeaStats everywhere
  ([#271](https://github.com/KhrulkovV/gigaevo-core-internal/pull/271),
  [`2ed48b7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2ed48b706e0605647cee96b4bb20dd74726e093e))

- **memory**: Typed Pydantic cards everywhere — remove dict plumbing from card paths
  ([#271](https://github.com/KhrulkovV/gigaevo-core-internal/pull/271),
  [`2ed48b7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2ed48b706e0605647cee96b4bb20dd74726e093e))

- **memory**: WriteStatKey enum for write_stats counters; enum-keyed dedup fallbacks
  ([#271](https://github.com/KhrulkovV/gigaevo-core-internal/pull/271),
  [`2ed48b7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2ed48b706e0605647cee96b4bb20dd74726e093e))

- **problems**: Rename optimization → tabular_regression, redesign contract
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **storage**: Pure ProgramStorage abstraction, Hydra-confined construction
  ([`5213e9a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5213e9a7babe17bc016044df94cee11cafecca29))

- **tabular**: Remove tabular_regression (replaced by tabular/california)
  ([`4030df9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4030df9aa748c4193d78875c77e09d36e6fb92d7))

### Testing

- **memory**: Align selector tests with structured-output select() API
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **memory**: Fix CI integration tests after structured-output cutover
  ([#258](https://github.com/KhrulkovV/gigaevo-core-internal/pull/258),
  [`47f68e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47f68e3f3f026d638ecb8e3cf6f820c48a76e96f))

- **memory**: Guard platform-injection test behind gigaevo_memory importorskip
  ([#269](https://github.com/KhrulkovV/gigaevo-core-internal/pull/269),
  [`8663f30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8663f3062853d3ec534e8652b5fa3b29f9ef6675))

- **tabular**: ProblemContext loads all 10 dataset dirs
  ([`3038532`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3038532b5f18f493498ba29214fcbeb590fb2bba))

- **tabular**: Scaffold tabular test package + conftest
  ([`f8c7e69`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f8c7e6972c50b21015177f076829e64e6b8a937f))


## v2.3.1 (2026-05-22)

### Bug Fixes

- **archive**: Idempotent re-add for elites already in the archive
  ([#256](https://github.com/KhrulkovV/gigaevo-core-internal/pull/256),
  [`b3d3232`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b3d3232fb294d4d2bca42b042d96b6ac847ea57c))

### Chores

- **config**: Drop stale experiment-specific YAMLs
  ([`82caec1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/82caec1320ce6f0de0aa6457a5b89781996666f3))

- **docs**: Trim references to deleted configs
  ([`e10bc71`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e10bc71fc4e7e3ec527320cf3baa0626c69697a5))

- **problems**: Delete dead heilbron_* problem directories
  ([`85b5fc0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/85b5fc02910dc7256fc5352fc6d9e8a654f21d65))

- **tests**: Drop tests for deleted configs + problems
  ([`845911d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/845911df3610921261b494a652b0a43402a7c99d))


## v2.3.0 (2026-05-22)

### Features

- Add spherical_codes_improver_v2 + safe_mode=False in default pipeline
  ([`3184550`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3184550db746e361d23cfc12cd1e4f90ae37a099))


## v2.2.0 (2026-05-22)

### Features

- **engine**: Add coalesce_refresh config knob (default True)
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))

- **engine**: Add ParentRefresher._fresh + mark_children_completed
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))

- **engine**: Add ParentRefresher.refresh_if_stale (coalesced mode)
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))

- **engine**: Coalesce parent refreshes across concurrent mutations (default ON)
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))

- **engine**: Dispatch run_one_mutant on coalesce_refresh
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))

- **engine**: Ingestor invalidates parents of completed children
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))

### Testing

- Pin _build_engine to max_in_flight=1 for chained-mutation tests
  ([#254](https://github.com/KhrulkovV/gigaevo-core-internal/pull/254),
  [`46e54a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46e54a0a60bd4d18aab630fba936e2324a6352ac))


## v2.1.3 (2026-05-21)

### Bug Fixes

- **cli**: Flush --db N no longer kills workers on db N0..N9
  ([`044871a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/044871a49a909a2ebcb24fc53cc423e59c097a0e))


## v2.1.2 (2026-05-21)

### Bug Fixes

- **config**: Default checkpoint_dir under Hydra run dir
  ([#253](https://github.com/KhrulkovV/gigaevo-core-internal/pull/253),
  [`702c84d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/702c84db62fdb48c3c00224601df8541a3992d5c))

- **config**: Default checkpoint_dir under Hydra run dir for per-run memory isolation
  ([#253](https://github.com/KhrulkovV/gigaevo-core-internal/pull/253),
  [`702c84d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/702c84db62fdb48c3c00224601df8541a3992d5c))

- **memory,selector**: Thread per-run checkpoint_dir + restore FP+tournament default
  ([#253](https://github.com/KhrulkovV/gigaevo-core-internal/pull/253),
  [`702c84d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/702c84db62fdb48c3c00224601df8541a3992d5c))


## v2.1.1 (2026-05-21)

### Bug Fixes

- Backport PEP-695 generics + nested-quote f-strings to Python 3.11
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- **pipelines**: Lazy-load CMA/Optuna stages in default_pipelines
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- **trackers**: Lazy-load tensorboard/wandb backends so [tracking] is truly opt-in
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

### Chores

- Lower Python floor 3.12 → 3.11
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- Minimal core + opt-in extras, Python 3.11 floor, config prune
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- **config**: Prune 15 stale/redundant YAML configs
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- **deps**: Split pyproject into minimal core + opt-in extras
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

### Continuous Integration

- **chains**: Gate mmar-carl extra on Python >=3.12
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- **memory-platform**: Gate gigaevo-memory on Python >=3.12
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

- **test**: Install opt-in extras needed by the test suite
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

### Documentation

- Install ladder for minimal core + opt-in extras
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))

### Refactoring

- **problems**: Rename chains/neurips_test → chains/nlp
  ([#252](https://github.com/KhrulkovV/gigaevo-core-internal/pull/252),
  [`d6ac3fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6ac3fe380705cdbd63ad2588816149d28615af7))


## v2.1.0 (2026-05-21)

### Bug Fixes

- **chains**: Wire RunnerConfig + add get_violated_constraints for IFBench
  ([#251](https://github.com/KhrulkovV/gigaevo-core-internal/pull/251),
  [`6d2161f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d2161f7adaab87b6853aa20a17bad53d7ee7a94))

### Features

- **chains**: CARL chain runner (port from #184)
  ([#251](https://github.com/KhrulkovV/gigaevo-core-internal/pull/251),
  [`6d2161f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d2161f7adaab87b6853aa20a17bad53d7ee7a94))

- **chains**: CARL-backed step-batched chain runner + RunnerConfig presets
  ([#251](https://github.com/KhrulkovV/gigaevo-core-internal/pull/251),
  [`6d2161f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d2161f7adaab87b6853aa20a17bad53d7ee7a94))

- **chains**: Port paper task suites + chain canonical_benchmark profile
  ([#251](https://github.com/KhrulkovV/gigaevo-core-internal/pull/251),
  [`6d2161f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d2161f7adaab87b6853aa20a17bad53d7ee7a94))


## v2.0.0 (2026-05-21)

### Bug Fixes

- **canonical_benchmark**: Make --llm-base-url and --model-name required
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **canonical_benchmark**: Pass --minimize for lower-is-better problems
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **evolution-stats**: Iteration-window aggregation + snapshot bump
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **intra-memory**: Ship unified diff (not full code) per child + soften mutator "untried
  directions" preference ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **memory**: Pick OPENROUTER_API_KEY when LLM_BASE_URL targets OpenRouter
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **memory**: Rename IdeaTracker._run → run_increment to match call sites
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **stats**: Rank line dropout when focal missing from snapshot
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **suggester**: Escape literal {} braces in lineage-exhaustion sub-bullet
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

### Chores

- Gitignore output/runs/tool-caches + capture pre-loop audit MDs
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Remove dead ProgramRecord.insights field
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Ruff fix + format on tools/pseudo_evo_bench (pre-loop)
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **merge-prep**: Drop 57 auto-loop / pseudo-evo / lineage / insights audits + tools
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **v2.0.0**: Commit erdos-minimum-overlap hand-recompute footnote
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **v2.0.0**: Drop broken tools/benchmarks/bench_{multirun,steady_state}.py
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

### Documentation

- Align user-facing docs with v2.0.0 contract
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Intra/extra memory mode guide + USAGE / MEMORY_ARCHITECTURE cross-links
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Promote 3 live-feature docs, delete the rest
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **audit**: V3.1 decision tree — counterfactual audit on 289 prior-run programs (DBs 13/14/15)
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **audit**: V3.1 mutation decision tree — channels, gate, decision table, worked examples
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **audit**: V3.1 tree — soften row 13 target-awareness + add rows 19 / 19a per counterfactual audit
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **auto-optimize-loop**: Cycle-2 PROPOSE — variance-floor replicate #2 (db=13)
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **auto-optimize-loop**: Cycle-2.5 PROPOSE — 4th NO-EDIT variance-floor replicate (db=14)
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **auto-optimize-loop**: Finalize cycle-0 Analytics + cycle-1 PROPOSE + scope-expansion note
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **auto-optimize-loop**: Spec + reference schemas + history/patterns scaffolds
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **canonical_benchmark**: Bring frozen-knobs line in line with v2.0.0 defaults
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **usage**: Drop spurious num_parents requirement + ruff format
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

### Features

- V2.0.0 — intra-memory pipeline becomes default
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **context**: R1+R3 — archive-quartile regime in mutation_context render
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **context**: R7+R8 v3.1 — archive distribution with worst/median/best + archive-percentile token
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **memory**: DAG-native intra+extra memory pipeline (per-parent lineage card + live global cards)
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **monitoring**: File emit target writes frontier_<metric>.png each tick
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **prescriptive**: MutationSuggestionStage + EvolutionaryStatistics wiring
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **prompts**: R6 — archive-quartile archetype gate + suggester tag-bias
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **prompts**: R9 v3.1 — archive-percentile gate + qualitative target awareness
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **prompts**: Soften target-awareness clause + add noise-dominated & empty-intra rules
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **stats**: R2 — MAD-based trend noise floor + archive_valid_fitnesses field
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **suggester**: Lineage-exhaustion override in rank-aware ambition
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **suggester**: Rank-aware ambition rule in mutation_suggestions/system.txt
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **suggester**: Revert rank+LEX to 9cca4344 baseline for cycle-6 A/B
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **suggester**: Server-computed EXHAUSTION ALERT banner overrides soft LEX
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **tools**: Trajectory_shape.py — log-based closeout analyzer for cycle comparisons
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

### Refactoring

- Drop dormant ExtraMemoryStage and its cache tests
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Framework defaults match canonical benchmark contract
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **pipeline**: Split standard→intra-only base + intra+extra subclass; add canonical regression
  benchmark ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

### Testing

- Compress 8 tautological prompt-loader tests into 1 parametrized test
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Generalize problem-pinned tests + remove fragile prompt-phrase-pin tests
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Remove xfail markers (delete brittle/unsupported, fix schema drift)
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- Simplify slow tests so suite stays under 5 min
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **config**: Align evolution constants with v2.0.0 defaults
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **prompts**: Drop phrase-pinning trash tests on production prompt files
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))

- **v3.1**: Archive-percentile gate, archive distribution, no Target/Regime tokens
  ([#250](https://github.com/KhrulkovV/gigaevo-core-internal/pull/250),
  [`a5ea71b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5ea71b7a5f60169141a9ed2c0517eedd2b7d5ad))


## v1.31.0 (2026-05-16)

### Features

- **cli**: --from-csv for `gigaevo plot comparison` (#247)
  ([#248](https://github.com/KhrulkovV/gigaevo-core-internal/pull/248),
  [`5dede22`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5dede228a1af3089de7f3fbe42a9098689eebd04))


## v1.30.0 (2026-05-14)

### Bug Fixes

- **profiler**: Caption + hover + stable colors (#230 #231 #238)
  ([#239](https://github.com/KhrulkovV/gigaevo-core-internal/pull/239),
  [`843919c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/843919cc73b3b7f2cdc5cfaf90f622958a7c9ce5))

- **telegram_notify**: Use datetime.UTC alias (mypy unblocker)
  ([#242](https://github.com/KhrulkovV/gigaevo-core-internal/pull/242),
  [`afecfe3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/afecfe349ea3063a8cf5ef30452d8f4104ba9828))

### Continuous Integration

- Bump test job timeout 30→45 min
  ([#244](https://github.com/KhrulkovV/gigaevo-core-internal/pull/244),
  [`389c37b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/389c37bb945b7f615bbf06f80cf5976de7afbd05))

### Documentation

- Add canonical-docs rule to CLAUDE.md + refresh tools/README
  ([#245](https://github.com/KhrulkovV/gigaevo-core-internal/pull/245),
  [`f678f22`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f678f22add4e9d6aaa0ddddafaa9967be9f30235))

### Features

- **cli**: Gigaevo metrics — dump metrics from Redis (#235)
  ([#240](https://github.com/KhrulkovV/gigaevo-core-internal/pull/240),
  [`e1363bd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e1363bdd9e8a3613375b3c0771f97930bcd91060))

- **monitoring**: Live frontier-compare loop
  ([#242](https://github.com/KhrulkovV/gigaevo-core-internal/pull/242),
  [`afecfe3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/afecfe349ea3063a8cf5ef30452d8f4104ba9828))

- **monitoring**: Live frontier-compare loop (#236)
  ([#242](https://github.com/KhrulkovV/gigaevo-core-internal/pull/242),
  [`afecfe3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/afecfe349ea3063a8cf5ef30452d8f4104ba9828))

### Refactoring

- **engine**: Rename EngineMetrics.total_mutants → iteration (#232 slice 1)
  ([#243](https://github.com/KhrulkovV/gigaevo-core-internal/pull/243),
  [`46693cc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46693cc2b11d4761f72e730ef6ce9fd743010d39))


## v1.29.0 (2026-05-14)

### Bug Fixes

- 4 test failures + 1 mypy error on main
  ([`e2c40ae`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e2c40ae035832cb635ee132661b054178de82d44))

- _card_type crashes on Pydantic models (Bug #3, PR #161)
  ([`de272dc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/de272dc5aef5dad8181b7537f03920cef5486182))

- _card_type Pydantic crash + E2E pipeline tests (Bug #3, PR #161)
  ([`b251f3e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b251f3ed751903cb352479f001b13591e13561fe))

- _card_type Pydantic crash + E2E pipeline tests (Bug #3, PR #161)
  ([`c933b94`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c933b94b39cdd604ecbb6e93f140300527336441))

- Add _aggregator + .metrics to mock Program in two_pass test
  ([`f534d3a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f534d3ad6600c59f89a1eba581f0ca9671fa3e19))

- Add precondition asserts to _refresh_prompts_from_fetcher for type narrowing
  ([`537e219`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/537e21983d816c2be0b13d53f724666fc38f9a1b))

- Adversarial sync deadlock — publish progress before hook, reset on timeout
  ([`7767a9e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7767a9e603cb77e72bae4d5a5d6590f0fd848b79))

- AdversarialFeedbackPipelineBuilder missing archive_reeval param
  ([`c4d6c51`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c4d6c51b139f7861679e0beb517ac6eeafb56fd1))

- Correct import sort order in test_mutation_operator.py and format test_write_programs.py
  ([`3b97dd9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3b97dd99e7d81eb896e66358490f0a62cd9f5ad7))

- Delete shadowing origin_analysis.py + fix ruff I001 in test file
  ([`8d8697c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8d8697c34b007fcf280df4ffb2cc5df0c93bca87))

- Docs audit, watchdog label fix, script bug fixes
  ([`ef5ec3b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ef5ec3ba2902fc38ea7fefa3ea4cce29891b026b))

- Fitness shown as raw value, not percentage
  ([`20206c8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/20206c8fe9fd4e2507d606142579a31b3f12140a))

- Flatten ideas_tracker aliases (list[dict]) to MemoryCard (list[str])
  ([`d031599`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d03159904eea7a096d758ad75ccbcfb37675fc9b))

- Handle Pydantic models in memory_platform normalize_memory_card
  ([`e4a46bf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e4a46bf4c472e11595f1984e53c3f134798d993d))

- Hover/memory — add steady_state + topology_3d_ret + lpt_chain to match best run
  ([`8b4bf80`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8b4bf80cc4f84d3468cd5de2108c58087ac0d00f))

- Is_inside_triangle accepts single point (2,) + batch (N,2), add shape docs to task descriptions
  ([`c1eca7c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c1eca7c37f7ce075c13d9f7fb0d744b495d4399f))

- Lint cleanup for closeout commit
  ([`0422110`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0422110998af45a67649ec41c037d67613bd612c))

- Lint errors from mypy merge (yaml import alias, missing warnings import)
  ([`d62d2b1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d62d2b10ce69f164e166309a80c213a8a01d7be9))

- Lint errors in ablation_v3_no_deep.py, update prereg_commit
  ([`c129021`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c1290210240bc31193f5fbf501ddb91e84e72423))

- Lint errors in gigaevo/cli/ (import sorting, unused vars)
  ([`2daf19f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2daf19fc20352db74743c01f45f1f1eab08319c1))

- Lint errors in memory_platform test (unused import, sort)
  ([`ed0128b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ed0128bfc4b1b2044772e237928369fa741ae2ef))

- MemoryCard.aliases type list[str] → list[Any]
  ([`03f3a25`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/03f3a253902bd369f063a03137505b3436c890ad))

- MemorySelectorAgent accepts checkpoint_dir/namespace/use_api overrides
  ([`a130913`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a1309131ec270a5826084c089657c60675b95911))

- Move load_dotenv() after imports to fix E402 lint error
  ([`77f7e09`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/77f7e09694f76e6a1e24180ddac798a445457195))

- Pop B seed basin depth + add launch script and watchdog
  ([`5aaeabb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5aaeabbe346f83b2dbee560022009bd644cd23f6))

- Process_cleanup no PYTHONPATH; flush accepts comma-separated --db
  ([`4026269`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4026269bba29e629005df8d8a00bc85a93379e8f))

- Publish programs_processed to Redis after every ingestion batch
  ([`ff42b51`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ff42b518b899f532711cf6cf65849e2f01f82cbc))

- Record_pids.py — fix import order for ruff isort compliance
  ([`1e5c234`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1e5c23478e4827abdb37373ee3ccf59339fca0a8))

- Reduce type: ignore comments and enable mypy config
  ([`817eca8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/817eca8456fa9133cc6a0eab262587512d117c02))

- Reduce type: ignore from 93 to 46 with real code improvements
  ([`efc1c21`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/efc1c214e3f65377209dcf6e68c931427bfc1cc4))

- Remove gigaflow/tools namespace workarounds — fixed in gigaflow
  ([`5a70309`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5a70309c2b0cbdba527cb0dbd09ef35e1126f611))

- Replace missed _synthesize_results reference in _search_local_cards
  ([`97b2cb6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97b2cb651f8aee7c31faece5f7bef9a185eb7ddf))

- Resolve 5 confirmed memory system bugs exposed by adversarial tests
  ([`99cb534`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/99cb534ed954892926f3fd07a07f2b7ea6620f97))

- Resolve 6 chaos-hacker findings (X1, X2, X4, X7/X11, X9)
  ([`e499933`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e49993392e7d2acf6ae443f99f807169988cd3fa))

- Resolve all mypy errors across the codebase (316 → 0)
  ([`b925057`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b925057d188bd464e66810a95ffd8b7e0c5a255e))

- Resolve bugs X3 and X5 in memory orchestrator
  ([`58c5f1b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/58c5f1b6abef67399b523499d180b86f88fada8e))

- Resolve mypy errors in tools/comparison.py from CLI polish
  ([`117be5f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/117be5f8bad1f1741eaebab5aed03634175a36f9))

- Resolve pre-existing mypy errors in CLI and monitoring modules
  ([`7b82786`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7b827860dfbcda49a8692ea86a9af5b24c0088d9))

- Resolve ruff lint errors in adversarial tests and CLI init
  ([`27a9ece`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/27a9ecec4232bc7e2351883e8a0a64801526dc5a))

- Route entity-map mutations through CardStore methods
  ([`3a3bbbc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3a3bbbc30966b03232b8a84f0763b9076ddce421))

- Second Pydantic .get() crash + launch Phase B (PIDs recorded)
  ([`e915a57`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e915a57cd16ae6240e4b4af75de108a035b0e60b))

- Second Pydantic .get() crash in memory_write_example.py line 656
  ([`295430a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/295430a7b608fb35e5d99615e8ac1d9c7fb0875a))

- Sort imports in memory.py to pass ruff I001
  ([`3db6a7c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3db6a7c7d7015173e61bae75c4f2658d249133ca))

- Sort imports per ruff isort rules
  ([`6eb1c5d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6eb1c5d18fe7231bebe206eb3d43518e11afe406))

- Update ideas_tracker to use new origin_analysis.analyse() API
  ([`7f2fa6f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7f2fa6fea1933ab503ffb5ae69dd065aa1720de9))

- Wire memory_provider from Hydra config into EvolutionContext
  ([`26fc27d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/26fc27d92c2bc924d4e9037e4fc1d57fc484642c))

- **#28**: Wire emit_hof_rotate into CellStratifiedRedisOpponentArchiveProvider
  ([`37be785`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/37be7859b0bb3130023ff0f03aa6e201f4890fa3))

- **02**: Revise plans based on checker feedback
  ([`8cf1033`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8cf10333208e4b29cc192b09935654b41dac0f91))

- **02-01**: Switch watchdog to arms-race plot with actual_fitness metric
  ([`0589668`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0589668428165ede746d653358cb795502ef7a2a))

- **03-01**: Metric discovery propagation + manifest watchdog_plugin
  ([`bb1911b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bb1911b328399b0a81af186aff0932ec21f799cf))

- **03-02**: Metric formatting in status/checkpoint + CLI registry wiring
  ([`519299c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/519299c437b74c0e6ec4fd0e750dbe88a4ed3284))

- **05-03**: Replace missed tools/ import in top.py
  ([`e855735`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e8557357707b0c404e6da3d04f2d5e28d9722907))

- **adversarial**: Align opponent sampling with archive parent selection
  ([`bdeaa58`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bdeaa58ddd620b1197ac516fed721a462b42121c))

- **adversarial**: CompositionInjectionHook lineage labeling (I-17) + closeout
  ([`b2d8bdf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b2d8bdf9048fc5dab1cf43cc9d4ac2338b8ec09c))

- **adversarial**: Drift-cap redesign — KF-07 deadlock fix + asymmetric sync
  ([`4ce9f42`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4ce9f421edf0f4559717470d79995dca41e3f766))

- **adversarial**: Min_delta=10 deadlocks when max_mutations_per_generation=8
  ([`85ae086`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/85ae086c0b8bf3ab2fd7e394e09c296d2469ccb1))

- **adversarial**: Remove DGTrackerStage INFO log spam (I-15)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **adversarial**: Remove incremental programs_processed publication to fix sync hook divergence
  ([`e803d39`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e803d3961c9ce1cdca64a0ad7eca5ddb0bef57d5))

- **adversarial**: Split F22 guard and remove dead per_opponent_timeout config
  ([`a2e6513`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a2e65132a47577911732f3042309629c6c133ca1))

- **adversarial,tests**: Resolve mypy drift + remove stale tests for deleted manifest fields
  ([`737d4f8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/737d4f8b426e58425d18c553762b6baf1af44a0b))

- **adversarial/adversarial-vs-solo**: Patch watchdog — Telegram + fitness curves
  ([`8b2c709`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8b2c7091c6335402aca108837bd1635243a0a5cf))

- **aggregator**: Inject MetricsContext via Hydra ref resolver
  ([`f419a45`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f419a45ae42421611d703502836666fdc636fc66))

- **analyzers**: _split_id no longer crashes on malformed LLM sequence output
  ([`6d4eea3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d4eea3e360bf1c10758ab0acad531c092fe0d4e))

- **analyzers**: Guard against malformed LLM output in classify_against_bank + regression tests
  ([`3f19581`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3f195811d39beefe83f5f266b43054e8062f105c))

- **analyzers**: Narrow None-guard for mypy in _merge_cluster_results loop
  ([`d724840`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d724840b9a3787067239b7d9c6cb33021e77e77f))

- **archive-gate**: Read behavior_space/selector via island.config
  ([`1067203`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/106720361493bf290cbc2c6b4c120bbfc6ebd0ac))

- **autonomy**: Fix lint errors in telegram_notify + resource_manager
  ([`96d12c6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/96d12c671936af78372a9380d4b66c94e71ebbbb))

- **baseline-repro/watchdog**: Embed plots in PR + fix Telegram 400 error
  ([`f83bf7d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f83bf7d03557611ea18e013eb055957ce63d35f9))

- **check_docs_freshness**: Drop false-positive agents/skills checks
  ([`bfc90b1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bfc90b1e603b25a1a87f9991dff68c7d9c5dabbf))

- **CLAUDE.md**: Add --skip-agents-md to gitnexus re-index command
  ([`f66d51f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f66d51f750150c3722fb0f6706a9a0f3d895395c))

- **cli**: Add -f/--format to all subcommands, delete check_docs_freshness
  ([`2a101c3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2a101c3bc440afbd7319790b0879554c0f59214e))

- **cli**: Remove dead --source/--csv-path args and validation after run() API simplification
  ([`c0be3b5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c0be3b59b0806d98c8dc14fb503e2cfd1d4e866a))

- **cli**: Status shows raw float, never percentage
  ([`9349b42`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9349b422e5c47b0e6155ab60e873447c09a5ba14))

- **cli export**: Add positional labels, auto-fan-out for multi-run
  ([`9e4e863`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9e4e86311bc645aba66556d37d20ffea5428963c))

- **cli logs**: Positional labels, list mode, drop glob fallback
  ([`9cd5c6d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9cd5c6d2f288366fe6acdc179ad0f444579392cd))

- **cli/manifest**: Support bracket indexing in _traverse_raw (closes B2)
  ([`950c36c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/950c36c595be09687183f07b6fe7086618f497ca))

- **config**: Define redis.prefix as systemic default (I-12)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **config**: Make register_resolvers() idempotent
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **config**: Replace MainRunSyncHook with ProgressBasedSyncHook in adversarial_asymmetric
  ([`440670b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/440670b0d6c2d00fe1dae50ccdd8d2f6a947556b))

- **config/aggregator**: Revert @package _global_ on none.yaml — bare group form
  ([`532ac3e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/532ac3e5c372b7c3ca0ff000ea5d82b537dc53c6))

- **conftest**: Move import after register_resolvers() call — ruff E402
  ([`e675e49`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e675e49cb892e3edc6a0762a35534493a4c37650))

- **core**: Promote iteration to Program field, fix MetricsTracker crash & D>>G desync
  ([`9c11508`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9c11508d39caef3758e6d13df3b48e7af7c5929e))

- **D.4/D.5**: Restore preflight + launch_generator into gigaevo/experiment/
  ([`5f11751`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5f11751a43b0ee071234f55fe5a93cc265a499e6))

- **engine**: Annotate inlined parents var to satisfy mypy
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Await metrics_collector cancel before storage.close
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Bound post_step_hook to 300s — prevent ingestor wedge
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Close two orphan paths in _final_ingestion_sweep
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Eliminate ghost-persist by inlining single-mutant primitive
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Extend per-parent-id lock through child-DAG via ParentRefreshTicket
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Final ingestion sweep runs under cancellation
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Persist-then-mirror snapshot write — no version skip on retry
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Rename final-sweep loop var to satisfy mypy
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Rewire post_step_hook + adjacent observability polish
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Serialise _write_snapshot to keep Redis in sync with memory
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Two deadlock-class chaos-hacker findings + regressions
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Wall-clock bounded final sweep, patient on stragglers
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **entrypoint**: Thread Hydra pipeline knobs through PipelineBuilder
  ([`558b15f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/558b15f10049218a8d1ca573158b5deebd91a7db))

- **execution**: Annotate ParseMetricsStage Box tuple generic
  ([`552fd55`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/552fd55fd36ced5b5475b40c6f650530e9ceb5d8))

- **experiment**: Resolve ConfigSpec.extra vs .extras collision (I-00)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **experiment-checkpoint**: Derive frontier metric from problem.metric_name
  ([`cff0dd1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cff0dd173aa3e2786c6d2b8f5e026217fcd969e5))

- **F**: Update test imports — gigaevo.experiment.preflight, not tools.experiment.preflight_check
  ([`a665a03`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a665a039fd76c746aabe09c6fee1789ddbd5d6a1))

- **flow_profiler**: Remove redundant re-eval exec bar that caused zoom artifacts
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **flush**: --db now accepts space-separated args (--db 1 2 3 4)
  ([`d7ea6a8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d7ea6a8163fc52d5405050fe8a39a48575714f9a))

- **generate_launch**: Single-quote Hydra interpolation refs in extra_overrides (KF-02)
  ([`8d3deaa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8d3deaad10ac349543b9713fae0fceeb11bfe1ad))

- **heilbron**: Add population_role, post_step_hook, watchdog dual plots
  ([`bc50d2b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bc50d2b57cc0904d65647f5465ebb161d3a47efe))

- **heilbron**: Switch to evolution=steady_state for all 8 runs
  ([`eff3ddf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/eff3ddfe3dd7b57248cf9601973356e7bf939284))

- **heilbron-adversarial**: Redesign bundle validated by sandbox
  ([`8381368`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/838136886bc9ca2caaf066148395c584dcd3de8b))

- **heilbron/adversarial-repro-v1**: Return (metrics, artifact) from evaluate
  ([`4e6756a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4e6756a86778f3e5a073857bcf5f3020723cf2ea))

- **heilbron/adversarial-v2**: Add evolution=steady_state to all runs
  ([`3ee9766`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3ee97665f30aed7825bd6c5c058c03a130b5df45))

- **heilbron/baseline-repro**: Watchdog — ensure internal hosts bypass proxy in NO_PROXY
  ([`7ce36bb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7ce36bb5fb4ac969bbbe5f04962b95c48cb08a45))

- **heilbron/d-tanh-no-lineage**: Convert treatment_checks to list-of-dicts + add per-run pinned
  blocks
  ([`1782103`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1782103fcc7977ca1218648a4714174d0e1a55d0))

- **heilbron/k5-budget-v3**: C1 — CellStratifiedRedisOpponentArchiveProvider reads production
  archive schema
  ([`c195a26`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c195a26a604317f9b352b67e4b3b8b68d2462904))

- **heilbron/k5-budget-v3**: Critical alignment gaps — BD configs, coverage stages wiring, test
  metrics
  ([`ddcfac9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ddcfac9e0b679ca75d518234a99d915266ab052c))

- **heilbron/pop_a**: Smooth resistance scoring (zero-sum with pop_b)
  ([`2de8267`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2de8267e4cf29be90ae20e163d9e00f1ac1e97c9))

- **heilbron/pop_a**: Terse metric descriptions (not duplicating task_description)
  ([`4273ba9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4273ba9394e62022c3576a4cccba089aba6608a3))

- **heilbron_adversarial**: Correct metrics.yaml descriptions for soft/binary variants
  ([`10dacfc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/10dacfc8d674080d6295687d53cf91bb652d227b))

- **heilbron_adversarial**: Replace absolute-path symlinks with relative
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **heilbron_adversarial/pop_a**: Correct binary resistance formula in task description
  ([`99b21ce`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/99b21ce1a1ca29e8716a22ab61317ca1dbb14be8))

- **I-18**: Program.create_child propagates iteration from parent
  ([`051a2ea`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/051a2ea9bb8b82f524b8dd433e0c8c310337966d))

- **ideas-tracker**: Normalize_improvement_item — whitespace string → Unspecified change
  ([`bf893a4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bf893a43f7561f8360c46153f92cbf53bed898e1))

- **ideas-tracker**: Remove unused _statistics import, cast pandas row values to str
  ([`faebc9a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/faebc9af09c8e5e3819dc70ccb823a69b1008801))

- **k5-budget-v3**: Revert ProgramStageResult.skipped for insufficient shared benchmark
  ([`02c6f56`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/02c6f56aae4cbb3fb9a076cae92fe3e8cebef712))

- **launch**: Allow --dry-run from {preregistered,implemented,running}, skip DB claim (I-02, I-05)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **launch_generator**: Add shell_escape kwarg to _build_run_cmd (I-03, I-10)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **launch_generator**: Pass through all config.extra keys as Hydra overrides
  ([`a5957c9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5957c92260373e439da564b0d2ae9c1e52c4111))

- **launch_generator**: Use config.extra.get() instead of config.get()
  ([`4fe6f29`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4fe6f29f2783f12edbe347a8668df350dcc792c1))

- **lib+config**: Enable launch_preview for experiments with ${...} overrides
  ([`1cd0df0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1cd0df0711d3baa77931be68b8b4cc85ae94da83))

- **lint**: Remove unused imports and fix import ordering
  ([`f1a41fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f1a41fe1e9f6996d052469435c1f890946d06ba3))

- **lint**: Ruff format manifest.py
  ([`4022d40`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4022d407b9568c03acf0d4dea538b601d5e02620))

- **lint**: Ruff format preflight_check.py
  ([`f570158`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f570158fb580f5dc76882806157bee2400a9e175))

- **lint**: Sort imports in watchdog_cmd.py
  ([`62cb35e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/62cb35e6101231ab94633676330184ae29d5c2b8))

- **lint**: Use collections.abc.Mapping (UP035)
  ([`f4c4f2a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f4c4f2a94edbf2b3c5f0ed74a7cd383880fceacb))

- **llm**: Fail fast on unreachable LLM endpoint in MultiModelRouter
  ([`d490f4a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d490f4a5c49647224dd08a8ed5eacfb77904fec7))

- **llm**: Langfuse v4 handler init; pin langfuse>=4,<5
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **llm**: Use model-configured API key in reachability probe
  ([`6d24110`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d24110eccb765e24d18dea1ecce783f76e1e710))

- **llm**: Wire real tokens into LLM_CALL, raise on silent None parse, harden Gemini 3 Pro
  ([`344c1ce`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/344c1ce9ff249901c5260fe5439c386e342a0b82))

- **manifest**: Accept null treatment_verification.note (C11)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **manifest**: Add RunSpec.pinned, ConfigSpec.pinned, LaunchInfo.config_fingerprint (I-01, I-09,
  I-11) ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **manifest**: Allow null treatment_verification.note + add INFRA_ELI5
  ([`cd81beb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cd81bebe61764d6339273c6f14deb5cdddfbf0f2))

- **manifest**: Restore AlertThresholds.excluded_events (regression from 97f823f9)
  ([`965045f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/965045f7bbb6eb9cf2cf3ac1dbeafe5b28cbb0ee))

- **manifest/preflight**: Refactor magic dicts to typed Pydantic models, fix B1/B5/B6 bugs
  ([`5b1b829`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5b1b829977d310d484054cd45a5c19281a84a658))

- **memory**: _run ordering + extract parse_string_list helper
  ([`47d7627`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/47d7627c087a430d7390496d49a70c0876429e9a))

- **memory**: Add error handling and diagnostics for production bugs
  ([`36ba396`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/36ba396c75305f3e8afc8940f108163c6fa3dc44))

- **memory**: Add logging to silent exception handlers
  ([`de554b8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/de554b8a5961fb9c443d9770f17e17d1418b9cd3))

- **memory**: BM25 import, max_tokens, dead code, duplicate normalize_memory_card
  ([`5fd6022`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5fd6022ff576e9b1ecf7573d6c3e5aae177cdbfe))

- **memory**: Card_loader streaming, card_update_dedup JSON logging
  ([`2b26716`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2b2671628bcaee83f27a40e0fac23e93e8c9e080))

- **memory**: Card_loader streaming, card_update_dedup JSON logging + E2E tests
  ([`c9bc7c7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c9bc7c74e24989411602d62a701049d94908758c))

- **memory**: Card_store thread-safety + remove incorrect note_ids in _load
  ([`78c0c77`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/78c0c777f70e01b330f71e2fd5cc387fe35bcf05))

- **memory**: Consolidate Pydantic config, standardize exception naming, fix resource leak
  ([`134958e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/134958ea3d4dfc796743315d2258c55f39b1f2d3))

- **memory**: Correct AmemGamMemory instantiation in write_pipeline.main()
  ([`05fbe87`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/05fbe878c44a759307a3f9f6b7a47f954d9ed1a4))

- **memory**: Correct get_card type contract and isinstance bug in write_pipeline
  ([`e26e496`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e26e496ab2a95ebae72c78fa38594a9116a02b31))

- **memory**: Correct is_valid filter — absent means pre-filtered, not invalid
  ([`0361752`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/03617521d92692320ef8dc75074cfe56e5d46a0a))

- **memory**: Correct Pydantic card access in memory_usage_example.py
  ([`a492313`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a492313fcb9b7dcb2284cc4cfb2c688c01a62cb5))

- **memory**: Defensive copy in CardLoader, fix base class LSP, tighten utils exception
  ([`a2e3d1d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a2e3d1d20729f4287623883485240cedf7be01b2))

- **memory**: Filter invalid programs + skip ghost idea cards in write_pipeline
  ([`934eadf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/934eadf7065c0af2429cedbc751aeda047027fe9))

- **memory**: Fix __exit__ traceback type annotation in AmemGamMemory
  ([`9050b58`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9050b583cb924655a4eebdf6ef1dec4a0a9dbab2))

- **memory**: Fix _SessionLog.flush() JSON crash + add regression tests
  ([`f70d329`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f70d329c8fa92f188701fd1b8dc42e784734fa70))

- **memory**: Fix import path and exception type bugs
  ([`a972afd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a972afdb2a6fa8dcd689fc9539c0dc26e77718d1))

- **memory**: Fix import sources and use model_validate in write_pipeline
  ([`366c4c9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/366c4c932756757c3015410940ceaf6a41047c89))

- **memory**: Fix type annotation on apply_usage_updates + lint
  ([`26425de`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/26425de4f099e212afd33c50342c39e8e1980f5b))

- **memory**: Fix type mismatches in fakes, add LLMClient resource cleanup
  ([`9e00e7f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9e00e7f538d926559fa6b44b11bc03781bb23e4f))

- **memory**: Fix usage payload type cascade in idea_bank.py
  ([`49e0ebb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/49e0ebb9c096148dedd019c97e683060395220ff))

- **memory**: Log exceptions instead of silently swallowing
  ([`33c2ca8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/33c2ca809c0700e2d8588e333a6b6ec2d4127d0f))

- **memory**: Log graceful-degradation exceptions in conversion/tracker
  ([`33860be`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/33860be78ba0b39714b25d337f3cd436ecfbb835))

- **memory**: Loguru lazy format, asyncio deprecation, silent errors, tighten exception scope
  ([`397011b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/397011b897ced3e9ea8984f11e9f1602285d51c7))

- **memory**: Mypy type error on _json_safe_dict
  ([#209](https://github.com/KhrulkovV/gigaevo-core-internal/pull/209),
  [`00122c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/00122c0c62e127590d6942d5cd704561ebc8be27))

- **memory**: Narrow exception handlers + fix loguru f-strings
  ([`ba2d14b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ba2d14be1bd5d590bc05573db4147e035d8d280d))

- **memory**: Port IdeaTracker analyzer factory to unbreak ideas_tracker=default
  ([#209](https://github.com/KhrulkovV/gigaevo-core-internal/pull/209),
  [`00122c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/00122c0c62e127590d6942d5cd704561ebc8be27))

- **memory**: Re-apply max_tokens=None, remove duplicate normalize_memory_card, remove load_dotenv
  ([`fd42be2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fd42be2ca13f271b31e3824f1cbc305ea2444f85))

- **memory**: Remove module-level load_dotenv() + dead assert
  ([`9b9e619`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9b9e61990fadd62157a24ef9aed7f5b6797dfd85))

- **memory**: Resolve import ordering and model validation issues
  ([`ada400b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ada400bb69b60d4b1538bd3eb373f5cf34167eba))

- **memory**: Serialize UsagePayload correctly in cards and write pipeline
  ([`3e43b12`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3e43b12ca5376b439bfe9e7af2d959d7359a5cee))

- **memory**: Standardize loguru call style and exception naming in ideas_tracker
  ([`f1e44c4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f1e44c4ad4fe31fd683f06af02a0aae23a77d1f2))

- **memory**: Strict ConnectedIdea validation and field name consistency
  ([`ba63b63`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ba63b6364cc16bd0729ceefa54781cf87fa53276))

- **memory**: Type annotations + GigaEvoMemoryBase ABC signature
  ([`d514efe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d514efe884b71a329ab8d545814b468531f1d572))

- **memory**: Update import build_dedup_retrievers → build_retrievers in card_dedup.py
  ([`9e02cab`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9e02cabd6665710ca3c2b746fc0672d73eb408df))

- **memory**: Usage_updates_path override — use config when caller omits it
  ([`5f80b91`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5f80b91fda80b9a630afee2cef4848d6cfae2d2b))

- **memory**: Write_pipeline exception scope + patch_gam_imports module path
  ([`d6a7f70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6a7f7056447f3142f3f3f71fee7f820f76640ee))

- **monitoring**: Rebuild httpx client when event loop changes
  ([`a06732d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a06732ded64188f157bc9600ba66b8934fbb1b59))

- **mypy**: Fix three pre-push type errors + update .gitignore
  ([`455e8b5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/455e8b539c17de2f0eedc525d17d01a58fabdff1))

- **mypy**: Narrow aggregator type + type ReduceSpec ops + LinearSpec terms
  ([`57d838a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/57d838ab0491d548028a83425139d5ac3ce57fef))

- **mypy**: Remove stale WatchdogPluginOptions reference from adversarial plugin
  ([`07f60a4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/07f60a49290e37499920c16d4195c76bb9d23a4e))

- **origin-analysis**: Type desc_cache as dict[str, DescMetrics] to satisfy mypy
  ([`e9aff8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e9aff8aaf8a860c2d42b03f16254e358fe6dc5b4))

- **plots**: Aggregate per iteration before smoothing and plotting
  ([`ae15fce`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ae15fce8ec69a1ebdd704faf418cfbe844477990))

- **plots**: Use program.iteration column and wire sentinel filtering through CLI
  ([`3e2b7f1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3e2b7f11f1154568eb4b45010f6b01e58b0b7676))

- **post-experiment**: Apply systemic fixes from heilbron/asymmetric-iterations
  ([`826665e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/826665e884bab0d42c0750ee90bf0405323b93e9))

- **prereg**: Record PR #207 in experiment.yaml
  ([`5e90c42`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5e90c4203f8abb79859459d41526b4412c66e77b))

- **prereg**: Set prereg_commit hash in experiment.yaml
  ([`8a856d9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8a856d9cf81d1fdd24bbb4b9e49f5b681ef0ca8d))

- **processes**: Sanitize bash variable names, coerce config types, broaden run detection
  ([`8a5e301`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8a5e301f9b1f0200778f468bdd557aadc2798e3d))

- **profiler**: Drop experiment-branded subtitle from page header
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **profiler**: Inline Plotly so HTML renders in sandboxed previews
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **refresh**: Bound _locks dict via WeakValueDictionary
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **run.py**: Drop stale cfg.max_generations reference
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **sandbox**: Per-arm hydra.run.dir to prevent loguru file collision
  ([`f3ee779`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f3ee7792f83797443530387264c8f1bc3e0c41df))

- **sandbox launch**: Cd to project root + fix archive_reeval override path
  ([`b70ea73`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b70ea73e0a83b00d892d1d4a1f1ba25fcbdae7d4))

- **sandbox launch**: Use full redis-cli path
  ([`75bc30e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/75bc30ed9fd17aff60ed1a319f2ca7979a920f84))

- **skill**: Experiment-checkpoint reads metric_name from experiment.yaml
  ([`dd4bf62`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dd4bf621b2c8ef3ff87fbbd883d9cbce2a8f2f89))

- **telegram**: Route through HTTPS_PROXY for servers without direct internet
  ([`832b922`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/832b922aed5f09d638cce94013f6828a24374324))

- **test**: Move regression test to module level (was inside class)
  ([`6c00b81`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6c00b8185cdafb54844fdbc64753d7183a4c1cf4))

- **tests**: Add pythonpath=. and --import-mode=importlib to pytest.ini for namespace package
  imports
  ([`3159437`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/315943743327f32d404043a9b95b0e66dc4f467f))

- **tests**: Fix pre-existing test failures in integration suite
  ([`d43750d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d43750d6d61d71f936988bd0d65d521454bbb7c7))

- **tests**: Get main green — purge brittle/drifted test failures (#234 slice 1)
  ([#237](https://github.com/KhrulkovV/gigaevo-core-internal/pull/237),
  [`ea3fbd3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ea3fbd3692b5446e1166c6fd6555f139519e9baa))

- **tests**: Repair five pre-existing failures on main
  ([`41ce1d9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/41ce1d925c99eeb32b6fb6c9e6244178271e08a6))

- **tests**: Resolve import ordering in trajectory tests
  ([`a65a0d8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a65a0d8a69966d2373d11ea33a013f5c9bd03c1f))

- **tests**: Ruff import ordering in golden-vector tests
  ([`f47cb48`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f47cb483c027b6beda8b20649acc6896b431d524))

- **tests**: Update method names after GamSearch rename
  ([`7ba6411`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7ba64111413d680d93cc22d6df932ccb2e0639df))

- **tests/cli**: Migrate fixtures to engine:snapshot
  ([`bf9741c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bf9741c0a03e122479dccfa81bfe4f6372c1b4a8))

- **tools**: Detect live writers in flush+preflight, support factorial designs
  ([`f4c0f49`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f4c0f49fe4bec4c7ffdb9af6c55445f7f21f490b))

- **tools/litellm.sh**: Use rpm instead of max_parallel_requests for chains
  ([#217](https://github.com/KhrulkovV/gigaevo-core-internal/pull/217),
  [`7670f7a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7670f7a812e36bbe6cca2bd8c4a6e197c5d4dc8f))

- **tracker,lineage**: PR #215 review — unify metrics schema + restore BD-axis
  ([`9af6e6e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9af6e6ecc87be5f9636efe230b38c9cb772ca172))

- **types**: Annotate channels list as NotificationChannel for mypy
  ([`1b98495`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1b984958e74df9e321fc89023daa4ba84e91ab1a))

- **types**: Annotate indices as set[int] in OpponentProvider
  ([`b4ea873`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b4ea873e1c8cd8c0c5322485a5406449aeeb8ffd))

- **types**: Assert isinstance(IdeaAnalyzer) in _process_program to narrow analyzer type
  ([`941020d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/941020da495d36b0d1385aee96686b7f20d4e4c8))

- **types**: Child/parent in lineage are already ID strings, not Program objects
  ([`4a82148`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4a8214874ba609c30cd3073d3bec030bc5a22200))

- **types**: Remove stale config_path/path_to_database kwargs from IdeaTracker usage in CLI
  ([`fe3a7e7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fe3a7e7bec794697c47b6383201e59af9faef658))

- **watchdog**: Annotate channels list for mypy variance
  ([`503cd6a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/503cd6abdcf86226b51186e7d69bb945779dbc2b))

- **watchdog**: Correct per-gen metric deduplication + 2x2 redesign
  ([`b2c6c01`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b2c6c015871a3470467e9be931559e406ed6d605))

- **watchdog**: Load .env on startup so Telegram works without pre-sourcing
  ([`a752c96`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a752c968a5c0a1c268ac9c7d69e9f7a4d9ba58c7))

- **watchdog**: Load_dotenv before checking TELEGRAM_BOT_TOKEN
  ([`3db942f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3db942fd7da13500a23c355de1b153a4747d7bbe))

- **watchdog**: Use distinct upload paths for arms-race vs comparison plots
  ([`d203dee`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d203dee0d250f4925792e71aabcc2e0cfcbc4a5f))

- **watchdog_cmd**: Wire TelegramChannel into dispatcher from .env
  ([`3f5bf73`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3f5bf73fb66aa0bef58f861867d5c5a830df52fb))

### Chores

- Add .worktrees to .gitignore
  ([`e87188f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e87188fbb4aef0df35714f5e8518951151727c9e))

- Add gigaevo-memory as proper dependency, remove sys.path hacks
  ([`db61ead`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/db61eadb3eefaa0cec7679a921ad841ba426c470))

- Add public release sync script + pin gigaevo-memory
  ([`e314e1e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e314e1e33dd41fe4f9bb2f08eb96bc7027f6931e))

- Add pyrightconfig.json pointing to evo conda env
  ([`4f86a6d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4f86a6d7ede56d791d3daeb6d823059f301ef02a))

- Add shared make_memory fixture to tests/memory/conftest.py
  ([`9be054a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9be054ae8ee21711b6beadeb1a0494c4ab894df5))

- Bump GitNexus index stats (33034 symbols, 83096 relationships)
  ([`6f3e3bb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6f3e3bb22313121b423fc39c6761c47629299ea0))

- Bump GitNexus index stats post-reindex (33069 symbols, 81965 relationships)
  ([`c298f51`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c298f51f76cd8b77140f34b8cff8ef7cf6b7da34))

- Commit session artifacts — hookify rules, gitnexus skills, superpowers plan, top programs, docs
  tool
  ([`ba78e13`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ba78e136039e48d97f17481ce789d016304180eb))

- Commit session artifacts — skill evals, Hiroshi grading, protocol review
  ([`607c421`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/607c4214c3bada64a4f6b0c0c9d3066b3deb4762))

- Complete v1.0 milestone — monitoring & tools overhaul shipped
  ([`9380f39`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9380f39b5d6657ccc4301181905dc89a1fca3fa7))

- Fix lint errors in ideas_tracker (unused imports, sort order)
  ([`c20bcea`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c20bcea1fa3c41a5fd0065ff4f209969775b07d0))

- Fix lint in ideas_tracker/analyzers.py
  ([`76e84cf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/76e84cfb90a02a4a47be864019ccdb731a88df7f))

- Format ideas_tracker/llm.py
  ([`d1d4e60`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d1d4e606105d2b211a7add1d55b30b6f3de2f491))

- Gitignore skill-generated artifacts and hookify local configs
  ([`09e6d70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/09e6d70e4e7b3ce3f06dabe3c9d746fd00d11139))

- Ignore .claude/worktrees/ directory
  ([`c6423f6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c6423f64765a86e5fb0394b03ecf02dfb0f33ce6))

- Move inline imports to module-level in test_ideas_tracker_pipeline.py
  ([`cbd8af7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cbd8af771052ad240dd0c8ac7745689fd54a9041))

- Purge stale files and rewrite .gitignore
  ([`081d477`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/081d47711e0e0b97d11fbdd6f28044557dd7237d))

- Record PR #169 in experiment.yaml and 03_plan.md
  ([`b414eeb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b414eeb490ec7a3718191736720219f855576616))

- Record PR #183 and prereg commit in experiment.yaml
  ([`24c3353`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/24c3353d558726f265d6897328065e4890e66f32))

- Record PR #188 and prereg commit in experiment manifest
  ([`91d9e05`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91d9e0553a4a902d76b25b78b27e7db81fed5e11))

- Record PR #206 in experiment.yaml
  ([`64ef394`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/64ef3946c63fd294b101508823f7b50ef3ee41a0))

- Record prereg_commit hash in experiment.yaml and 03_plan.md
  ([`6afac4f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6afac4fa6460c1d80a44fc3083ca23c087f9f9de))

- Record prereg_commit in experiment.yaml
  ([`ef91860`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ef918601c0efd8edd693df8beaf9aa1a15ab066a))

- Record smoke test completion, set status to implemented
  ([`ff4f972`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ff4f97257d1e7b75847fb2d051b9ef95e96f2939))

- Remove stale benchmark_survey.md
  ([#221](https://github.com/KhrulkovV/gigaevo-core-internal/pull/221),
  [`25470e3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/25470e378762eded8bffcfa5fbe887239345b8bc))

- Remove tools/no_proxy.py — reads infrastructure.yaml which is not in the repo
  ([`5899fbb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5899fbb4e8cafb782c245a324f3615004ed8a6a2))

- Remove unused DedupDecision import from memory.py
  ([`e478259`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e478259869ee4172eb173eba6c557c1f25f76596))

- Ruff format follow-up on test_mutation_agent
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- Ruff formatting cleanup
  ([`55faae0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/55faae0435fc196ebf13b0c45b888c5a3a3a14f5))

- Stale-code sweep for steady-state engine PR
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- Strengthen GitNexus pre-flight gates + add memory refactor plans
  ([`aaaee98`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/aaaee98efd73b5b3b4aebc1fbdd5bfabb6239225))

- Update agent/skill infrastructure, CLAUDE.md, and add refactor plan+spec
  ([`2523061`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2523061813a444fa527a72d993152d579fbc28dd))

- Update GitNexus index counts in CLAUDE.md
  ([`6e940d8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6e940d8837b144b31f4f2609942a53df3280c4d8))

- Update INDEX.md with heilbron/adversarial-v2
  ([`e33b1c4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e33b1c4c4ba50d0fcc18b761ea9766481bd00480))

- Update stale test comments to match refactored API names
  ([`b73d108`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b73d108cee5a54afd6698a15e2deabe77d80a56b))

- **01-02**: Delete project-pm skill/agent, remove pm_audit references
  ([`c41f86a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c41f86a1ac4ab286c3f0fd05af5427ae2d58cc63))

- **deps**: Unpin gigaevo-memory from private git URL — it's now public
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **exp/heilbron/adversarial-repro-v1**: Consolidate long-standing branch state
  ([`6c33fb8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6c33fb889e69852659baddd0a94d88b79b66a3e1))

- **heilbron/adversarial-repro-v1**: Add missing environment_freeze.txt at closeout
  ([`ee02ef2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ee02ef2bc71e162e8474f0974c0889f05d2fa3ae))

- **heilbron/d-smoothing-minimal**: Record pr_number=222 + prereg_commit
  ([`b95614f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b95614fcccbe7fa498d438e6240bbca5443b4895))

- **heilbron/d-tanh-no-lineage**: Launch artifacts + status=implemented
  ([`8aae098`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8aae0981fea62f5b949084b6dd660a1986735d3f))

- **heilbron/d-tanh-no-lineage**: Record pr_number=223
  ([`d393975`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d393975a9a7ebce47336aed5aa81bc90a3878a28))

- **heilbron/d-tanh-no-lineage**: Record prereg_commit=7b7c5f5e
  ([`953ad6a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/953ad6a87a88b61e53fe416ea5b9cb6ccc358e84))

- **heilbron/v1-honest-repro**: Post-merge closeout — KF-14/15/16 + open question + paper-draft
  deferral note
  ([`b7c6ce9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b7c6ce962c1828b5383aa2751a7ddcdef51b0f98))

- **manifest**: Fix lint after polish series
  ([`b29529e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b29529e39e40794d78d898af476e12e658a0692d))

- **manifest**: Fix mypy + missed CLI callsite from chunk 10
  ([`c9b9764`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c9b97641001df3dc3cd1b42354779ad7227f3db2))

- **memory**: Delete dead code CardIndexStore (card_index_store.py)
  ([`7d57833`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d57833dfec1dd8cdb663daebadeee0f61368840))

- **memory**: Fix import ordering in test files (ruff I001)
  ([`a185b1b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a185b1b9488ac3fab75047e212449fcf09f20b64))

- **mypy**: Exclude gigaevo/memory/examples/ from mypy (demo scripts)
  ([`803f451`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/803f45196b5e8abd151b5a90c92d8334a23d4427))

- **steady-state**: Drop legacy _in_flight_sema hasattr guardrail
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **tests**: Move inline imports to module top-level
  ([`396e5d3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/396e5d325dfc98a0cf8cd411aa1a2593b0e5d6a6))

- **tests**: Move inline imports to module top-level across 6 memory test files
  ([`24c3d90`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/24c3d9046f6b975dc1edd371a06b848b43d308b2))

- **tests**: Move inline imports to module top-level in test_dag_memory_flow.py
  ([`239bee9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/239bee97268c4d96ec0a5e8f0029a655b8db05c1))

- **tests**: Move inline imports to module top-level in test_ideas_tracker_pipeline
  ([`d208146`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d208146d904fa7b4e4ff37e60bcb137859df081a))

- **tests**: Remove broken fixture-driven watchdog integration tests
  ([`e3be648`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e3be6481e24a02b776f80f4ca35626ddfbfb6026))

- **tools,report**: Litellm token logger + routing rationale + report refresh
  ([`7fa58e9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7fa58e9f319527ed5cb1042ffe1a32c20353a10a))

### Code Style

- Apply ruff format to memory test files
  ([`20eeca9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/20eeca97eab79fbb54d742cbc0240b0eaec4f495))

- Auto-format 4 files per ruff format
  ([`74373e6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/74373e6c13ce9cb2297d3501fd25eab5ae279434))

- Fix import ordering after no_proxy removal
  ([`13a114f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/13a114f9d22408fc812fd54208ce764db339d9e9))

- Fix import ordering in analyzers.py (ruff I001)
  ([`d5fbeb8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d5fbeb86f264cfc893193263d0fc96545107149d))

- Fix import ordering in E2E test file (ruff I001)
  ([`bc56405`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bc56405d23f747aa70844f30ff822ed068c5637b))

- Fix import ordering in ideas_tracker.py (ruff I001)
  ([`ae70ed5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ae70ed5a6aebb9e7113ce39893075d229e516d86))

- Fix ruff import sorting and unused import warnings
  ([`5201bf1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5201bf1935101bf67156d1bc9559129be70bae45))

- Format api_sync.py
  ([`d0c40ee`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d0c40ee803853d1f9d425555f62bfc49d593acac))

- Format manifest_cmd files for ruff compliance
  ([`25e8164`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/25e816407f4677d5151600ec19478d413d2c52fd))

- Format run_watchdog.py
  ([`76c35d3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/76c35d3dae61d39de6247b6f66a941aefd7b21f7))

- Format run_watchdog.py and telegram_notify.py
  ([`3e0ee5f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3e0ee5f81cb7b2945acf459d2f789d5575889f96))

- Format run_watchdog.py for ruff compliance
  ([`db42b8f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/db42b8f7af5c9eff18f8c5f3cf1412ee675d8159))

- Format test_evolution_engine_complex.py
  ([`581e00a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/581e00ad4a65bec075daa6e8113ea32efef23d40))

- Format test_progress_sync.py
  ([`9b3cf37`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9b3cf37653383d23e6f18abb959116a97a8ad214))

- Ruff format all files flagged by pre-push hook
  ([`d04092b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d04092b225096e4555d0bd3519ec86b8e53ab832))

- Ruff format asymmetric_pipeline.py after cache_on flag commit
  ([`8666f5c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8666f5c31ccde14a523e5059644ee5ff39eaaa4a))

- Ruff format fixes
  ([`bca0660`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bca066053fcd3f37ac2dde2ca572a12a3c43dc1f))

- Ruff format fixes
  ([`c53527d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c53527dabb816e15f826e701de2e715499247962))

- Ruff format fixes after merge from main
  ([`147cfd5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/147cfd52566c07255aaeb9078cd99c3937b88d95))

- Ruff format run_watchdog.py
  ([`a44ebb2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a44ebb29056c447b24832e5fc229931c88891d46))

- Ruff format telegram_notify + resource_manager
  ([`a5c243c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a5c243c2dc0cfacf35ab35395278b541c58dc10c))

- Ruff format test_config.py
  ([`7692e71`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7692e71fa2fe1d5cb061dd694107ed09bb6f8c67))

- Ruff format test_refactor_bug_fixes.py
  ([`e301cf9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e301cf9ee1fe40c341ce9625fe610ab1122f1ce1))

- Ruff format watchdog files
  ([`535c19c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/535c19c50468b9d6dd00e26122cbbfc12642400d))

- **memory**: Ruff format ([#209](https://github.com/KhrulkovV/gigaevo-core-internal/pull/209),
  [`00122c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/00122c0c62e127590d6942d5cd704561ebc8be27))

- **run**: Ruff format wrap on Redis logger.info call
  ([`0bce364`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0bce364212fd5b8f5fbd3549afe0b328048f474f))

- **tests**: Ruff format fixes for CLI tests
  ([`493fe09`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/493fe09160f8be69fa5e3c1d48fe0d1f2326cbf7))

### Continuous Integration

- Bump job timeout 10→30 min and per-test timeout 60→120s
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

### Documentation

- Add autonomy-stack operational guide
  ([`fb8571a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fb8571a48cbf3e91722a2126e857709afb871bf5))

- Add comprehensive MEMORY_ARCHITECTURE.md + fix import ordering
  ([`d28ab8d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d28ab8dcd7b31fda1c6bf025c2072d7ca0abd1cf))

- Add issues log and Phase B auto-launch monitor script
  ([`d6b6064`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6b60645441bf184922d05ce486ed625eda5b340))

- Add manifest module organization guide + CLI documentation
  ([`9b320ca`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9b320caf98c5ec2d139dacb7dafa9e5ab2f9d3fd))

- Add monitoring scripts and procedures for Phase B
  ([`28e4892`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/28e48920ed38be43bbaf85fd47a28e776f736741))

- Add MutationAgent TDD sprint design spec
  ([`7da4eae`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7da4eae37abd3864f5ec91e442ad0793e4c19027))

- Add origin_analysis refactor design spec
  ([`c97a352`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c97a35245c5aeb2e5f0a1f6f76d74b1f7ed0a039))

- Enhance deprecation note for inject_fakes_into_memory
  ([`8b9cb71`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8b9cb71b3dd7d9529a5b0d83f0604ed3772a1cd2))

- Fill PR #204 in experiment.yaml and 03_plan.md
  ([`1bcd2ed`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1bcd2edf89633d38a8b6a6e6ff865f433e517219))

- Fill prereg_commit hash (8ce7ab3f) in experiment files
  ([`572a7b4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/572a7b4fb46e7138048976f948e5b3c18d5d754a))

- Fill prereg_commit hash (c9d04534) in experiment.yaml
  ([`ead6504`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ead6504a210f1d8184b0c22f7d2c9c72d4342dcc))

- Fix stale server counts and missing experiment presets
  ([`6c90571`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6c90571f428ea877f0d2a83d08b3e132c71f3bd8))

- Improve memory README with clear flow diagrams for new users
  ([`c27c34c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c27c34c35f9c34a8714d8a1e5392fd666d356cb8))

- Map existing codebase
  ([`9e8f3b5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9e8f3b5aefc1382062f4fa34e1f67fdafbaf2ef2))

- MutationAgent TDD sprint implementation plan
  ([`763b4f2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/763b4f2726c3d992ee9066343a938959ac63b97d))

- Record PR #197 in experiment.yaml and 03_plan.md
  ([`2581173`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/25811732b4b02361b772bfe7d5fec8fd5643ef7d))

- Record prereg_commit ebc12cc3 in experiment.yaml and 03_plan.md
  ([`9245278`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/92452782aee31ef976e5e51dceef16970a8a7218))

- Replace all legacy PYTHONPATH tool invocations with gigaevo CLI
  ([`418af4d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/418af4d2efcb79f64a01794519f7f16e0909294b))

- Sweep top-level docs for May release (#233 slice 1)
  ([#241](https://github.com/KhrulkovV/gigaevo-core-internal/pull/241),
  [`8fa118f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8fa118ff300dd76e34cf13673b5b556e8dec54bc))

- Update CONTEXT.md + PATTERNS.md from heilbron/adversarial-v2
  ([`6142b35`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6142b350290eac97e8b800f05c2fbd74f2a71585))

- Update MEMORY_ARCHITECTURE.md with final line counts
  ([`a8b2778`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a8b277877b7e08438d69f173af3ad0b9c60b3d61))

- **01**: Capture phase context for CLI tooling update
  ([`1a04431`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1a04431e03a1f8cefb9ae254dd6cff0188102944))

- **01**: Create phase plan for CLI tooling migration
  ([`abb0a5f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/abb0a5f1d572ab670faf124d53ead2121900e0ec))

- **01-01**: Add self-check to SUMMARY.md
  ([`cafde86`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cafde86fc608ba3ee39e168be91f10090e4acc80))

- **01-01**: Complete manifest CLI subcommand plan
  ([`736bb86`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/736bb869b013eafe38f16aff1b19c28c0c980fed))

- **01-02**: Complete skill/agent CLI migration plan
  ([`e815f3b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e815f3be93c9e2fba4fc638f204da45c81201674))

- **01-03**: Complete heavy skill migration plan - SUMMARY, STATE, ROADMAP
  ([`6bbaae4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6bbaae43dbf5bd68e16098de26d42f4e3785b965))

- **02**: Create phase plan — 3 plans for adversarial injection + watchdog fixes
  ([`a9458c2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a9458c2a0d7d0bfbdaff1b3d60472935ac253f50))

- **02-01**: Complete sentinel filtering and watchdog plot fixes
  ([`1fea168`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1fea16868221d6ce0327b66ba8a0d1205d94ace8))

- **02-02**: Complete composition injection and post_step_hook plan - SUMMARY
  ([`a2614bb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a2614bb19701f8e0e4c4a65bbda8a7b02e5169db))

- **02-03**: Complete D-G improvement tracking and per-program D selection - SUMMARY
  ([`7ccdcc6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7ccdcc6df0f032b246f28b8a469edbd9c0b7d590))

- **03**: Create phase plan — fix CLI metrics reporting and manifest wiring
  ([`ab4f01a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ab4f01a0eb3a485062b0083a80674f5f1b801f1b))

- **04-01**: Complete foundation knowledge stores plan
  ([`ded02eb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ded02ebf0b784e920d5a8092fd7fb0bdb4584018))

- **04-02**: Complete GSD plan wiring into experiment skills
  ([`5e62a17`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5e62a1781aedd83b33f44c7de6992745c531ee27))

- **04-03**: Complete event auto-capture in lifecycle skills plan
  ([`3144ce7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3144ce73442c2c459bbbc55eada0699e191a045e))

- **04-04**: Complete pattern promotion and fix tracking plan
  ([`27fe544`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/27fe54407251b6093ec70f97413294b64c1d0a0f))

- **05**: Capture phase context — CLI/watchdog/manifest polish
  ([`c6fc3cb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c6fc3cbdcf9c11a83c847578759fddd759ed865e))

- **05**: Create phase 5 plans — CLI/watchdog/manifest polish
  ([`94e9872`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/94e987277e56d50bc9643a24b0f4cfc8e4a949b1))

- **05-01**: Add execution summary
  ([`c603783`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c60378351bec26ac89b3315eaca83552bf03fbea))

- **05-02**: Complete plugin resolution rewrite and subprocess elimination
  ([`568e1be`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/568e1be03e07ad7dac345d22cf7e44c725cb902e))

- **05-03**: Complete CLI import migration plan
  ([`4169245`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4169245c99bcc73a64bdcefd6dde4fff4d833d47))

- **05-04**: Audit skills/agents for CLI correctness, add manifest to CLAUDE.md
  ([`a9a544b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a9a544b0125b83b96dedcd807dda110e74215b0e))

- **06**: Create phase plan — polish watchdog CLI to replicate old-watchdog behavior
  ([`eb1e53d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/eb1e53d76466b3edc43325a281d3c61cc3cb51a0))

- **06-01**: Complete foundational interfaces and lifecycle features plan
  ([`4ef1fa2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4ef1fa25617b1baaedf8cb9a52332d33b297477b))

- **06-02**: Complete plugin plot delegation and telegram formatting plan
  ([`40f6d7e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/40f6d7eca3575945786de32f065142a914225e54))

- **CLAUDE.md**: Move GitNexus HARD GATE outside gitnexus:start block
  ([`179ad25`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/179ad259aecba12401cdb4fe2a8f5909402955ab))

- **CLAUDE.md**: Trim to SOTA context efficiency — 79% smaller
  ([`935ab58`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/935ab58f3f27d3cb9901191d95e1a1d176c6902a))

- **CLAUDE.md**: Update GitNexus index name to feat+shared-benchmark-filtered-lineage
  ([`7120ec8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7120ec88aa13cf894ec8a950578fe676bc0b03d8))

- **cli**: Document gigaevo inspect, manifest, and run-spec shorthand
  ([`7d3c952`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d3c95218b651ca026a54b8f0c1c25fde1dd9ae8))

- **closeout**: Heilbron/asymmetric-iterations experiment results
  ([`d1815d4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d1815d4d61177d5bdfedd68b4ae6aa0d6abbf1b1))

- **E**: Comprehensive Experiment Manifest section in tools/README.md
  ([`6e30e4b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6e30e4b1d64eece36273bc12472266565e1bdc1d))

- **experiment**: Rewrite schema section for v2 sub-model architecture
  ([`4bfcc5a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4bfcc5a14f433da67e6f362056efdf4c56a6d6d7))

- **G**: Update skills — gigaevo CLI instead of deleted tools/*.py
  ([`83f6a9c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/83f6a9c07f36a2c849e1494c116d6f6c3eb7c418))

- **heilbron**: Log deadlock issue and restart #2 in issues log
  ([`535a4c3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/535a4c3c736b25611c615c57caeec9d2517b0e8a))

- **heilbron**: Note LineageStage filter replaces scalar-trend stage
  ([`a23506e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a23506e85f9b6f259a6f2fbcf7a77b7032c5dbaf))

- **heilbron/adversarial-repro-v1**: Catalog 25 pre-existing test failures on main
  ([`0e33a49`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0e33a491c880d3eac54b1bf83d03942d954c33c4))

- **heilbron/adversarial-repro-v1**: Sync local PR_DESCRIPTION.md with final body
  ([`b0a3402`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b0a34021c8cfe904254e767df25613424df059f0))

- **heilbron/k5-budget-loose**: Redesign + failure modes + sandbox checklist
  ([`d595a7b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d595a7b98c13040c2f3e25292469d37e717a580f))

- **heilbron_adversarial**: Clarify helper imports and signatures in task descriptions
  ([`4a789be`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4a789be494b6942ae45ed490da1027dd18069882))

- **ideas**: Defer G-side LineageStage shared-benchmark filter
  ([`515eb25`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/515eb25985f4a103887d5f127710639959ec2d1b))

- **k5-budget-v3**: Post-fix scorecard + variable-to-proof mapping
  ([`29e4765`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/29e4765f41379da99dfffdc15d53b62069d487be))

- **k5-budget-v3**: Record C1 fix verified via three independent sources
  ([`845b5ae`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/845b5ae5b839e5c628c33fab83ba0a8355305b08))

- **k5-budget-v3**: Reject option (d) DB-shift — db=0 reserved
  ([`b034d12`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b034d12c47aab7530e5099ca1c8012668f1a926c))

- **lineage**: Document is_valid writer invariant in aggregator
  ([`d02f4ab`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d02f4ab2df40a931d22e126ec56b8ea4498943e6))

- **manifest**: Add experiment module README
  ([`6871867`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/687186752ab93e1749a8ad66884984e091370eff))

- **patterns**: KF-10, KF-11, KF-12 from heilbron/adversarial-repro-v1
  ([`368eb33`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/368eb33bb4fb2715d94570fd8cafea0fbda463da))

- **patterns+launch_generator**: Note that \${} YAML escape is cosmetic (I-04)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **phase-03**: Mark phase execution complete
  ([`ffbe0cf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ffbe0cf06018b72ba35c37b4c2692b8b26a146ba))

- **phase-05**: Complete phase execution
  ([`4bf0525`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4bf052531522f1b7231944c201b89f34332c6d97))

- **pipeline**: Correct LineageFilterConfig docstring (min_shared>=1)
  ([`40d1406`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/40d14061a01b05365984cbe3f7c465aa6bbaa423))

- **plan**: Aggregator-first metrics implementation plan (9 tasks)
  ([`53b4a68`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/53b4a682edb3bd38a48cd5e0da63ea799b98fb2c))

- **plan**: Drop remaining +aggregator= references in Tasks 5/6/8
  ([`87eadb7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/87eadb7f3aa2e36d5d96b618371040b38ece6a68))

- **plan**: Use aggregator=none default + NullAggregator sentinel
  ([`46ef966`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/46ef966df6477203ce010d7e65173aee4b4b2049))

- **plans**: Paranoia tasks 19A-19F + hard-rename stopper (Option A)
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **plans**: True-JIT-refresh steady-state engine — 21-task implementation plan
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **plans**: Two-sema mutation-throughput implementation plan
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **retrospective**: Heilbron/asymmetric-iterations closeout + research strategy
  ([`0b46cfa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0b46cfa9615d54a437bf83968abc2e2de381a737))

- **skills**: Wire pin contract + LAUNCH_PREVIEW.md into workflow
  ([`0a65d4c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0a65d4c5f69c360cf816b87f33d6d2a5d95316cf))

- **specs**: Mutation-throughput two-semaphore redesign
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **specs**: Record JIT engine dry-run smoke results
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **specs**: Refine steady-state redesign — async stream, multi-parent, iteration axis
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **specs**: Steady-state engine audit + true-JIT-refresh redesign
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **state**: Record phase 5 context session
  ([`e678c9a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e678c9a3ce45a88762dc84c6935620d54954675c))

- **template**: Surface Redis prefix == problem.name convention (I-06)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **v3**: Log-audit V9 proven, LLM reach blocker documented
  ([`a43144d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a43144d9c59469e51c90a665edfe53bd3e1300fb))

- **v3**: Polish task description + add v3 metrics to heilbronn/points_13
  ([`5a6b4dd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5a6b4dd9ca9205a2da6bda452ed8824b841a481f))

### Features

- Add _llm_active counter + LLM/DAG occupancy breakdown in BackpressureSample
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- Add adversarial co-evolution problem infrastructure (Tasks 1 & 2)
  ([`35dd142`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/35dd142e0f31cc2fbb025cff99e7571a4df58277))

- Add adversarial co-evolution watchdog with PR plot uploads
  ([`fcc61e0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fcc61e08b8240a838a31e0fc694f242221063393))

- Add adversarial pipeline config and optimizer launch script
  ([`63e9551`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/63e955143c9c013cd8c7584dbf0eddda383cafb3))

- Add adversarial tests for memory system (36 tests)
  ([`0c42178`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0c42178b8d76acf230269cf091d3f2f90e14fe62))

- Add anomaly-detector agent + failure patterns library
  ([`3da6aa9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3da6aa9978306b76ec6c775cf2639986fda31beb))

- Add higher_is_better parameter to opponent selection
  ([`d7d8d23`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d7d8d2301289cf69d75323979ff238995c796b05))

- Add preflight Check 22 — stopping rule CRITICAL gate
  ([`de651ad`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/de651ad8633709e681b4bebce6b150ef94a96f49))

- Adversarial co-evolution pipeline v2 + experiment preregistration
  ([`172a83e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/172a83ef5b8ba0ae3a616b4d159f126fe633cfd2))

- Full experiment lifecycle automation with 3-gate approval flow
  ([`44ac111`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/44ac111427c7fe0b0500e28b53d7c800fe81d231))

- Integrate superpowers skills into experiment lifecycle + update CLAUDE.md
  ([`f2443fc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f2443fcfe96ca9004a82e6c8f74eae736fbf7542))

- Integrate superpowers skills into remaining project skills
  ([`9f91b1c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9f91b1cd26fa5c0a799864bbae671d41c91c0608))

- Iteration-1 structural improvements to experiment lifecycle skills
  ([`c896383`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c896383bd0606f094bd8c2a9c3434d81be162821))

- Iteration-2 research quality improvements to experiment lifecycle skills
  ([`1caff89`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1caff89c8bda8f7670526c11640b9ed53d67e082))

- Iteration-3 research quality improvements
  ([`65142e0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/65142e08a40602f755854df00acd564224a81c4f))

- Iteration-4 research quality improvements — deviation docs + reproducibility
  ([`bc1198b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bc1198b5a4a88a8bf014d9310a549f355e102bf2))

- Paper-quality CLI polish — formatting, frontier control, arms-race plots
  ([`2acd4c9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2acd4c98e374d24d445b2062cdd45e239c36b56e))

- Polish gigaevo/experiment/ module — delete preflight.py, 32 run_watchdog.py files, add launch
  orchestrator
  ([`0b9a69a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0b9a69a5308f49efc1ca4f26c26ca7e583bea6ac))

- Steady-state adversarial co-evolution support
  ([`d2fefa4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d2fefa436045baa0a8f8a696e2f04eca3dd2b81b))

- Surface peak LLM/DAG split in profiler text + HTML output
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- Watchdog automation — self-restart, post-launch health check, anomaly-detector recovery
  ([`ea7380c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ea7380c61f2720069f0940d52563ccb76d128069))

- **01-01**: Implement manifest CLI subcommand group
  ([`bf19391`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bf19391a4772e7d08fd1df99e4cb1b512d9caa78))

- **01-02**: Migrate 6 skill/agent files to gigaevo CLI calls
  ([`6547523`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6547523e2006196bcc695873504af192f12e89a0))

- **01-03**: Migrate experiment-closeout and experiment-checkpoint to gigaevo CLI
  ([`6386d8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6386d8a28854575bf6993d0e473753ca583ebc2e))

- **01-03**: Migrate experiment-launch and experiment-restart to gigaevo CLI
  ([`4399648`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4399648941ec95f7994e45f421687c3b2a0294ad))

- **01-03**: Migrate run-experiment and merge-rules to gigaevo CLI
  ([`74e9d08`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/74e9d080f4f94f191713880718e6ef1448630467))

- **02-01**: Add sentinel value filtering to prepare_iteration_dataframe
  ([`f325691`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f325691a40778d43f3095f8b7a7cb9a60c941743))

- **02-01**: Wire sentinel_value through _fetch_run_data in plot_group.py
  ([`e7a9b42`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e7a9b423eba1dc1216bc5245822f4de70d24735a))

- **02-02**: Add post_step_hook to EvolutionEngine, wire via Hydra, activate in launch.sh
  ([`1afc3c4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1afc3c4a824409054169dd07c7f30c7b46e9c40b))

- **02-02**: Rewrite CompositionInjectionHook with code composition and delta gating
  ([`acf8776`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/acf8776f44e9b0422523899bd4b8d02c915b3a79))

- **02-03**: Add per-G-program D selection to GradientInPromptStage
  ([`a371ea7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a371ea7c6b5229d35852750062cb785499181de9))

- **02-03**: Create DGImprovementTracker with Redis sorted set storage
  ([`840c990`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/840c99084dbd11bbc3372f5d619dae33706c1fbe))

- **02-03**: Wire DGImprovementTracker into CompositionInjectionHook and Hydra config
  ([`2fd09f2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2fd09f293dc00c948848e7eb7df6a0ea9f9f45e8))

- **04-01**: Add EVENT vs ISSUE format guidance to issues log template
  ([`146f7b2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/146f7b2df4a76fd42de7da155ba656801303a2f9))

- **04-01**: Add Known Failures section to PATTERNS.md with 5 real entries
  ([`085a35c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/085a35c7a7183d6025a230a782dc66061cea0bd2))

- **04-02**: Wire GSD plan generation and event auto-capture into experiment-launch skill
  ([`3e280bc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3e280bc9b087f4391c7351f2f1c5a41e55a57e97))

- **04-02**: Wire GSD plan generation into experiment-implement skill
  ([`1cf0fd7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1cf0fd798bce61f0fca49e168a949049949cd656))

- **04-03**: Add event auto-capture to checkpoint and diagnose skills
  ([`87e37bb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/87e37bba775a7cd056f8d69ac1fccf20c65fe075))

- **04-03**: Add event auto-capture to experiment-restart skill
  ([`39cff79`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/39cff7955f0f88be7b92b8810ef00065201ac910))

- **04-04**: Add fix report generation and PATTERNS.md status updates to post-experiment-fixes
  ([`07b4f03`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/07b4f0350c542e07afd9ed4b45a77bee5bc7eb50))

- **04-04**: Add Known Failures promotion to experiment-closeout Step 13a
  ([`a6c0ad5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a6c0ad5f0b4fe783c8690f4ec8f122ab529326a9))

- **05-01**: Create flush_ops, dataframes, plotting modules — migrate tools/ functions
  ([`2935800`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2935800e79402baf8b1c4c2036c5362ef99542ff))

- **05-01**: Create gigaevo/monitoring/manifest.py — Pydantic manifest ops with Redis locking
  ([`53d73e8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/53d73e8b7b5b1802e199307e347733cea5f21c5f))

- **05-02**: Delete heilbron plugin, rewrite solo/prompt_coevo with inline matplotlib
  ([`fb31e17`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fb31e17475acf99974b28501dd5ce0b47597231d))

- **05-02**: Rewrite resolve_plugin, add WatchdogPluginOptions, rewrite adversarial plugin
  ([`da1e77d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/da1e77d4d2de08c331f28d964d2b06c10d992f7f))

- **05-03**: Migrate manifest_cmd.py to Pydantic shape
  ([`edd6ec9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/edd6ec9f7bc369c1d3f6d08d8533a13e5c71b0d7))

- **05-03**: Replace tools/ imports with gigaevo/ package imports in 7 CLI modules
  ([`def8034`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/def80347d14624f7e958dd7ff09c986651a437d7))

- **06-01**: Add model drift rule, Redis checkpoint/completion markers, and NO_PROXY auto-setup
  ([`851611c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/851611ce80ef157364b115e3ff533f965bf6cfaa))

- **06-01**: Add WatchdogSection manifest schema, format_telegram_body ABC method, and config fields
  ([`a210e58`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a210e58262e709aab3eb6c7b5ce861526835e9af))

- **06-01**: Add WatchdogSection schema, lifecycle features, and CLI manifest wiring
  ([`26719a4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/26719a41412c3851aef20cd715611e0394713a72))

- **06-02**: Add plot retry logic to WatchdogEngine
  ([`4a3e987`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4a3e987627685f9ed96bb48e82f930ad164055d4))

- **06-02**: Delegate plugin plots to CLI subprocess, add format_telegram_body
  ([`dfd7b0d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dfd7b0dc4c62c1b9e8690c92ab9cec5e97342702))

- **06-02**: Wire format_telegram_body into WatchdogEngine cycle
  ([`d1500e6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d1500e6f33ef7afc17dc5896ec49caffa264cf99))

- **06-03**: Add experiment-path plot upload, Redis rolling comment, and baseline wiring
  ([`26cc0dd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/26cc0ddb76245add20c1d70fdbb8e5e5c5257da0))

- **06-04**: Add YAML test fixtures and fixture-driven integration tests
  ([`16e7512`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/16e75129b460518a1ac75e0e9e64f9edff46269f))

- **06-05**: Integrate watchdog config into experiment lifecycle skills
  ([`b162d35`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b162d35e78357157eddff701c691b1954128e3a1))

- **19-01**: Create gigaevo CLI package with status and collect subcommands
  ([`eddf732`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/eddf73229eb3f1dec50d67bd6843fe9435417553))

- **19-01**: Implement plot and analyze subcommands with multi-metric support
  ([`8340567`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/83405670ef1b947565cd8083ba69455a12eea498))

- **adversarial**: Configurable opponent sampling mode via OpponentSamplingMode enum
  ([`7e87d64`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7e87d643494f8dbe970fe34600e501c116161025))

- **adversarial**: Deterministic top-K HoF with stable tiebreak
  ([`545df12`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/545df12f35f0174702ce4de58d06d7cdf9c7be44))

- **adversarial**: DGTracker global best-pairs + injected-pair dedup
  ([`2b268e8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2b268e82cb6b7fb287881874ff0a11ad38616a6f))

- **adversarial**: DGTrackerStage with role + length cross-check
  ([`e313b86`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e313b868e6631b791a528d26dce45ee9c78c551f))

- **adversarial**: Gate InsightsStage cache_on under cache_insights_on_opponents flag
  ([`063c9cc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/063c9cc9dd45d2f89e518d4d32da5d8b583c4744))

- **adversarial**: Implement asymmetric pipeline for heilbron/asymmetric-iterations
  ([`e017e4a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e017e4a7d4a80982540ffe110090087820dd6e4d))

- **adversarial**: Insert ParseMetricsStage between CallValidator and consumers
  ([`bafae39`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bafae3997a42252c6ee5bc28ac3dda03720d2765))

- **adversarial**: OpponentResultProvider strategy — cached D, exec G
  ([`19f1b8c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/19f1b8cc6d4a3438aeff5d8d2e3532556f86ac70))

- **adversarial**: Wire DGTrackerStage + cache_on edges in asymmetric builder
  ([`1d25a9a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1d25a9a865b9a1b5b1cb25aed6ff3f9c2eea2f3a))

- **adversarial-repro-v2**: AF-7 — add aggregator= overrides to cfg_run files
  ([`617ee4a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/617ee4a8417940b26d0b23773ffc35db5a36f76e))

- **adversarial/adversarial-vs-solo**: Implement experiment — code, config, launch, watchdog
  ([`4f8d474`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4f8d4741cb992337ff937025c7c8f5566d2978dc))

- **adversarial/heilbron-prover**: Implement experiment — smoke test passes
  ([`65387f3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/65387f3f544347c12c337111c310ee5dab584a9f))

- **archive_gate**: ArchivePotentialGateStage with all fail-open + cascade paths
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **archive_gate**: Enable lazy insights by default
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **autonomy**: Auto-load .env in telegram_notify.py
  ([`8a92a80`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8a92a800e10fff367b4801808d275670153f2e00))

- **autonomy**: Research autonomy stack — SOTA-grounded agent/memory/automation upgrade
  ([`d6f8663`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d6f86634b0870372f289943af429ee64a01b9447))

- **cleanup**: Purge deprecated run_watchdog.py — watchdog is CLI (I-13, I-14)
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **cli**: Add --generate-script flag to gigaevo launch
  ([`f379134`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f379134bca1719104b112532ab48dd0873ee3656))

- **cli**: Add `gigaevo profiler` subcommand for log flow profiling
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **cli**: Track C — gigaevo events plot (general, registry-driven)
  ([`7ea8543`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7ea8543a245083c13042e32717497210c720fb60))

- **config**: Add archive_gate_provider field + build helper
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **config**: Aggregator=none default + ${ref:aggregator} singleton wiring
  ([`5b89768`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5b89768a23b8d85987849047e3ac7429fb4fbaca))

- **config**: Declarative Heilbron aggregator YAMLs + pipeline wiring
  ([`886f596`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/886f5962745279a330341c9ef066e9d862dbec93))

- **config**: Heilbron task-group file — tacit tradition pins
  ([`610affc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/610affc2677355e66ab420b84d496d33ca0af1a5))

- **dg_tracker_stage**: Forward per_opp_metrics dicts to tracker
  ([`91b84d7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91b84d7cebc57ac4c3d77ed21d4cc2eca2953a21))

- **engine**: Bucketed generation-order archive refresh (fixes cross-program tracker race)
  ([`ca423cb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ca423cbd44950811c2c1aa60a397593c821734a5))

- **engine**: Graceful deprecation for retired EngineConfig knobs
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: ParentRefresher + ParentRefreshSelector ABC for JIT refresh
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Post_step_hook timeout knobs; iteration-window stats; deadlock stress
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Refresh_passes=2 closes two-sided cross-program tracker race
  ([`a8da66c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a8da66c6fc14a7bd478076161261ed10098f1061))

- **evaluate**: Emit per_opp_metrics artifact for heilbron pop_a/pop_b
  ([`b799f24`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b799f2457539d7ddf05eb58ce02510afb49f7379))

- **events**: Canonical events Phase 2 — emission seams + adversarial port
  ([`318b439`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/318b4390b98853a5b971e504dbb6fd566da6fd94))

- **events**: Canonical events registry Phase 1 (split registry + auto-registration)
  ([`8d0bbbd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8d0bbbd4a21dfbffb1040d57139e3fbc4a27a854))

- **events**: Phase B3 — registry-backed log_audit + legacy log consolidation
  ([`6250685`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/62506856e98a08bbb516e5766214951dacd76ac0))

- **events**: Track B4 — Redis minute-bucket counters + EVENT_RATE_ZERO alert
  ([`8ff94d2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8ff94d205d62a062554c03f338a9864cc1c164ef))

- **exec_runner**: Apply worker_side_eval hook to result before pickling
  ([#228](https://github.com/KhrulkovV/gigaevo-core-internal/pull/228),
  [`e82121e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e82121e4a7a7ae1c6b3ba601eaadb56c454b8aa9))

- **exec_runner**: Worker_side_eval hook for non-picklable results
  ([#228](https://github.com/KhrulkovV/gigaevo-core-internal/pull/228),
  [`e82121e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e82121e4a7a7ae1c6b3ba601eaadb56c454b8aa9))

- **experiment**: Dry_run helper — resolve Hydra config per run
  ([`91e25bd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91e25bd0ea0bff4677f1c55f655d9736dc15ba6d))

- **experiment**: Preflight pin-check + LAUNCH_PREVIEW.md artifact
  ([`1ea381d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1ea381d10d6e64d32f2bd1023805ad7ed8ecde66))

- **experiment**: Track A — drop parallel schemas; preflight from merged overrides
  ([`3aa871b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3aa871b38529f85715df2719d37ff486d97bbd79))

- **heilbron**: Add watchdog with Telegram + PR comments, set status=running
  ([`794ea57`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/794ea57954083aef1d1f58135557249cc89d592b))

- **heilbron**: Update D task description for adversarial context
  ([`8aed7da`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8aed7da378c43e32817d41a9746b31b5c095bbc1))

- **heilbron-adv**: Smoothed tanh fitness + widened invalid sentinel
  ([`e10ed3d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e10ed3d1963dc9fa0a2fc8876860756aeceaf080))

- **heilbron/adversarial-dynamic-updates**: DAG-based dynamic re-evaluation + soft fitness IV2
  ([`785a807`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/785a8070ece807309ddf71794267c8042f3fc6ab))

- **heilbron/adversarial-dynamic-updates**: Implement adversarial watchdog with 3-panel plots
  ([`9a228d0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9a228d04305f77ceb615af131d41f4448ff4b06c))

- **heilbron/adversarial-dynamic-updates**: Redesign to 2×2 strict GAN variant
  ([`865313a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/865313acb8813c11e2c1b26018d92895fbc26d93))

- **heilbron/adversarial-dynamic-updates**: Set status=implemented, treatment verified, smoke test
  passed
  ([`e1cbfe2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e1cbfe2495cda191732b69525e52f90c0c090dea))

- **heilbron/adversarial-repro-v1**: Implement config, runs, launch.sh, treatment_checks
  ([`5dc0324`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5dc0324dd408792367f7985768a0237d28d3b881))

- **heilbron/adversarial-repro-v1**: Smoke-tested implementation — role fix, treatment_checks,
  I-13/I-14/I-15
  ([`407f32b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/407f32b3d387f4546a24447032912faa05abf7e2))

- **heilbron/adversarial-repro-v1**: Treatment-verifier fixes — pin max_generations, expand
  treatment_checks
  ([`3fc0bea`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3fc0bea5872af46592c41859a0e39d21da81e4be))

- **heilbron/adversarial-repro-v2**: Expose refresh_order+refresh_passes in Hydra; generate
  launch.sh
  ([`4066f6b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4066f6bf07095e25e39e7f6f05cf9a181d68c0c8))

- **heilbron/adversarial-v2**: Arms-race plots, fix checkpoint skill, update watchdog
  ([`f4c83af`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f4c83affcfbdd6d707409e95522fabf190ceba3f))

- **heilbron/adversarial-v2**: Implement bidirectional opponent feedback pipeline
  ([`ffcb03e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ffcb03e646fed12accb9047613fb587e281edd6f))

- **heilbron/baseline-repro**: Implement experiment — launch.sh, watchdog, smoke test
  ([`9340928`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9340928ac02ffc32fec3259946ae9e21fa09dac3))

- **heilbron/d-smoothing-minimal**: D-side tanh-smoothed fitness
  ([`3eb41b4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3eb41b43f42d0be09a94b0782143eb871e54a9c2))

- **heilbron/d-smoothing-minimal**: Generate launch.sh
  ([`26bbce9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/26bbce949108c101ae399e268394172d7fb2910d))

- **heilbron/d-smoothing-minimal**: Treatment verification
  ([`fdf4685`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fdf4685621d368df357621f0f6f2fac4e3a327f6))

- **heilbron/d-tanh-no-lineage**: Add disable_lineage_on_improver kwarg + ordering-safe gate
  ([`504c89a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/504c89ab5cde33a6ec9ba49513d6773dbd24766c))

- **heilbron/d-tanh-no-lineage**: Wire disable_lineage_on_improver default in adversarial_asymmetric
  pipeline
  ([`dced7fc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dced7fc7eec08db50076186b78166c373664a6dd))

- **heilbron/k5-budget-loose**: Implement experiment — runs, launch.sh, watchdog
  ([`bca2d5c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bca2d5c84c9b92a1b9e8c4516b0e51236adf79cb))

- **heilbron/k5-budget-v3**: Add tracker coverage stages (Phase C-2)
  ([`7a2f09f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7a2f09f8dbd06483b3b856213b74bc92a001d4ed))

- **heilbron/k5-budget-v3**: Complete Phase C — 2D BD block (configs + metrics)
  ([`12611b1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/12611b1323716d663cb3d6bdab7847bc9a9c2e77))

- **heilbron/k5-budget-v3**: Complete Phase D scaffolding — 16 runs (8 k=3 + 8 k=5 pairs), servers
  assigned
  ([`58d343d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/58d343d0edc9ae7a829a4437bea2b40dde170070))

- **heilbron/k5-budget-v3**: Implement CellStratifiedRedisOpponentArchiveProvider (Phase C-1)
  ([`dc64475`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dc64475f9989792a885a37232a8ab6d7fbc8a384))

- **heilbron/k5-budget-v3**: Phase B Prong 1-2 — tracker inverted indices + TDD tests
  ([`599eb0d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/599eb0dd5872cb73daf627e179e91c16c8838e1e))

- **heilbron/k5-budget-v3**: Phase B Prong 2 — shared-benchmark lineage stage + TDD tests
  ([`b646d47`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b646d47eef6bb487bcf300dc3930565c1e56d7b2))

- **heilbron/k5-budget-v3**: Phase B-log/audit — structured logging + audit tool
  ([`3320abb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3320abb75012a2a6bdd44d40b88e3b3b2e11a27a))

- **heilbron/k5-budget-v3**: Resolve #27 — trim to 8 runs (n=2), DB 1-8
  ([`5f69378`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5f69378075f4918f01c0e801764b98e186cd1021))

- **heilbron/k5-budget-v3**: Smoke PASS — log_audit v3 validation + VERIFICATION READY
  ([`a3a49e9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a3a49e9c3806fc87644b951afb2157e98c0ec025))

- **heilbron/k5-budget-v3**: Wire CellStratifiedRedisOpponentArchiveProvider
  ([`3687f57`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3687f5704677adb5caad11fa1a03fdff7050d703))

- **heilbron/pop_a**: Evaluate.py returns (intrinsic, artifact); golden-vector test + refactor
  ([`cc55f80`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cc55f80ebad74ee8a5c4c6490eabcae226fb859f))

- **heilbron/pop_b**: Evaluate.py returns (intrinsic={}, artifact); golden-vector test
  ([`0c24210`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0c2421095a7955363a01f9be1f6d0c5a92ce3e6c))

- **heilbron/v1-honest-repro**: Implement experiment — code, config, launch, treatment verification
  ([`5de9a5a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5de9a5a82977479472d30300c7b9865022d8c9d9))

- **heilbron_adversarial**: V3 metrics + v3 task descriptions
  ([`64f9e6d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/64f9e6d79bf26a17d290f3b056c832906333aa68))

- **hydra**: Wire archive_gate flag + provider for auto pipeline
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **ideas_tracker**: Implement CSV and Redis loaders for standalone usage
  ([`51d0043`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/51d00439a2e984e68f8341ba3a13b57e21616216))

- **k5-budget-v3**: Wire SharedBenchmarkLineageStage for D runs
  ([`0c1dab1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0c1dab1fe6dfe6584da0c7b04cd6153d171b8439))

- **launch_generator**: Emit experiment=<task_group> as first override
  ([`b98fea2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b98fea2134301e1fddacff52b16c33654d06e7df))

- **lineage**: Log [LineageStage] n_parents on every invocation
  ([`830273f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/830273fe2f74ba7b8989662d7f537f805630732b))

- **lineage**: SharedBenchmarkFilteredLineageStage — filter + HoF-invariant evidence
  ([`f9e4a2b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f9e4a2b798bb0fc5eee6ccaf36d5bb4943a8c3f3))

- **lineage-agent**: Optional TransitionEvidence renders SHARED-BENCHMARK block
  ([`f35e3c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f35e3c08f751021dba015f437ffec6ad1a63d633))

- **manifest**: One-shot v1→v2 migration CLI (step 6)
  ([`d3c8c54`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d3c8c543df1885ca3ee26f3ff166f4dc176e6fde))

- **manifest**: Task_group + pinned + config_fingerprint schema
  ([`91cf2f2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91cf2f2ebcc02f7d245486b3ea6a17230418514d))

- **memory**: Add CardLoader utility for centralized card I/O
  ([`5842637`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/584263740f4f708e041485a8691ad12d4b5bc3df))

- **memory**: Add comprehensive usage tracking pipeline tests + fix _extract_task_deltas
  ([`fc6525e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fc6525efef29ae66ae1db1bde90a7d3195214d08))

- **memory**: Add MemoryState explicit lifecycle state machine
  ([`5352d54`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5352d5412b178797af93d5f5346841698c738d62))

- **memory**: Add UsageEntry and UsagePayload Pydantic models
  ([`0e1bf67`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0e1bf67976e681f25abef45e59c6a96c6f0d8947))

- **memory**: Upgrade usage field to typed UsagePayload model
  ([`51300a9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/51300a9f599fe2dc777a0e00fbccb5c58cf1cd5c))

- **memory**: Wire MemoryState lifecycle tracking into AmemGamMemory
  ([`7b25c0b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7b25c0b973c6468e909881cbd5402db78445c937))

- **metrics**: ConfigurableAggregator + OutputSpec primitives (general adversarial)
  ([`11f9afa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/11f9afa78e628e007865d2e7dc316f26e513fb7f))

- **metrics**: MetricsContext.is_valid / is_sentinel helpers
  ([`d2fbb19`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d2fbb19a0c01d7b1d96a5b26f1791fc226fd38b7))

- **metrics**: NullAggregator sentinel — signals 'no aggregator configured'
  ([`a1ad668`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a1ad668eac74882b978ac535a41bfd8f2a18e77e))

- **metrics**: Value_counts annotation on format_delta_block
  ([`3d5bf0e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3d5bf0e3be0341091449307ff0d17725eb09e754))

- **monitoring**: Emit LLM_CALL canonical event from MutationAgent
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **monitoring**: Excluded_events opt-out for event_rate_zero alerts
  ([`6054c32`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6054c322600edb91739d0fc05edd03e3d5f03974))

- **monitoring**: Live flow profiler daemon for run.py
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **observability**: Queue-depth scalars + per-stage LLM token attribution
  ([`83fe69c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/83fe69cd312d1cb413c6dedbab3b921de6e5897a))

- **pipeline**: ArchivePotentialGateStage — skip InsightsStage for dominated programs
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **pipeline**: Wire ArchivePotentialGateStage opt-in via builder kwarg
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **profiler**: Utilization view — LLM/exec overlap + mutation archetypes
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **profiler**: Y-axis last-N window + iteration-ordered rows
  ([`69eb8b1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/69eb8b1758f5a04d374f22d27ab27e3460f3bd52))

- **scheduling**: Add CachedFirstPrioritizer for re-eval-first DAG launch
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **stages**: CacheOnlyInput for cache-key-only stage inputs
  ([`25975f6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/25975f6547a05a8898159a4b9ed79291381c13d8))

- **stages**: ParseMetricsStage — aggregator-driven metrics composition
  ([`b46d1f2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b46d1f26759e7befa614714d6c352c1d7802e1aa))

- **tracker**: Per-G full metrics dict (replaces scalar-delta storage)
  ([`dfefc09`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dfefc09633e5061165a49c7570c1d46463ec272c))

- **v3**: Gate 0 FAIL → fallback BD + D-in-prompt amendments
  ([`df45812`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/df45812c1572a9d9d88ae572ddc53f6c0b191b58))

- **watchdog**: Dual-line plot — frontier (thick) + per-gen mean (faint)
  ([`55be349`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/55be349d2b0900d7c7a35b440d3a51ffaee958ae))

- **watchdog**: Fail-fast validator for adversarial role requirement
  ([`8e90f1b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8e90f1bcdf1feec711d17e3f7086fcb2ef2bebee))

- **watchdog**: Send hourly plot + table to Telegram
  ([`fdecc34`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fdecc3409174efeb4c2f4d91605dfb3d60ece107))

- **wrapper**: Expose worker_side_eval kwarg on run_exec_runner
  ([#228](https://github.com/KhrulkovV/gigaevo-core-internal/pull/228),
  [`e82121e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e82121e4a7a7ae1c6b3ba601eaadb56c454b8aa9))

### Performance Improvements

- Batch _persist_index calls in memory system, eliminate double-writes
  ([`7dae9ae`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7dae9aec9ef9c86dc3763ee4b4b4b1cc966ed035))

- Eliminate double memory_system.read() in _upsert_local_note_fast
  ([`c219d8c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c219d8cc14e34fe23ebbf2b44ef9cd6582d527bd))

- Quick wins — set diff, shared serialization in rebuild
  ([`cdc130f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cdc130fc339b286e3e81b266e2c8bbf8142e3824))

- **lineage**: Skip wasted LLM calls for failed children
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

### Refactoring

- Add _has_agentic property, TYPE_CHECKING imports, compact warnings
  ([`0f971eb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0f971ebf48b162d89f90b1f13cdf3aee4197e48d))

- Add aggregation.py + pipeline.py + __init__.py + end-to-end tests
  ([`e08ea5f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e08ea5f1c7259e1d7cc5088119e2e4782f2eabd8))

- Add ConfigDict(extra="forbid") to all ideas_tracker models for consistency
  ([`9cffbc0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9cffbc07d1cb8316bb32b27bb2ad965300a7b8f1))

- Add ConfigDict(extra="forbid") to usage models, use model_validate for Pydantic v2
  ([`1cab912`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1cab912007410796aa733cfb00ccccfe914aa0b6))

- Add origin_analysis/events.py + tests
  ([`8128482`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/81284824f1e709ef2b9650a28aef53a1df41589f))

- Add origin_analysis/loader.py + tests
  ([`dfbd53e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dfbd53e7110a28069f82eae4f707a7d0cecbc7da))

- Add origin_analysis/quartiles.py + tests
  ([`8608579`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8608579d49c2feb70049c306c94ae157cdaba42e))

- Add origin_analysis/siblings.py + tests
  ([`9bb8963`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9bb896334d9fc142e66c6e8cf9f8d974a253c815))

- Add origin_analysis/statistics.py with pure math helpers + tests
  ([`a146f1c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a146f1cbd1752da60b4651589cc0a5f71dd99d6d))

- Add origin_analysis/types.py with shared dataclasses
  ([`53c06cd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/53c06cd2507a16ce1f29eff80f52fcb8a7b90eed))

- Convert all dataclasses to Pydantic, enforce kwargs-only
  ([`60d941f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/60d941f197f9fee768ca4d5a2c22967ea395331c))

- Default DAG sampler to CachedFirstPrioritizer and gate InsightsStage on validation success
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- Extract _refresh_prompts_from_fetcher from build_prompt
  ([`360ef3c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/360ef3c4f3cb40b43ed3e013b5c68e4b85800bc7))

- Extract CardIndexStore — consolidate 6 dicts + 4 methods
  ([`e1703fd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e1703fd399240de12a34f2e5f196868987376734))

- Extract GAMRetrieverManager — isolate retriever lifecycle
  ([`d85f191`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d85f191589325e23250090032bb8d7619b91b8c2))

- Extract LLM/storage factories, slim memory.py to 487 lines
  ([`cdb87c1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cdb87c15bc32eb9a602fe25e95a3bcae184b66f5))

- Extract locking to gigaevo/experiment/lock.py
  ([`c55d166`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c55d166bed5cd7ad677b7ad77de62fdbc660e32b))

- Extract save_card decision logic into CardDedup.process_incoming()
  ([`6f64dc4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6f64dc404e3279fcf745b2a5b851fc466d01a79c))

- Extract search and synthesis pure functions to card_conversion
  ([`723fb06`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/723fb0663c69c53f85774dd3832ec7698afc02ae))

- IdeaTracker → PostRunHook, Program-native memory pipeline
  ([`63b0eaa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/63b0eaae2bc2b483d4faa9eb1555f60d53b60835))

- Migrate 14 test files to MemoryConfig via make_test_memory factory
  ([`169951e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/169951e943e03090c22d5d7351efb8ad3829b4ac))

- Migrate from tools/*.py to gigaevo CLI, delete 16 duplicated/stale tools
  ([`4f936f8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4f936f809dd40a3357c0b1c0c8769f1178f4dc55))

- Move entity mapping management to CardStore
  ([`035617f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/035617fcb6ff92e83ea513d69b462287848f641a))

- Move manifest.py to gigaevo.experiment, eliminate sys.path hacks
  ([`6f590a6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6f590a6b2173eb5ed9db9d3185420282aeb32e5f))

- Push path derivations + API save/delete into collaborators
  ([`69d4f02`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/69d4f027804809290d749051066414d4720fe8b8))

- Remove backward-compat attrs, read config directly
  ([`0c22c5b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0c22c5b38f515130aded89ea814559fc7dc219a7))

- Remove backward-compat properties, migrate 39 test refs to config API
  ([`c1669f5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c1669f53a2a9c51f559cf2535d1e77ffa86bdb88))

- Remove legacy kwargs from AmemGamMemory, clean MemoryConfig
  ([`5aa142e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5aa142eebe9ea0d6ee91c4baddadf59e27cb376b))

- Remove no-op branch in parse_response rewrite path
  ([`ced9fbc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ced9fbc447740b0011427d5487b013ab8d68c6fc))

- Rewrite adversarial guide as general reference, remove broken tests
  ([`694a6ca`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/694a6ca93b429bdec4c6fcd128aaf3a7df049ade))

- Simplify _build_memory_block — collapse double-check into early return
  ([`7147005`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7147005df9df34f50cc6de8b597467b8bf16944c))

- Slim memory.py to 481 lines, eliminate deprecated functions
  ([`c46eff5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c46eff50df27f54747d8362fa54a6ef3099e18f9))

- Update statistics.py to use analyse() from new origin_analysis package
  ([`d41a4ff`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d41a4ff0fd6b80b38acc80c8ffc674651e2cea7b))

- **adversarial**: Code-level D∘G composition with permanent dedup
  ([`2c5ef92`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2c5ef926c1b74a83b746c57759ef245ee3c8b91e))

- **adversarial**: SharedBenchmarkFilteredLineageStage reads snapshot
  ([`6297f0b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6297f0b46079d2e99e049c6b73ed50d8a0811630))

- **adversarial**: Sync.py reads engine snapshot
  ([`e1dcb5e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e1dcb5ebd8a860ec42d7473bad23b7e1562fe270))

- **cli**: Consolidate record_pids + pr_comment into gigaevo CLI
  ([`4137f3f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4137f3f8c9b103d93e0c84e8327632fabc6fd9b5))

- **cli**: Delete orphan collect.py + analyze.py subcommands
  ([`dcaf497`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dcaf497bee7c50ee2268e8ab9d90fe4bca2277af))

- **cli**: Drop stale `servers` references in manifest_cmd docs
  ([`3a6166f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3a6166f185e796c1cbde80fa01b1180140aba6b8))

- **cli**: Expand root --help with Examples + argument-ordering note
  ([`abd1d9c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/abd1d9cb7fffdd83df8498c30a6f1be1bc4b9c09))

- **cli**: Expand sparse docstrings + document metric auto-detection
  ([`2b885d0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2b885d0806fd1206cce117369d07cbc5fa7c1371))

- **cli**: Hard-remove `manifest set`; route `update status` through state machine
  ([`3963fb8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3963fb86b0a9c267644851aabc2f84b0bb86635b))

- **cli**: Phase 1b — delete lifecycle.py, redirect to skills
  ([`5e8ede6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5e8ede6c043d40c8ec468b41b38db4f4cd6a7ca5))

- **cli**: Rename inspect_cmd function to inspect (bare-name convention)
  ([`e1e4875`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e1e48758e145e09495d2c09f6211ff942a00cd47))

- **collector**: Set *_in_iteration aggregates to None under JIT engine
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **D.3**: Gigaevo manifest reset-status — delete reset_status.py, update docs
  ([`a36e0d9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a36e0d92f0cdaf24f3754512d1668bf7d900d6b2))

- **D.4**: Gigaevo preflight — delete preflight_check.py, add CLI command
  ([`12b7eb4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/12b7eb4907cfa22b47f00e18d9f5cca0dbf02d05))

- **D.5**: Gigaevo launch --generate-script — delete generate_launch.py
  ([`48ced02`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/48ced02130e7092f23d9770ef887283aea8b36c1))

- **D.6**: Gigaevo flush --kill-only — delete process_cleanup.py
  ([`241f354`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/241f3543f02ac4b8cac5d28fcce0c57afe6bfd28))

- **dashboard**: Read/write engine snapshot in demo seeder
  ([`7fef425`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7fef4253c390bdeaa2f848bd42df59e716800946))

- **dg_tracker**: Schema-agnostic record_batch + delta key
  ([`cb5493f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cb5493f480f9f0a95314f1dd2ff35bc6ae5778ba))

- **diagnose**: Reads engine snapshot
  ([`38dbc8f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/38dbc8fdd4a912d1fab86fba19b8c1e71947174d))

- **engine**: Add EngineSnapshot with dual-write
  ([`3bdf011`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3bdf011378ceb0a3bf0297f3bc26db6252905a8c))

- **engine**: Apply PR #227 review fixes — naming + deprecated test cleanup
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Collapse elite→parent indirection in mutant_task
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Delete generational EvolutionEngine.step() / run() loop
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Delete legacy engine:* scalars
  ([`49f9c23`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/49f9c23e9cd93f62deb9ddc892a4be4b9333ad86))

- **engine**: Drop dead code + fix cancel propagation in final sweep
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Drop dead error counters + step() vestige, inline helpers
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Drop dead mutation_ids branch + dead fields, lock schema with extra=forbid
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Drop redundant CancelledError arm + tidy Any import
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: JIT-refresh polish — empty-archive backoff, metric wiring, vestigial
  GenerationBoundary ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Make stopper authoritative, drop max_generations field and ETA logging
  ([`058191d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/058191d39b70258e66d522b8a69e3a0dba4ed3a5))

- **engine**: Replace _in_flight_sema with _producer_sema + _buffer_sema
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Rewrite max_in_flight docstring for two-sema semantics
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Single-counter total_mutants; drop refresh_pass; hard-rename stopper
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: SteadyStateEvolutionEngine composes dispatcher + ingestor + ParentRefresher
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: True JIT-refresh steady-state engine
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **events**: Delete LINEAGE_TREND (replaced by plain logger line)
  ([`b421c85`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b421c8565dbcb869c331bbcac92bcfff7ca09715))

- **experiments**: Plot_arms_race reads engine snapshot
  ([`ef3515c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ef3515c4254741ff258261b9f4663096d1b8f761))

- **heilbron**: 2D selector option rename + metric spec cleanup
  ([`a9ebf30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a9ebf30843d4b7664bd566227039c98185422a14))

- **heilbron-adv**: V3 clean naming — drop ALPHA, unify on (fitness, wins)
  ([`3897bcd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3897bcd53f4d083661f24662541335630dbe56f8))

- **heilbron/adversarial-dynamic-updates**: Replace GIGAEVO_SOFT_FITNESS env var with separate
  problem dirs
  ([`d3b6a1e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d3b6a1e77f4277dea7381cfe925d1e592feee0c2))

- **idea_bank**: Wire UsageEntry/UsagePayload models into build_usage_payload
  ([`4774dc5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4774dc5efe1e4455a11a7cf49ab8d51c06776e0c))

- **ideas-tracker**: Add analyzers.py — Analyzer protocol + ClassifyingAnalyzer + ClusteringAnalyzer
  ([`327a8a2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/327a8a21da748b136b744da9cfd1a9665f3f51ab))

- **ideas-tracker**: Add idea_bank.py — IdeaBank replaces three-layer bank
  ([`941dfee`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/941dfee495aeb5cc7e41f5b3d7cd133c1f96a7eb))

- **ideas-tracker**: Add llm.py + move prompts/ to package root
  ([`6ce0440`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6ce0440f7afb95c16cd464bd086fed6f75aac673))

- **ideas-tracker**: Add models.py — Pydantic models + normalise helpers
  ([`1eb9523`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1eb952341133a89a1d11a9db6f62203e6b6a43b5))

- **ideas-tracker**: Rewrite ideas_tracker.py — clean pipeline + _SessionLog
  ([`59ff9e0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/59ff9e07a6eb530bc14d409fc7701de93ec8b4d8))

- **ideas-tracker**: Update tests for deleted components/ and utils/
  ([`b8cea26`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b8cea26493e8c44c62e46e99e7e831c7e2ee42b5))

- **ideas_tracker**: Move pandas and origin_analysis imports to module top-level
  ([`dd58c05`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dd58c057d6dcc72cda1064882dc86c7da301e4ba))

- **imports**: Phase 2 — migrate all callers to canonical manifest location
  ([`da8bef4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/da8bef45a0ceb205bd5fd423b3cede4025fdef2b))

- **manifest**: Add typed sub-models for checkpoints, stopping rule, notifications, config
  ([`ce33f6b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ce33f6b197c0060c1f1f141e99571c48103fe877))

- **manifest**: Add v2 sub-model groups + v1→v2 migration
  ([`7793c70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7793c70a6cca00e3dc2b27df05fb1edc6060c029))

- **manifest**: Chunk 12 — ruamel.yaml round-trip IO
  ([`b937aa4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b937aa4f26280817ea07a3627d593e9c9ea5892e))

- **manifest**: Chunk 14 — ConfigSpec.extras via model_extra
  ([`1b7b679`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1b7b6796e7fccf70c84ef41e626578a2fb3b7d5d))

- **manifest**: Chunk 15 — split generate_pr_description into pure helpers
  ([`0718eac`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0718eaca89f8c9819eca5fda559914dad38ee4c3))

- **manifest**: Chunk 16 — update_manifest accepts return-dict updater
  ([`ac62c01`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ac62c0182cb8cbdbc1b2771c39612c025d7861ea))

- **manifest**: Chunk 17 — strict mode for load_manifest
  ([`9ad7d27`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9ad7d27ac74e5434b1e7ad8db22494b39105f4c5))

- **manifest**: Chunks 3-8 — exceptions, load path, env override, readability
  ([`33019cc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/33019cc6c6939383aabc7a8fbce7c0dd705f0e0b))

- **manifest**: Chunks 9-11 — atomic DB claims via Lua + CAS release/refresh
  ([`7d58a80`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d58a804f9c08aaf507fa8eedf8ee168ebdd16f4))

- **manifest**: Define RunRole enum in manifest.py, delete dead tests
  ([`aae9f24`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/aae9f24b2faa2858383561d6a5989ae00460cdab))

- **manifest**: Phase 1 — merge dual implementations into canonical module
  ([`cd9c842`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cd9c8427f8f7377f853e18f8f26a2c4bcb3c5695))

- **manifest**: Remove unused RunSpec.run_env field
  ([`3e54a31`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3e54a3104df78cb72a596f38f4b0d62f1b5ec20b))

- **manifest**: Rename contract.config.extra → shared_overrides + emit experiment=<task_group>
  ([#212](https://github.com/KhrulkovV/gigaevo-core-internal/pull/212),
  [`97f823f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97f823f9c7407bc2e5e165341bf2fa99569ca84c))

- **manifest**: Switch loader to OmegaConf; resolve ${oc.env:X} at load
  ([`defb113`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/defb113a588a58e87d601538d762e4856582285d))

- **manifest**: Widen RunSpec.role to open str; enforce plugin vocab
  ([`c89fccb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c89fccb27ad3faa3a0481deacf7392f4faea5028))

- **memory**: Add custom exception hierarchy and narrow catches
  ([`85ec818`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/85ec818e4af242cb5f03a4cbe2c353a052c6eb4b))

- **memory**: Add docstrings, type annotations, and named constants
  ([`34605be`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/34605be2fec58505dc4612f4d67882c2fcb4c1be))

- **memory**: Add extra="forbid" to frozen config/decision models
  ([`97a9195`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/97a919578b0f3d84985c6fcb7a73c210e11dc2e1))

- **memory**: Consolidate _to_float/_parse_cell/_median into shared module
  ([`9e8eb2d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9e8eb2df76e289935d6fa107dd9a401fe2f1f238))

- **memory**: Convert DedupDecision to Pydantic, fix write_pipeline debug log
  ([`6ba8172`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6ba817218bbdce3a14f09b806e5ed8ef5b6a2ee8))

- **memory**: Deduplicate code and delete dead paths
  ([`fd75056`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fd750564a8d7c9bb1b9556373cfbc7497f328755))

- **memory**: Delete GAMRetrieverManager dead code
  ([`267d514`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/267d5145155bdbbf83db35160a525e75e564cd4f))

- **memory**: Directory reorg — vendor moves, example scripts, docstrings
  ([`9fc84e8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9fc84e8564b4ba8d644e6753ca68811c99741c53))

- **memory**: Eliminate importlib.reload in write pipeline + fix exception scope
  ([`21ad546`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/21ad546eda36578c911e4a2076a66cd32443ae57))

- **memory**: Fix 5 bugs + add comprehensive tests
  ([`a762154`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a7621545d18a9849510ac759e670a8b2fd5f4930))

- **memory**: Improve code consistency and type safety
  ([`8c2c2ac`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8c2c2aca40c49287be44807a382eb94904780c96))

- **memory**: Inline-import cleanup and DRY consolidation
  ([`4814ff4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4814ff4299890e7e16b00169bdf02979471333d8))

- **memory**: Make build_usage_payload a public function
  ([`1347f19`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1347f19d0b473da1f49233076a0666cb2ed18b11))

- **memory**: Migrate config.py to OmegaConf, remove legacy deep_get/load_settings
  ([`618aebd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/618aebdc5cf1af12d4863584b29fe49a777a385e))

- **memory**: Move all inline imports to module top-level across memory system
  ([`3aaef82`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3aaef82e3218d01132199131975f01915fe0fd9f))

- **memory**: Remove dict union from connected_ideas
  ([`444e21c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/444e21c6f7ac29552464452e82d3c836c1578386))

- **memory**: Remove duplicate merge_usage_payloads from card_update_dedup.py
  ([`138f03a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/138f03a1271c0fa548b12f46b4744c6fbcbe70c2))

- **memory**: Rename _document_for_note -> document_for_note in protocol
  ([`9987ea3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9987ea36af8e241e88a979fa9cf6aab3ecf09948))

- **memory**: Rename functions for clarity across memory system
  ([`741168d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/741168d99d3c5534492272dc00de03cce3923afa))

- **memory**: Rename unclear functions for readability
  ([`b3160ce`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b3160ce6bc2dcaedffcd6959c43ac4becbdbe78f))

- **memory**: Rename write pipeline and analysis files
  ([`ac0287d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ac0287d2f49d0111394263e014a256bc9d624b4b))

- **memory**: Replace _safe_float/_median_or_none with canonical to_float/median
  ([`2a68a0e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2a68a0e3449155dd0ee749d1b98d628dbe3377a9))

- **memory**: Replace _to_float in shared_memory with canonical to_float
  ([`1d4ccb5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1d4ccb5b89270c65d6647128e6c96f7fe33a198d))

- **memory**: Replace Any with Protocol types in card_search + gam_search
  ([`a178a3d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a178a3de310b29e4286dc26d22dd5380b04bc705))

- **memory**: Replace env-var/importlib anti-patterns with OmegaConf lazy config
  ([`cd77282`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cd77282f69615e633be9aded48e5d1906d9e7799))

- **memory**: Replace from_mapping() with Pydantic model_validate()
  ([`1b01cef`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1b01cef3136de8e6b7f5b524822e22015d834a58))

- **memory**: Replace RuntimeError with custom exceptions, convert base to ABC
  ([`c3a8f5f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c3a8f5f3aa9dd72a9626fbaccf594a0a867853a6))

- **memory**: Simplify memory_read_example.py + fix unused pytest import
  ([`5a0647e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5a0647eca961fdf818e83960d32a039ce4f8334d))

- **memory**: Split card_conversion.py into focused modules
  ([`deb5b23`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/deb5b235a1461ba04cf66054cc13c5b282c41283))

- **memory**: Strip dead helpers from runtime_config (to_bool, deep_get, etc.)
  ([`3fb7ce1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3fb7ce10aadb03daf87d379acb86c4d14302cc80))

- **memory**: Switch logger calls to lazy formatting
  ([`7d29b45`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d29b4564e295d916bfa2cd6f897f456eafdc82c))

- **memory**: Three structural improvements
  ([`12137f7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/12137f7b3446098ef8747401bfb3171b7285fcba))

- **memory**: Use CardLoader in card_dedup.build_retrievers
  ([`c78417f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c78417f9ca0ae13be03bd6e78e084fd88a949aa1))

- **monitoring**: Redis_queries reads engine snapshot
  ([`d0f1655`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d0f165579ce4f0593102d8b3877521be3eb35b88))

- **pipeline**: Drop blueprint startup banner — subsumed by STAGE_EXEC
  ([`a6c055d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a6c055d1ae27c4d670b00d3b165162c25a5e067b))

- **pipeline**: LineageFilterConfig + replace_stage("LineageStage", filtered)
  ([`f29495a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f29495ae44d0b042d2c07aaa5adf228c6b1ad08c))

- **progress**: Migrate MainRunSyncHook + monitoring to programs_processed
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **prompts**: Coevolution/sync.py reads engine snapshot
  ([`2615dee`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2615dee1900caca422462917ee4ad644f2338091))

- **run**: Drop pipeline banner — subsumed by STAGE_EXEC
  ([`14eaf53`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/14eaf53ff43454da72bb203eff3071187c49d1d6))

- **sbf-lineage**: Aggregator DI + program.metrics-schema evidence
  ([`c4ae79e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c4ae79ef5a2133ce696b490f217f7226dc726f07))

- **stages**: Rename CallValidatorFunction output to raw_validator_output
  ([`2cbab38`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2cbab3827b973da422567c49b25acf1c64649a45))

- **tests**: Move inline imports to module top-level in test_roundtrip.py
  ([`d074ed6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d074ed6d6dd99de2570bc2d6c83825993bed901d))

- **watchdog**: Strict-pydantic + role-based G/D dispatch + arms-race annotation
  ([`91585c6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91585c6069e4a91f4aee267ce3c0209e8b7f5923))

### Testing

- Add E2E memory flow regression tests for full serialization cycle
  ([`352f92b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/352f92b82bbf3a08aec5ddadb1702209fe777ad8))

- Add full-flow memory_platform serialization tests
  ([`1f69760`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1f697600a1b84c0a44c969445b706417ec1183f0))

- Add memory_platform normalize_memory_card serialization tests
  ([`1fe16e0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1fe16e09a66537c07131f4d7a7856380ca58ba36))

- E2E pipeline test for ideas_tracker → memory write (comprehensive coverage)
  ([`6b3bb70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6b3bb70b7c6f874d136e26ba7bf98b8bef1bf4b9))

- Fix usage_updates_path regression test mock
  ([`7e37660`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7e37660b40b4e838aa2d9da16875c3a023ddd9ab))

- Migrate test_memory_backend_agentic.py to constructor-time DI
  ([`5e73b5e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5e73b5ef4193738e48e0486c716a856c4f1ca1fd))

- Skip 3 pre-existing failures blocking CI on main
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- TestBuildMemoryBlock — no key, first-wins, whitespace
  ([`a1f6d62`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a1f6d62237a9b9957ee51f16381d4159fb53a501))

- TestBuildUserPromptWithMemory — memory appended/absent
  ([`0e1d6a0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0e1d6a0ce583497dd9ec096be1f886d767646e90))

- TestDynamicPromptFetcher — dynamic refresh and fixed no-op paths
  ([`070b8f8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/070b8f851b51bfb52e872fb437f4b1a91f58b518))

- TestFixJsonEscapedCode — cover all 4 code paths
  ([`26cf9df`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/26cf9df1b334c831513132b8f02d5ff106cecf86))

- TestJsonTemplateGuard — JSON echoed as code is rejected
  ([`6aecbc3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6aecbc3e9dff7a92e58c70e6a6f3294b32f773ac))

- **01-01**: Add failing tests for manifest CLI subcommand group
  ([`7daafeb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7daafeb8c0dc3adcc76f9dd0eb077c09fffc523e))

- **05-02**: Add tests for resolve_plugin and WatchdogPluginOptions
  ([`c3566da`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c3566da2f1153b4af0d47921d2fbf75867e864e1))

- **05-03**: Update test mock paths from tools.* to gigaevo.* in CLI tests
  ([`d7322ed`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d7322ed8798b10f140311eef75e9a3b6cc3641d0))

- **adversarial**: Address review nits for sampling-mode PR
  ([`6d80573`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6d80573974ab5bbbe3962e25d358b8ceb98077aa))

- **aggregator**: Drop TestHeilbronConstructorYAML brittle hardcoded class
  ([`5830169`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5830169e99f0f24ebeadc0025bc40c09eec29f03))

- **archive_gate**: Hydra composition smoke test
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **archive_gate**: Pin cascade contract — SKIPPED → on_success deps skipped
  ([#229](https://github.com/KhrulkovV/gigaevo-core-internal/pull/229),
  [`7d0094e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7d0094eeff973b438ebdc0387175a92542d60089))

- **cli**: Add parametrized CLI smoke test (Phase 7)
  ([`167bd0e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/167bd0e533ce4bda517b9397bbf018851e0864a0))

- **engine**: Add SOTA invariant test suite for steady-state concurrency
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Cancellation + resume-after-kill invariants
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Concurrency stress + simulation suite (load × async patterns)
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Drop dead category banners in test_evolution_engine_complex
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Migrate test suite from _in_flight_sema to two-sema pair
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: ParentRefresher failure-mode resilience
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: Resume — both semaphores re-init at full capacity
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: T7 - Slot-leak chaos test for two-sema architecture
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **engine**: T8 - JIT DAG-refill behavioral test
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **events**: Drop LINEAGE_TREND from registry + seam expectations
  ([`af28d3c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/af28d3c87bc115b69f73b38e1ff014ae923da1b3))

- **evolution**: Relax strict-serial assumptions for two-sema pipeline depth
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **heilbron/adversarial-repro-v2**: Semantic proof for two-pass refresh + MutationContext cache
  ([`2df94ce`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2df94ceb92107452e185ec0e2dd1e8d5b5a0e721))

- **heilbron/d-tanh-no-lineage**: Ordering invariant for disable_lineage_on_improver gate
  ([`f154f0f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f154f0f0422bd836975974f7220184de28f032c8))

- **idea_bank**: Add direct tests for build_usage_payload Pydantic output
  ([`768eaf9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/768eaf9e9cd21f2e46c5ac952c5e4bbe38cde2dd))

- **ideas-tracker**: Update pipeline tests for new module structure
  ([`98cb5fa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/98cb5faad144334c99aef0b2f22ba5a2e6b1bb6b))

- **integration**: Add 6 end-to-end lifecycle state machine tests
  ([`d404d51`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d404d51130849bb79be0d5f446a69c96b0b83548))

- **integration**: Real-Redis end-to-end smoke for two-sema pipeline
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **integration**: Real-Redis smoke for JIT-refresh engine (Task 19C)
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **integration**: Tighten T10 invariants per code-review feedback
  ([#227](https://github.com/KhrulkovV/gigaevo-core-internal/pull/227),
  [`5c6057e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c6057e984b065d180839ed9168c98fc66793f97))

- **launch_generator**: Add integration test with real v2 manifest
  ([`99d5bcf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/99d5bcf1a84ea34e388f9be973a062f08e684d55))

- **lock**: Add 13 unit tests for extracted lock module
  ([`dbc009b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dbc009b73345de72aed1cfba41f52d87550c7e4a))

- **memory**: Add CSV→IdeaTracker integration tests
  ([`0a0adde`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0a0adde8a52cd593dae8dab34b30ec9f750392f1))

- **memory**: Add E2E tests for write_pipeline, memory search, rebuild
  ([`1cb4241`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1cb424113c6aac49db99abbc9c25f2c968f460c4))

- **memory**: Add E2E tests for write_pipeline.main(), IdeaTracker._run(), SelectorMemoryProvider,
  NoteSync.upsert_agentic
  ([`fc5f2d3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fc5f2d3d3ffc889c7ae13891ceddbf8738a3a0fa))

- **memory**: Add E2E tests — persistence, A-mem search path, write_pipeline loop
  ([`8045746`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/80457468ed1f90012a71c39bf4aa91fa944e1046))

- **memory**: Add streaming + JSON debug log tests for CardLoader and card_update_dedup
  ([`1a4cc70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1a4cc70f270d5f5603f55affc159215649e77e7a))

- **memory**: Add TDD regression tests for IdeaTracker factory + Hydra
  ([#209](https://github.com/KhrulkovV/gigaevo-core-internal/pull/209),
  [`00122c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/00122c0c62e127590d6942d5cd704561ebc8be27))

- **memory**: Clean up stale tests + add fixture-scoped LLM mocks for ideas_tracker
  ([#209](https://github.com/KhrulkovV/gigaevo-core-internal/pull/209),
  [`00122c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/00122c0c62e127590d6942d5cd704561ebc8be27))

- **pipeline**: RED — builder replaces LineageStage with filtered variant
  ([`80d8a9e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/80d8a9ec0c2a593b45241e5cdc5077eb83f50b38))


## v1.28.0 (2026-04-03)

### Bug Fixes

- Eliminate ~76k test warnings (0 remaining)
  ([`df637e6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/df637e67b2c8bf808288cc634655f35f3d8a984b))

- Flatten ideas_tracker aliases (list[dict]) to MemoryCard (list[str])
  ([`f6e620d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f6e620d5a0431265e8e25ad9b77ec1d36eb408a3))

- Lint + format pre-existing errors in experiment files
  ([`9241296`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9241296c3d3dc383d08f4f7ca6496c2382443f6b))

- Lint errors in ablation_v3_no_deep.py, update prereg_commit
  ([`a118bbf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a118bbffee6a6061a7dec0f8e924b0ecadce74cc))

- MemoryCard.aliases type list[str] → list[Any]
  ([`a31443f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a31443f851f5185d38b280eb767f4559975fcdf5))

### Code Style

- Ruff format
  ([`d05507d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d05507de30612d483f3967505b4f02f543cdc17b))

### Refactoring

- Remove all 27 type: ignore comments from codebase
  ([`91a175f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91a175fde1dc941cb6fe397a6d7adccb01a75a7b))

- Rename memory test files to describe what they test
  ([`68828e2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/68828e283de54d061b9eb89c5c546e06967dfef7))

### Testing

- Integration test for ideas_tracker dict aliases (Bug #2, PR #161)
  ([`ff54673`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ff54673d9ed918eb93e92856f45da9db0ad8e2a6))


## v1.27.0 (2026-04-02)

### Bug Fixes

- Format card_conversion.py
  ([`228b8f3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/228b8f3e68bcdede0a5fe5bdb4335542edf3f648))

- Lint import sorting in A_mem + GAM_root (pre-existing)
  ([`5e3baa3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5e3baa34d5dce321a24abaace314dd590b1ff57f))

- **memory**: Address chaos-hacker findings on public API
  ([`91aec06`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91aec061a2357e5bf5d0a9e57dfe8f981fd6f95a))

- **memory**: Correct concept_to_card return type annotation
  ([`78327d0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/78327d0ab04a0e71b8d77e3e980399eb959bdbe6))

### Features

- Add gigaevo.memory public API exports
  ([`7790b82`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7790b82ec79eb67e783bb5562f120dd376f76bc6))

### Refactoring

- Replace 50 print() with loguru in A_mem + GAM_root
  ([`59853df`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/59853dfef4348e5a68de55f8020ef47d65d07990))

- **memory**: Add future annotations, reduce hasattr/getattr usage
  ([`ae4e403`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ae4e4037ea512fb2ffd27e7c63a8a73701d41ae5))

- **memory**: Consolidate 20 test files into tests/memory/
  ([`e6f8480`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e6f8480b0619a6f57ef6476ee49f8455ad6d9741))


## v1.26.0 (2026-04-02)

### Features

- Dict → Pydantic migration complete — normalize_memory_card returns AnyCard
  ([`f2ea951`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f2ea95137a07a0eb3335566a4557bc8d6fcc3a5b))

### Refactoring

- Normalize_memory_card returns AnyCard (Pydantic models)
  ([`5926631`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5926631d646b08e998deac3c22fec7a0f0c0538d))

- Replace print() with loguru, remove sys.path hacks in ideas_tracker
  ([`8663644`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/866364480a59433456881ed89475d11fea220493))


## v1.25.0 (2026-04-02)

### Bug Fixes

- Add break condition for processing when no new ideas are present
  ([`a6a3a18`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a6a3a18916408611c4473dc332731ed680314909))

- Add break condition for processing when no new ideas are present
  ([`0e20d87`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0e20d8747fbab2d0fd474a0a6bf77508296c3f88))

- Changed cooccurrence threshold agressive scaling to fixed minimum
  ([`5c5c29b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c5c29b43752cd50fb2dbeaf670d2bdd4d135591))

- Changed cooccurrence threshold agressive scaling to fixed minimum
  ([`c8a7e5b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c8a7e5bb22a3908df1d7611a7107e3e3619449a0))

- Circular import in logger
  ([`55e3b1f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/55e3b1fe956b73cf731c63f2a4d70ed92f953429))

- Circular import in logger
  ([`522d28e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/522d28e99ca011775cdb2c677ca2f541767b258a))

- Clean up memory PR merge — lint, format, junk dirs, broken imports
  ([`21035ab`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/21035ab9003e25d810415bbbcb41064d129d65ec))

- Correct serialization of dict and lists in pd columns
  ([`f7a6bbe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f7a6bbeb669bd64b71eaa6d4662e0e9a658370dd))

- Correct serialization of dict and lists in pd columns
  ([`2dd2eba`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2dd2eba10f6af1ef41dc30995f67bb18164e0f27))

- Dead retry in _decide_card_action — parse_llm_card_decision returns None for garbage
  ([`80649dd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/80649dda8429daf5793540b4d4219fada39aa975))

- Eliminate RuntimeWarning in generate_mutations tests
  ([`ba4195d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ba4195dc07bd420cb7c570b7be2d33e2170f6d5b))

- Handle parent_ids as string in ideas_tracker
  ([`7601ed9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7601ed9b7083477f85f172b8c5b3277d4d177e09))

- Handle parent_ids as string in ideas_tracker
  ([`3148878`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3148878c2396141af69318f78bab8b3710d13c05))

- IncomingIdeas update logic fix
  ([`b9a1781`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b9a1781003c1f9874187aa09915f489de30e1c76))

- IncomingIdeas update logic fix
  ([`2947dda`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2947dda9a988c503c3a23c62ced28d3852d77441))

- Lint and format errors for CI (ruff check + ruff format)
  ([`b884547`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b884547912cb3b7327683b3d63251e729a5c8c6d))

- Phase 1 — 3 confirmed bugs fixed in memory system
  ([`5e9addd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5e9addd3cd02de1c3017ecde5cf3289771f4886c))

- Remove debug print
  ([`bf4a981`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bf4a981fa28d86a53240ca70ba86fd09225a3949))

- Remove debug print
  ([`cda32cb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cda32cb6069d76a8f8793c04f01df23f441a05bb))

- Remove short id separate storage and generation
  ([`d0fd1a6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d0fd1a6fb061488c17cb6d4829deb436303b41a3))

- Remove short id separate storage and generation
  ([`d1fdde8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d1fdde8aec4faafaa20bcab87dc7d0459b955ca4))

- Restore RedisRunConfig + fetch_evolution_dataframe re-export in tools/utils.py
  ([`30bc8ea`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/30bc8ea4c05040ba0cfc52e8ad87b7714d586e3b))

- Wrong key name fix
  ([`66cab68`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/66cab6851a8ebb222c1943fbc1c5f1a05950d2fa))

- Wrong key name fix
  ([`9cc9912`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9cc99122595b17cee965cd74105c66b5429b3ed8))

### Chores

- Removed unused prompts
  ([`5cff8f1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5cff8f18f6e443d3023529ad3f07f53df0c5abd7))

- Removed unused prompts
  ([`52852cc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/52852cc9cc2bc39c5a2376bd9f510dabb9c357bf))

- Update docstrings
  ([`3123cba`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3123cba56d437550da2ba1e148b1587823c7b5c5))

- Update docstrings
  ([`6a44189`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6a44189ad41c12e3c5d2653375632d01bc86f50c))

### Features

- Add best idea extraction based on top_k selection by fitness and delta fitness
  ([`04ab7c6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/04ab7c6e4e8652feb93863c44ab13a8ae1054e84))

- Add best idea extraction based on top_k selection by fitness and delta fitness
  ([`995929e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/995929e079c83aaa01e50b32765ce828af6eabf7))

- Add changes extraction to mutation agent
  ([`cd924c7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cd924c72084c29dda25ad9daa0af978adaddd91c))

- Add changes extraction to mutation agent
  ([`ad342ca`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ad342ca4d78ac24f79318da4ea5398c90e7d1f47))

- Add extended record card dataclass
  ([`8aeab4e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8aeab4ee27fe1493ff4a2d702fd65b9b270b81e3))

- Add extended record card dataclass
  ([`2d80776`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2d807769fd04e6f7e2421f0b97bf2ee617abf838))

- Add idea description rewriting logic
  ([`0201eb3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0201eb3b18d1b0d28c707d04f4145c2581089694))

- Add idea description rewriting logic
  ([`ea8ddd3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ea8ddd33a7e32dcd398a7082b09029b59602c8bb))

- Add idea origin analysis script and minor refactor ideas_tracker.py
  ([`ab184b8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ab184b886ba6aaee50fd646b061c8de07b2cd197))

- Add idea origin analysis script and minor refactor ideas_tracker.py
  ([`5a38c1d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5a38c1dc9035799430e8411bbc0531b0c704c45c))

- Add idea tracker
  ([`e9e911d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e9e911d7deadc0762e613d4ea8d536e74ba0d4be))

- Add idea tracker
  ([`eb9701b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/eb9701ba7bb9fbdd549da3dceadfa440d0027c91))

- Add logging for idea tracker
  ([`21b6f58`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/21b6f5859181ff37a394365712589ce632de8c66))

- Add logging for idea tracker
  ([`0563796`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/05637963cea3ec09dd93d7e4f93b2a711d1ee81c))

- Add ProgramCard, ConnectedIdea, AnyCard Pydantic models
  ([`edbdd1e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/edbdd1efe4ca4f69e8f5d8189fba59a1c2b8c406))

- Add update logic for extended record card
  ([`0bb5038`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0bb503807bff566dc2b738f2a5e2ce45a7d04fbe))

- Add update logic for extended record card
  ([`8529a30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8529a3025c3c3f6c6c1b19983eaca5ea7487a6c6))

- Csv loading to IdeaTracker
  ([`3cc0605`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3cc0605511c217b47e074546cea5bcb0fe2f895c))

- Csv loading to IdeaTracker
  ([`5b610ca`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5b610ca1cfab27a5eed430cbd8552257e520bc87))

- Experimental ml pipeline for impact estimation based on linear regression feature weights
  ([`ffaac97`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ffaac97458bd11bf9b8612745cbecef052319149))

- Experimental ml pipeline for impact estimation based on linear regression feature weights
  ([`bda2e79`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bda2e793e387267b6a5ecae6b606b41f82e53640))

- Implement idea enrichment with LLM-generated keywords and summaries
  ([`1d4f350`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1d4f350e3e958a9343f1b4b1b7e6d8a3e0704513))

- Implement idea enrichment with LLM-generated keywords and summaries
  ([`102cb74`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/102cb74ef3cf8051ce1ac9bc6a05e130eb47a2e2))

- Support for extended record card
  ([`a7cb492`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a7cb492e5f6365a5df37374a003b11b1ab1a7907))

- Support for extended record card
  ([`0549fe7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0549fe7587078ff0ddd4928fbf6e73f3909d86a2))

- Task description loading
  ([`bbfc57a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bbfc57a3eaed9b6e216c5e83f98fe6f2f9ed3b2e))

- Task description loading
  ([`2c3ff68`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2c3ff6864318783385d45195ea1cfe345dd79535))

- Update main logic to work with extended record card
  ([`b1ed1ac`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b1ed1ac75a079b26b42df7e319eb687756456f7d))

- Update main logic to work with extended record card
  ([`a98d6b1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a98d6b1e9e69d239b963c352cf94e4374f10bdc7))

### Refactoring

- Add Protocol types, fix mypy errors, CardDict alias
  ([`2f09116`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2f091166030a5a4893755c85e53e96a26b7e2431))

- Extract _note_fields_changed, remove stale comments and blank lines
  ([`c3fbede`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c3fbede31dec9941c9adbb1f89ae7e839cc5f3e4))

- Extract card_conversion.py + utils.py, add MemoryNoteProtocol typing
  ([`406f2cd`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/406f2cdeb29f04e6fad2042264aab727d7997a1f))

- Extract DEFAULT_MODEL_NAME constant, remove ad-hoc string
  ([`0f67bc1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0f67bc13317979792c50573b998d70acb397a3ad))

- Extract more pure functions + memory_write_config.py
  ([`7e1bdbe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7e1bdbe383529e1ab27a4b12c04b494c2adc7ffa))

- Phase 2 — import cleanup, context manager, remove __del__
  ([`dd31c4f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dd31c4fed46de6eead0a8ba14507e9f755846b55))

- Phase 3 — extract _ConceptApiClient + utilities
  ([`8dacf30`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8dacf30cd93b9c59e59dc1d4ada5aa7e7fa3d7a5))

- Record card extended minor refactor
  ([`bbda084`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bbda084fab66b071bb5a99c5a03450fa94319622))

- Record card extended minor refactor
  ([`b5eac06`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b5eac06a2c47e70e29f09cbf67c5489bb77dcd61))

- Remove debug code
  ([`5a9f8ec`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5a9f8ecaae4b273549e1cec62f98c88379366e27))

- Remove debug code
  ([`117a325`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/117a3253574f56133ed0e88321e1494c0e4f5846))

- Rename test files and classes to professional naming
  ([`6ae0134`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6ae0134744a7b42b112fbee7e3c41d8e2cd2343a))

- Replace all print() with loguru logger across memory module
  ([`5091f37`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5091f37410029b881312d06da0aa55794984131a))

- Replace ML impact pipeline with origin analysis computation and improve docstring clarity
  ([`bcb2ff7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bcb2ff768d014ee7c55972eb37b8510f07511f1f))

- Replace ML impact pipeline with origin analysis computation and improve docstring clarity
  ([`3f57d5f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3f57d5f5ea882437bd0ee189a47d9eb94f392567))

### Testing

- Chaos-hacker bug exposure tests (16 tests)
  ([`1e62411`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1e62411879716283514f868fcf35650fa91604d7))

- Cycle 10 (final) — API search, LLM synthesis, close() (21 tests)
  ([`2d63100`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2d6310088d0e911a804aa64bfe8bf49740688cac))

- Cycle 11 — fake agentic memory infrastructure + 24 tests
  ([`72c49e5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/72c49e531948a79f72d14895aab5257178e34b4b))

- Cycle 12 — fake Chroma/GAM + full dedup pipeline (15 tests)
  ([`d94a674`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d94a67478707ecddc3f27768c82ec84b5f4ec8c9))

- Cycle 13 — 7 realistic E2E scenarios + 2 unpatched real-memory tests
  ([`282f7f6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/282f7f6c4f26caef687f0b9595592ec1e28b26e1))

- Cycle 2 — API client, dedup decision, truncate (28 tests)
  ([`77aa25d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/77aa25dcf635b5f7b506ac9a6105ece3e1dcd08d))

- Cycle 3 — deeper AmemGamMemory internals (21 tests)
  ([`543b2e2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/543b2e234a69ed378cca85e636ded592efc9ab6a))

- Cycle 4 — integration tests + chaos-hacker regression fixes (25 tests)
  ([`4b92198`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4b921986d92b34c7252e4567fca391edca4b5eb1))

- Cycle 5 — mutation operator memory flow, sync_from_api, API body checks (17 tests)
  ([`fcbc352`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fcbc35270c31ace767c043873323de834b578f20))

- Cycle 6 — 8 e2e scenarios + data_components (64 new tests)
  ([`1040f02`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1040f027e94310d0fa23e280747ae64c009988da))

- Cycle 7 — contract tests + engine interaction (34 new tests)
  ([`616f88c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/616f88cb042e5f1cf9bcfd1ff94eb3657854330d))

- Cycle 8 — full-loop evolution with memory (11 E2E tests)
  ([`d7ef94d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d7ef94da085711f4ef6cfb7c22c9bc3169fe9980))

- Cycle 9 — LLMMutationOperator real constructor + memory (14 tests)
  ([`9035068`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/903506877ddd0768dc50896fa63a424a9f5a0534))

- P0 exhaustive tests for memory module core (211 tests)
  ([`572c28a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/572c28a3dc8c8e232e061f704c81e6eae2d1f89a))

- P1 dedup edge cases + OpenAI inference tests (100 tests)
  ([`9945c0e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9945c0e499f0243a78f029bad47f26c78da5f8cb))

- P2 memory_write_example edge cases (22 tests)
  ([`70375bb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/70375bb8c9b3e0cd8b040c4550bd4cdc7e8be924))


## v1.24.1 (2026-04-01)

### Bug Fixes

- Remove last 4 dead .claude/rules/ references from CLAUDE.md
  ([`9caf0c3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9caf0c35898ea169bf269f24520398a15bc4ba78))

### Chores

- Remove GitNexus from CLAUDE.md, skills, and gitignore
  ([`b9f90ed`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b9f90ed5a6eea049e5c17ad6e4ff13a8d6a170e6))

### Documentation

- Add Quick Start sections with runnable commands to all feature docs
  ([`4d9a809`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4d9a809fa29e2b1ce4d0eeb86760542609dffd9c))

### Refactoring

- Rename scheduling/lpt_ridge → lpt_chain, clarify scope
  ([`10cf394`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/10cf394fba7d8ac8ac78e728665989c67e17f941))


## v1.24.0 (2026-04-01)

### Highlights

This release focuses on **performance infrastructure**, **experiment tooling maturity**, and **repository hygiene**. Two experiments were completed (hover/steady-state-v2: POSITIVE, hover/map-elites-topology: NULL), and the framework gained production-grade load balancing, scheduling, and monitoring.

### New Features

- **Steady-state evolution engine** — continuous mutation/evaluation interleaving that eliminates the generational barrier. Two async loops (producer + consumer) with backpressure via `asyncio.Semaphore(max_in_flight)`. Opt-in: `evolution=steady_state`. Expected throughput: ~8-9x improvement over step-wise generations.

- **LPT scheduling for DAG evaluation** (#136) — longest-processing-time-first scheduling assigns expensive programs to evaluation slots first, reducing tail latency. Discrete-event simulation benchmarks in `tools/benchmarks/`.

- **LLM load balancer** (`llm=balanced`) — Redis-coordinated endpoint pool with least-connections routing. Mutation servers shared across all runs via Redis DB 15. Replaces manual `llm_base_url` per-run configuration.

- **LiteLLM proxy integration** — `bash tools/litellm.sh` auto-generates config from `experiments/infrastructure.yaml` and starts a LiteLLM proxy for chain server load balancing. All chain requests route through `INTERNAL_IP:4000`.

- **Chain feature extraction** — `ChainFeatureExtractor` computes structural behavior coordinates (DAG depth, retrieval count, step count) from real chain programs for MAP-Elites behavioral characterization.

- **Experiment diagnostics** — `/experiment-diagnose` skill: automated failure analysis for running experiments. Checks Redis health, PID liveness, log errors, and Hydra config overrides.

- **Experiment restart** — `/experiment-restart` skill: kill all processes, flush Redis, and re-launch cleanly.

- **Throughput monitoring** — `tools/throughput_plot.py` and 6-panel dashboard in watchdog: mutation rate, eval throughput, fitness distributions, validity panels. Posted hourly to experiment PRs.

- **Fitness vs wall-clock time** — `tools/fitness_vs_time.py` plots fitness trajectories against real time instead of generation number.

- **Prompt co-evolution** — user prompt co-evolution alongside system prompts (`prompt_fetcher=coevolved`).

### Bug Fixes

- **120s read timeout killed 96% of chain evaluations** — removed read timeout (`timeout=None`, keep `connect=30s`) to allow long-running chains under load.

- **CancelledError orphans** — `except Exception` didn't catch `BaseException` in steady-state engine, leaving programs persisted but IDs lost. Fixed with `persisted_id` sentinel + `except BaseException`.

- **Mutation LLM double-escaping** — LLMs using `with_structured_output()` sometimes double-escape quotes in code fields. Fixed by `_fix_double_escaped_quotes()` in mutation agent.

- **Frontier metric recomputation** — when NO_CACHE stages re-evaluate programs, frontier is now recomputed correctly using `clear_series()` + full rewrite instead of appending stale values.

- **TOCTOU races in SteadyStateEngine** — scoped drain + TOCTOU-safe `ingest_batch`, `add_elite` with optimistic locking and WatchError retry.

- **Ghost program detection** — mirrors parent engine's `_await_idle()` logic to clean up orphaned program IDs.

- **Proxy bypass** — added mutation server IPs to `NO_PROXY` to prevent Squid proxy from blocking LLM calls.

### Experiments

| Experiment | Result | PR |
|---|---|---|
| hover/steady-state-v2 | **POSITIVE** — continuous interleaving improves throughput | #138 |
| hover/map-elites-topology | **NULL** — 3D structural BC (dag_depth, n_deep_retrieval, n_steps) did not improve fitness | #142 |

### Repository Cleanup

- **Removed leaked vartodd/circuit_evolve code** — problems, configs, custom/, gf2lib/, npy/, launch scripts (12,800+ lines deleted)
- **Removed experiment runtime artifacts** — PNGs, pids.txt, cfg_run_*.txt from all completed experiments
- **Consolidated tools hierarchy** — experiment-specific scripts (archive, preflight, protocol gates) now live in `tools/experiment/`; general tools in `tools/`
- **Removed all hardcoded paths** — skills, agents, tools, and docs now use `$PROJ` (git root) and `$GIGAEVO_PYTHON` (env var) instead of `/workspace-SR008.fs2/...` or `~/...`
- **Fixed .gitignore contradictions** — `.claude/` and `CLAUDE.md` were tracked but gitignored
- **Cleaned root directory** — moved `benchmarks/` → `tools/benchmarks/`, `demos/` → `docs/demos/`
- **Rewrote Redis data model docs** — complete key namespace reference with all metric tags, archive persistence, iteration vs generation glossary

### Documentation

- **CLAUDE.md** — added tools index, skills table (12 skills), agents table (9 agents), `@tools/README.md` include for Redis data model
- **tools/README.md** — structured tool index with categories (general, experiment lifecycle, infrastructure, benchmarking, scaffolding), accurate Redis appendix
- **Removed dead references** — `.claude/rules/*.md` files that never existed on main, redirect stub `docs/redis_schema.md`

### Testing

- 56+ new tests: race conditions, streaming, failure modes, mutation-killing, TOCTOU guards, NaN handling
- Removed deprecated test classes (TestSafetyMechanismBreakage, TestEngineGenerationTimeout)
- Full suite: ~3500 tests, all passing

## v1.23.0 (2026-03-15)

### Bug Fixes

- **bugs**: Round-2 — migration KeyError on None island + DAG empty-nodes crash
  ([`54810b0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/54810b0d9cc305dba83dd0cf65a3b0d03468e427))

- **bugs**: Round-4 — 5 junior-researcher attack surface bugs
  ([`073cc33`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/073cc333f342b0a54a41f4491b33729a25d2fc5e))

- **bugs**: Round-5 — H1 sentinel bypass + TOCTOU dag_runner + H2-H4 guard tests
  ([`d039cf1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d039cf13d9d3e3533ec9768358487349075d8cc0))

- **tests**: Update test_evolution_engine.py for get_all_by_status migration
  ([`a33604b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a33604bb68ed6ded20757162057715ff36b69980))

### Chores

- **generalization**: Add launch script and run_status.sh
  ([`61e76af`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/61e76af44632004b1bcf5ad39ba50cedbe540370))

- **generalization**: Add launch script and run_status.sh
  ([`96bbb42`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/96bbb42765c1b8e0403fc1589873c1121b40bd15))

- **generalization**: Add test eval script, PR description, gitignore indexes
  ([`6405a22`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6405a2260574edc53b912f4a3d36c84b7bcc25a0))

- **generalization**: Backfill pre-registration commit hash in 03_plan.md
  ([`3bd7fea`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3bd7fea204b0b9b31e04570ed7f20c73449d38b6))

- **generalization**: Gen-1 smoke check — all 4 runs alive, split bias OK
  ([`17b25db`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/17b25dba677d8c7d9ae185c1644c850256ec9505))

- **generalization**: Record binding prompt review sign-off in 03_plan.md
  ([`6a0d2e6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6a0d2e6cdc6e176678fb04c59281df6ec345cc24))

### Documentation

- **memory**: Chaos-hacker round-5 findings summary
  ([`4c742fc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4c742fcea386acbda921d1594f2611f48bc495db))

- **memory**: Restructure Claude memory + propagate gen-count fix + add closeout step
  ([`aca5f0d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/aca5f0d0b1210332cf32d142bd4c9772248776de))

### Features

- **generalization**: Implement static_holdout_f1 problem + generalization prompts
  ([`1dbb05c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1dbb05cf18b392caf5c6df05923a1c2693b3ae7f))

### Refactoring

- **tests**: Move round2/round3 tests to semantic locations
  ([`3b3117e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3b3117e7b0f76e1bf0cdcfcccc95bb05d97ec1d0))

### Testing

- **integration**: 21 new integration tests — DAG ordering + engine edge cases
  ([`1bb5235`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1bb52359b7ae0b68266845fb480244304f12f286))

- **round3**: Regression tests for Bug A and B fixes
  ([`f778ad7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f778ad784c1088ec984995230d5f43f7106af9ee))

- **security**: Fix safe_mode bypass + add regression tests from audit
  ([`ca2d4cf`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ca2d4cf335708db9dcb98cb541138f1dbca38cf6))


## v1.22.1 (2026-03-14)

### Bug Fixes

- **results_report**: Remove stray ESC character (U+001B)
  ([`e2b01fa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e2b01fa6de1b0c6334ca7e742bb2a5e49179fea7))

- **status**: Use run_state Redis key for generation count
  ([`e7648ab`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e7648ab00c903131902a2e4751201499b8818b30))

### Chores

- **gemini_mutation**: Pre-merge cleanup — environment freeze + PR description
  ([`a36f29d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a36f29d38fda884c4cf02b5ab5ded8c9a2fd1974))

### Documentation

- **hotpotqa**: Add LaTeX results report for paper
  ([`02adea3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/02adea39dc69f82ce4df36d836313c16a9f544b1))

- **hotpotqa**: Make results_report.tex self-contained compilable document
  ([`bf0ff3f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bf0ff3f23bf8c4e454494551279549f51bf3c958))


## v1.22.0 (2026-03-13)

### Bug Fixes

- **resume**: Make redis.resume produce a contiguous run
  ([`07091fb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/07091fbeb0ec5760d66e7892b600bf68343f1d91))

### Features

- **gemini_mutation**: Pre-register experiment — Gemini-3-Flash as mutation LLM
  ([`0f42851`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0f42851217e9888c4009087ce9d59d451ab5ce83))


## v1.21.0 (2026-03-12)

### Bug Fixes

- **build_colbert_index**: Cap num_partitions=32768, kmeans_niters=4 for tractable CPU k-means
  ([`1b605b3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1b605b3bd4a064ecd7db1b1e33fd9eca59d3fd22))

- **colbert**: Replace faiss GPU k-means with PyTorch batched k-means
  ([`765e8aa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/765e8aadc02ed46c59cc13dc6ecb2f9405012d20))

- **colbert**: Simplify build script — patch applied directly to colbert source
  ([`f86d3a4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f86d3a4b4feb7481a72c960027e40375392c4f31))

- **colbert_feedback**: Export HOTPOTQA_COLBERT_SERVER_URL in run_test_eval.sh
  ([`ac1e423`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ac1e4230354c502568b8d616afd2b71ae22f3127))

### Chores

- Fill pre-registration commit hash and PR number in 03_plan.md
  ([`454f817`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/454f81725e8abfd71fc87a80c739fbb5e25f9e4a))

### Documentation

- **colbert_feedback**: Amendment 5 — gap investigation results
  ([`efdf087`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/efdf0877972798b575f4b130b94bf96314c1b0e8))

- **colbert_feedback**: Record index build completion in 03_plan.md
  ([`a8f4e9a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a8f4e9a94495f4fca2d64125557bb291bd1dfd00))

### Features

- **chains/hotpotqa**: Add ColBERT+rich-feedback experiment (colbert_feedback, Phase 3)
  ([`20c8314`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/20c831427540e0d12ed0ac3e64dac0266ce7f06f))

- **colbert_feedback**: Add ColBERT search server + update launch/validate/plan
  ([`d7683d4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d7683d4cbd8cffe614bec69a29382932d2b78bd7))

- **colbert_feedback**: Watchdog + benchmark server-url support
  ([`1272c1e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1272c1e42504d3cc9f842733a55535f9fffadf60))


## v1.20.0 (2026-03-09)

### Features

- **chains**: Hotpotqa: add Retriever class and colbertv2 retriever
  ([`b681195`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b681195f49bfc878622e0c94ba227669fd549798))


## v1.19.0 (2026-03-09)

### Chores

- Add cold_start entry to INDEX.md + create experiment branch
  ([`b0b47af`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b0b47affd7feb83a81be76e1007e005a10e5a129))

- Fill PIDs into run_status.sh — T1=3812756 T2=3812757 T3=3812758 T4=3812759 watchdog=3813084
  ([`cef1b24`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cef1b2489f4007eda485bbb1c9a15c57ab5a9229))

- Launch.sh, run_watchdog.py, run_status.sh for cold_start experiment
  ([`53bbb1f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/53bbb1f9e0d34c6550c04d4d826b34706a35f642))

### Features

- Add baseline initial_programs to static_f1_600 for cold-start support
  ([`f5adb9f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f5adb9f3868bc4ba59ae5f1275ffd7d675a0931d))


## v1.18.3 (2026-03-08)


## v1.18.2 (2026-03-08)

### Bug Fixes

- Watchdog gen count — use log file instead of Redis s field
  ([`1d229e4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1d229e445e2338855dd796a929dfe5c3eefeee7f))

### Chores

- Backfill pre-reg commit hash + add crossover entry to INDEX.md
  ([`2733e86`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2733e8606438ea87daf68663933de19f9819b93e))

- Fill PIDs into run_status.sh — P=3660148 Q=3660149 R=3660150 S=3660151 watchdog=3660461
  ([`d52fe31`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d52fe31a1b8b0c0a8e577af0f74dfa50ec6335f8))

- Launch.sh, run_status.sh, run_watchdog.py for crossover experiment
  ([`e2c3629`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e2c3629d4fee4452503e38e59081a15e91d3fa35))


## v1.18.1 (2026-03-07)

### Bug Fixes

- 12 infra correctness fixes from codebase audit
  ([`ef89ba9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ef89ba95a54d4e2905486170a92face53bfdd33e))

- Check_experiment_complete.sh SIGPIPE bug + environment_freeze.txt
  ([`815b47e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/815b47e6c7c1d80198da35459f16e339bf1f1b46))

- Extend prompts_dir to all pipeline YAMLs + docstring accuracy pass
  ([`939ec7d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/939ec7de27241aab1409907c36b897982c4cfcac))

- Gen10_test_eval.py val_em gap correct for F1 runs
  ([`9c9e65f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9c9e65ffb1441edce8343852bf291109e582a7a9))

- Move analyze_test_results.py to push experiment tools dir
  ([`e6367be`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e6367bed53ad2ff0ecf3cbfe50207aad5064f43c))

- Pin push run_test_eval.sh sha256 in 03_plan.md (was val_gap hash)
  ([`7763287`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/77632878015a27cbb4d85ee92b0c9711380a75df))

- Propagate known bugs to templates and docs to prevent recurrence
  ([`b545282`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b545282dda3053deb19f51cffb57461cd0091917))

- Raise chain LLM HTTP timeout 120s→600s + hard reset all runs (Amendment 4)
  ([`c0186a8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c0186a8ff3250cd6b7af764b0f3b29e67dd5c1b7))

- Remove stale failures[:10] cap from docstrings and pipeline comments
  ([`ac3c07d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ac3c07dd0cd6cb80404e4339a69168cd0b21b34c))

- Tighten APPROVED grep + correct agent memories for Phase 5 readiness
  ([`3ee5854`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3ee5854af29cae70e037c18c7d988790cf516863))

- Update PR_DESCRIPTION.md template — val EM → val fitness
  ([`e99f891`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e99f89146fd343f278618936ac35ad9f202834e0))

- Watchdog PROJ path (3→4 parents) + stale Run D config
  ([`a3dae8f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a3dae8f4daa79ee00921f6830a0d77cae0c85307))

### Chores

- Add run_status.sh template for push experiment monitoring
  ([`6a1e86c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6a1e86cf6cda39b47900dac0e7cdb8e4b7bb12fd))

- Infra improvements while runs are live
  ([`466db87`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/466db87f477598b034cf389b07c1fa4b0916cc32))

- Launch.sh for push experiment + CONTEXT.md updates
  ([`c2da4a7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c2da4a752ee7c45e3e26cd8c18878ec7cc20a215))

- Pre-fill 05_results.md skeleton + analysis script + INDEX.md entry
  ([`a8e288b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a8e288b8f0fc79f4b29e5ee1fa12d853a4a0cd50))

- Replace Run D EM+NLP+600 → F1+NLP+600 (Amendment 3)
  ([`58ec1fa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/58ec1fa3bc64723600076031a73ba252c9a0134c))

- Update INDEX.md and CONTEXT.md naming consistency
  ([`05bce0a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/05bce0adceee1a4ce0a39d559b79f2fed3e330db))

- Watchdog + run_status.sh for push experiment
  ([`81807ea`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/81807ea3faf4a4e135c9627d70ac60c95fda1d5b))

### Documentation

- Hotpotqa_asi.yaml is required for ALL hotpotqa variants, not just static_a/ra
  ([`856125c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/856125c759c5b63264df2c2c4ae0f2cbe2b0635f))

- Update all experiments/<name>/ → experiments/<task>/<name>/
  ([`1b67951`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1b679516e7f5cc006fb9396adf6820d8a4df06f5))


## v1.18.0 (2026-03-06)

### Bug Fixes

- Wire stage_timeout through DefaultPipelineBuilder + validation speedup
  ([`55772ba`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/55772bae28d491ee60d4247e2747c82612b9f522))

### Chores

- Update agent memories (push experiment + path fixes)
  ([`6f50092`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6f5009271a863c2d17f8c8cf68517ea37273e470))

### Documentation

- Fix INDEX.md — hotpotqa_thinking test EM ~60% not 62.3%
  ([`e8407d5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e8407d584a2f13be60b47f2d09f7f8538eb614b0))

- Set pre-registration commit hash in 03_plan.md (push experiment)
  ([`f47847f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f47847f358b44b120d927e94ecb9e597d28e1fe5))

- Update INDEX.md — drop pre-protocol exps, close out nlp_prompts + val_gap
  ([`7ed9ba9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7ed9ba9266f3db02da2bc193b6f9fe76a8613870))

### Features

- Pre-registration 03_plan.md + static_f1_600 problem directory (push experiment)
  ([`9173c10`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9173c10da22b3e581e0be45db8383a73b37a9fa9))


## v1.17.0 (2026-03-06)

### Bug Fixes

- Amendment 1 review fixes — F1 objective, EM=0 criterion, rationale, Gate E
  ([`866f106`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/866f10653986696881138f7ff1a020a2141ddd1d))

- Distinguish timeouts from generic failures in stage logs and status monitoring
  ([`2228dd8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2228dd8a635f99921b04580da29cff4bec1aca0c))

- Launch.sh preflight loops use CHAIN_URL_F (not removed CHAIN_URL_P)
  ([`bb79e6f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bb79e6ff92f6513485a02df46db897248d22b291))

- Replace dry_run=true with --cfg job in launch.sh; update CLAUDE.md
  ([`13e968a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/13e968a1a6039a30c49dbc28d3af68374f0f506e))

- Status.py gen count bug + add Redis schema doc + run_status.sh
  ([`b85e388`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b85e3886058ca30bd94a9daaac7d1d0dbc592a95))

### Chores

- Record Amendment 1 commit hash in 03_plan.md (866f106)
  ([`2f64837`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2f648375b089fedc2d197c1b4ee4c56f2802b933))

- Update PIDs in run_watchdog.py — launch 2026-03-05 12:21 UTC
  ([`acec7c1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/acec7c1e73c83143f9ef5b41df07124e3139c813))

### Documentation

- Fill pre-registration commit hash in 03_plan.md
  ([`77f3ef6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/77f3ef66c6b1b5734818ebe1ab7f69cced1b4dea))

- Move task_hotpotqa.md → experiments/hotpotqa/CONTEXT.md + CLAUDE.md lookup table
  ([`e920581`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e920581c442b6244e1437a55a627174d82da9dec))

- Split task-specific content out of CLAUDE.md into task_hotpotqa.md
  ([`91410b3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91410b3fed5afca2e8c7d689071e65e01e64fc44))

- Update 04_launch.md for dry_run removal and crontab unavailability
  ([`7757d6d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7757d6d022154d6066782f2e90d49a1528ca8edc))

### Features

- Add static_600 and static_r600 problem directories for val_gap experiment
  ([`6dfa6d9`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6dfa6d9ae9a2f5b29c78f4ec7bd81ef5c607d91c))

- Amendment 1 — replace Run P with Run F (fixed-300, F1 fitness)
  ([`5c4370b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5c4370b84d345c043736a71c334608ec160a5d06))

- Gap_analysis.py + lineage.py + eval_checkpoint.py + README onboarding fixes
  ([`45faa45`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/45faa451494aa0bbd8050ac22d9cdaa377462d64))

- Launch script and watchdog for hotpotqa_val_gap experiment
  ([`fa6a14d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fa6a14d3ad0e3adc86ac6ced2eb6269d3bf1affa))

### Refactoring

- Nest hotpotqa experiments under experiments/hotpotqa/ project dir
  ([`25652c0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/25652c024c55f98e7ae7f1ce3d180d1022743cd0))


## v1.16.2 (2026-03-05)

### Bug Fixes

- Shutdown worker pool before event loop closes on Ctrl+C
  ([`6bbbd1e`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6bbbd1ea6260c3857a4dacd4e7f2deedbc4cafce))


## v1.16.1 (2026-03-05)

### Bug Fixes

- Remove hardcoded ~ paths from shared scripts
  ([`407f8d3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/407f8d3bad585c066a9bacaa5846551596885180))

- Replace hardcoded gh path with command -v gh in tools
  ([`d30bf8a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d30bf8add16a56dbecc874740175df48dfbdeede))


## v1.16.0 (2026-03-05)


## v1.15.1 (2026-03-04)

### Bug Fixes

- Correct @package directive in prompts/default.yaml
  ([`40954d0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/40954d05ad2d0c8333d38f9374b61ff69a9b1755))


## v1.15.0 (2026-03-02)

### Bug Fixes

- **chains**: Address reviewer fixes
  ([#68](https://github.com/KhrulkovV/gigaevo-core-internal/pull/68),
  [`4c84ba5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4c84ba526eeb1630b7c2bd636cd5fdf820e63766))

### Chores

- Update coverage badge to 86% [skip ci]
  ([`e2c0813`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e2c0813cda7627b2f2cd53dcb6ec2c5a38e37cac))

- Update coverage badge to 87% [skip ci]
  ([`cbe7155`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/cbe71559bda5c385902981044512d2fd126f87ff))

- Update coverage badge to 87% [skip ci]
  ([`1dd894d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1dd894d829cdc7217613d5dd7ce66b1c5b277333))

### Documentation

- Fix changelog link — point README to root CHANGELOG.md
  ([`43964dc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/43964dc704ba657a79b0eea26575090f5be1f236))

- Update README test structure and coverage badge to 85%
  ([`ba5ad09`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ba5ad09340b144c6febabfdad1cddac77ac445f6))

### Features

- **chains**: Speed-up chain_runner, add aime,hotpotqa_full,hotpotqa_qa,hover,ifbench,papillon chain
  problems. ([#68](https://github.com/KhrulkovV/gigaevo-core-internal/pull/68),
  [`4c84ba5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4c84ba526eeb1630b7c2bd636cd5fdf820e63766))

- **chains**: Speed-up chain_runner, add new chains problems
  ([#68](https://github.com/KhrulkovV/gigaevo-core-internal/pull/68),
  [`4c84ba5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4c84ba526eeb1630b7c2bd636cd5fdf820e63766))

### Refactoring

- Rename test files from _adversarial/_extended to _edge_cases
  ([`60dc53b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/60dc53b9a3ff4de8855e1bb60ebd2b6ca2db3e2a))

### Testing

- Comprehensive test coverage expansion with audit hardening
  ([`5fb12ac`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5fb12ac81c5f504f03da6ca143e93e38e2df0e67))

- Deep audit hardening with 207 new mutation-analysis tests
  ([`c101646`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c101646b20150de497db1c914a6bf61b0da85f46))


## v1.14.2 (2026-02-25)

### Bug Fixes

- **prompts**: Download bug
  ([`9e6da70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9e6da700c8d1d5990d2875ea1af540ac039d1f5a))

- **prompts**: Fix broken import
  ([`0c7e63f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/0c7e63f37e562b6582a8d2e0bd5319335aafd953))

### Chores

- Update coverage badge to 78% [skip ci]
  ([`82b4f00`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/82b4f00dc19b329bed7a6b250cb2747c1e377fbc))

### Testing

- Add extended test suites for coverage-gap modules
  ([`c2bf999`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c2bf999c14c1d7e81df32dde8f85101dd45667c0))


## v1.14.1 (2026-02-25)

### Bug Fixes

- **prompts**: Remove single-step exp.; add full chains evolution
  ([#63](https://github.com/KhrulkovV/gigaevo-core-internal/pull/63),
  [`8a9ec44`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8a9ec44c6c9cd0cc35c0d34c3ff8a5a7d83527a9))

- **prompts**: Removed wrong directories
  ([#63](https://github.com/KhrulkovV/gigaevo-core-internal/pull/63),
  [`8a9ec44`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8a9ec44c6c9cd0cc35c0d34c3ff8a5a7d83527a9))


## v1.14.0 (2026-02-25)

### Bug Fixes

- Timeout polish for optuna stage
  ([`b9d914a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b9d914a783791b202f815d53d2578ebce5a664c0))

### Features

- Add time-budget deadline to Optuna trial loop
  ([`6c98665`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6c98665a959edc9fc99c8a85c9c90b9bfe649f15))


## v1.13.0 (2026-02-25)

### Bug Fixes

- **ci**: Sync release job with latest origin/main before semantic-release
  ([`34efe54`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/34efe5499262c2e6716d561290fcd6d6f91da2b2))

### Chores

- Update coverage badge to 77% [skip ci]
  ([`8824566`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8824566e3fde3e840ee964e62e24dccd7907a14d))

### Features

- Filter optimization stage errors from mutation/LLM prompts
  ([`6107559`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6107559ec55c895e63c5b558438458b26b574d28))

- **ci**: Add self-updating coverage badge to README
  ([`61a9ef1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/61a9ef11f67563e1303ea58b34c88cc2cff83b37))


## v1.12.0 (2026-02-24)

### Bug Fixes

- Add cwd to exec runner
  ([`003eddb`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/003eddbe1c20da493afb52704226c6fb0b69efbb))

- Add metrics storage in redis
  ([`29d4bb7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/29d4bb7dbc4a3067630682b2bcf85723b3e0f11e))

- Add missing file
  ([`5b0e661`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/5b0e6611a4ed6383b0e5225685bad62c0fffc718))

- Bug fix for zero fitnesses
  ([`e30d4b7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e30d4b78490e357aa35caa39f5e01d3e23158905))

- Close subprocess transports to prevent "Event loop is closed" warnings
  ([`075cd4c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/075cd4c77661e352522f5b462121bd6c19c66e56))

- Cloudpickle 'register_pickle_by_value' for correct root imports handling
  ([`9b70499`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/9b7049971c9f44bf667f4b70cafdf70623bfc4ed))

- Cma deps
  ([`e3cc526`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e3cc5269b89cb8fc7e7f5d0e3fe06133a5120de6))

- Comprehensive wizard
  ([`51eb90a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/51eb90ab1ea5ba165789b65044ade14a214b4901))

- Fix bug in caching behavior
  ([`362dcb6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/362dcb6b45ac7769c38505160d7b6201e4b0ddce))

- Fix bug in dag cache handling logic
  ([`61e98ad`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/61e98ad53870dd3372798fa0549630d6ba683ad4))

- Fix faulty caching for programs with optional input
  ([`ca55919`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ca559198da4609d0fa08cbaadd65723c6c9418a2))

- Fix missing traceback from lineage
  ([`28eeff4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/28eeff4ba99be7c961082b056eca01ee93a358cf))

- Fixed exec runner to handle project directory
  ([`568b51d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/568b51d2505db3b9fcd78c2626ce857e6c000a5d))

- Grammar errors
  ([`a0b779b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a0b779b0be141d00cfbb551f0d2a1818e6e7cb38))

- Logging
  ([`7597117`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7597117b47b51ad0252f96f947e11c3778dee49f))

- Minor boundary fixes
  ([`a0dca3f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a0dca3fbee8446924b68ade7587966a4399fe5dd))

- Minor optuna polish and done
  ([`87e02f0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/87e02f0a634f96e790bf8e29ce1b8ecc6dade034))

- Move exec runner; speed up python execution via worker pool
  ([`6bce277`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6bce277e0a507055966792c3bfe1169b554bf3da))

- Optuna stage patching
  ([`8bfba7f`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8bfba7f8dd686425832fb5fdcd5a962dff203915))

- Optuna stage polish
  ([`7c628e7`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7c628e7c749bd40a525d121993683d5f6a17a268))

- Pickle to cloudpickle
  ([`7c47ac5`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7c47ac5317fcdf870664c61ef31a4fec6aa0eb33))

- Remove indices from constants, fix fitness descriptions
  ([`c5e2649`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/c5e26492077a2a4e23203db92042d9a1d6ee54c6))

- Remove unnecessary wizard configs
  ([`997f094`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/997f094f3638ddbc6feaa7619b9ba4f8ceaf40b8))

- Replace deprecated class Config with model_config = ConfigDict()
  ([`10a8fa1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/10a8fa1de25a2104597cbba0189dc510bda79635))

- Restore Optuna prompt constraints and remove reasoning max_length
  ([`98771fa`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/98771fa20d597b7e7703ff7c84d8350dfe24ea43))

- Undo default endpoint
  ([`4b4adc1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4b4adc1917495a19df65853e6d9ceb7bfb533e6f))

- Update three alphaevolve problems
  ([#51](https://github.com/KhrulkovV/gigaevo-core-internal/pull/51),
  [`aaaea70`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/aaaea701fa019ebd7dc529bf89f53a5288551aa3))

- Windows compatibility
  ([`b96caa2`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b96caa2624b8a53594bbde34e30fdf7d8b0596aa))

- **ci**: Fix semantic-release not updating CHANGELOG
  ([`a7da06a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a7da06ae195e21da7690791776970d500a74c8ec))

- **ci**: Remove orphaned v1.12.0 tag to unblock semantic-release
  ([`f60e8b6`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f60e8b6103430bf226bff3481912a8f21f138f8f))

- **ci**: Use startsWith instead of contains for release skip filter
  ([`bfb0e2a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bfb0e2aa4ac342164807a8f13b45287c3afbab73))

- **prompt**: Remove .nltk artifacts, add dependencies, upd. .gitignore
  ([`53cc178`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/53cc178416d469ea69c4d966ae2276bc801c1588))

- **prompt**: Remove debug lines
  ([`203ba94`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/203ba94af06ed52c37acb16e60cdb49defa0559c))

### Chores

- Add santa challenge problem directory
  ([`477a205`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/477a2057702912ca5f14dabc8d36f6bcb88b8135))

- Modify santa challenge problem directory
  ([`6ed6881`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6ed6881327e388b0b42b256534bed02dc5577d74))

- Polish comparison scripts
  ([`b6e9e89`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/b6e9e89a536cf9c312e2f4ced5db1bb4f797c776))

- Refactor optuna stage
  ([`fd1ca3a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/fd1ca3ab26ad5c9f813cf8cc8c45823980335276))

- Santa2025 problem for n=100
  ([`ef864fc`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/ef864fc4a7ccdd7a711c2064a22ec9b8d112b379))

- Slightly polish code
  ([`86fd5a8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/86fd5a8fb4ffb6022ccd832a21935016d8a8e5ff))

### Code Style

- Clean up stages — loguru placeholders, builtin generics, type annotations, constants
  ([`72a447a`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/72a447a6f31e2e53a694ac405ae1343da2f57d87))

### Documentation

- Add Testing section to README
  ([`245464c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/245464c53cba78ee05e395df114d5a54f57ffe75))

- Update README test section with current structure and run instructions
  ([`da5cd54`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/da5cd54d08060b6c6d38cba18fffd9887538f6f6))

### Features

- 1) add new caching system (based on change in the inputs 2) structured output for mutation
  operator 3) slightly polish insights
  ([`dfb7a73`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/dfb7a73769599792bc6ca217c13fc8baaa0aff7e))

- Add artifact from validation support
  ([`91a4402`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/91a4402f8d13185d251347e1f342070cb5b3cd3d))

- Add cma-es parameters tuning stage
  ([`76a32a1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/76a32a1453fa12d5b8ed619b70620ef78120e292))

- Add first half of missing problems
  ([`4fca319`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4fca3191f5c1023005d72926a28c19f01445475f))

- Add global stats to context
  ([`57cee54`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/57cee54d57ea0e613f4db7a81c8cffc33c06eff3))

- Add missing alphaevolve problems
  ([#46](https://github.com/KhrulkovV/gigaevo-core-internal/pull/46),
  [`f8a15e1`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f8a15e1ab58341158f3a38585258bef015000320))

- Add optuna optimization stage
  ([`8da53df`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8da53df7a004d057db5ed20dfd14035c773f2fe9))

- Add Optuna payload routing and bypass for direct optimization output
  ([`8e37b0c`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8e37b0c53ddbc46ca35854aa6f2c27521b903074))

- Add second half of alphaevolve problems
  ([`574d555`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/574d555a7be6a9de56fe1d05dfa8c904f57497ca))

- Add token counters to metrics
  ([`63323c3`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/63323c3ff419ac9fc518c8ef79a7a93247b934e6))

- Boltzmann/weighted elite selectors, Optuna int preservation, profiler
  ([`1011e38`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/1011e38239d4782df48c7e13bc2321de5c48d10e))

- Dynamic space, more ram stability
  ([`bb3bd4b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/bb3bd4ba551b9b14c3020fe658e2bde4f1aa2add))

- Normalize fitness to [0,1] in FitnessProportionalEliteSelector, fix greedy collapse
  ([`90589b4`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/90589b4a2257950f67a30f52fdfeb85fcc43ac52))

- Polish storage code with claude
  ([`6826922`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/6826922240a9b8da90e11b0420985e3931583d47))

- Removed bad problems, fixed first half of valid ones
  ([`063858b`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/063858bfe58de936cea604e584d2b7320d7fb88c))

- Small efficiency improvements
  ([`f71f9d8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/f71f9d8aed9603ac470d020efd45fe6a9624a63e))

- Unconstrained insights categories
  ([`a4ba5fe`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/a4ba5feb14706456da0ec6e66e92324cdfb4c49d))

- **comparison**: Improve style and polish
  ([`4375952`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/4375952da2c7f6516340ad23b422d7fe38457cfc))

- **prompt**: Add gsm8k, aime, ifbench, pupa, and hotpotqa problems
  ([`44ecb07`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/44ecb073db9f8df5e91433b4049531f5d39bf102))

- **prompts**: Add shared functionality; add aime & jigsaw problems
  ([`7dfdce8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7dfdce855732bb0055abc7c1ab08e36032a7797f))

- **prompts**: Refactor single-prompt evolution, added chains (utils+hotpotqa)
  ([#62](https://github.com/KhrulkovV/gigaevo-core-internal/pull/62),
  [`d63343d`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/d63343df7b662acb610e9f10cc4d94fffdf184f2))

### Performance Improvements

- Pre-compute DAG inputs and improve stage resilience
  ([`e6e3566`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/e6e35666b08b8f22f661d1bd220c2cd6c8dca0d7))

- Reduce Redis round-trips and eliminate deep copies in hot paths
  ([`2c556e8`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/2c556e85dcffd2d069909e8bd71aa72eb0d29dcf))

### Refactoring

- Refactor wizard specs to follow pydantic
  ([`151f120`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/151f120bb1a4a045634d76660606ff6bb30f414e))

- Rewrite uncertainty_inequality
  ([`3f88ed0`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/3f88ed00fa7d7a0b2026f9fa29db6e01e28d3568))

- Simplify program state machine — remove EVOLVING, rename states
  ([`70b9b34`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/70b9b34a2ddef2f0838072f97802c58100bccf9f))

### Testing

- Add comprehensive coverage tests for 12 modules (3086 lines)
  ([`4546525`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/45465254847344c82879ebb7272f03ee6eb218f8))

- Add comprehensive test suite (1132 tests) and reorganize into subdirectories
  ([`8d47344`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/8d4734457cbe744d9c8e681965af05913a6a9412))

- Add comprehensive test suite for core modules
  ([`7628e91`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/7628e9164c4e07efda81c6de97e2cc9d0d97f530))

- Fix flaky ScalarTournamentEliteSelector tests
  ([`5155932`](https://github.com/KhrulkovV/gigaevo-core-internal/commit/515593294efb480e09562588c6da7d96bd7770cb))


<!-- Cleaned up orphaned v1.12.0 tag to unblock semantic-release -->

## v1.11.1 (2025-11-18)

### Bug Fixes

- Set flush_at and flush_interval via client instead of constructor
  ([`887232d`](https://github.com/KhrulkovV/metaevolve/commit/887232dc6d4fb78c8124ca3baafa1df31209b36f))

### Refactoring

- Optimize Langfuse integration
  ([`6876a9d`](https://github.com/KhrulkovV/metaevolve/commit/6876a9d39583fca850b6e4c0cb44c56ff0604a3b))

- Pass flush_at and flush_interval to CallbackHandler constructor
  ([`fc6fbd2`](https://github.com/KhrulkovV/metaevolve/commit/fc6fbd2200316c2b1e2d3b79558ae48bbc61837f))

- Remove redundant try-except for CallbackHandler initialization
  ([`72884b5`](https://github.com/KhrulkovV/metaevolve/commit/72884b5856f1009b4d865c7d35567270ca719634))

- Remove unused flush_traces method
  ([`3b5ccb3`](https://github.com/KhrulkovV/metaevolve/commit/3b5ccb35456ad019bbfa25b6dc28b8bc14846743))


## v1.11.0 (2025-11-18)

### Chores

- Add terminal gif
  ([`4286fd2`](https://github.com/KhrulkovV/metaevolve/commit/4286fd2c826a19ffb99fa5be877c3332caeb28be))

- Add terminal gif
  ([`398056b`](https://github.com/KhrulkovV/metaevolve/commit/398056b7110b24f88b4e39a8007b0c1b3677366d))

- Add terminal gif
  ([`ed4aba1`](https://github.com/KhrulkovV/metaevolve/commit/ed4aba1217763fb19c97a99ad40744bf0ec52228))

- Fix license
  ([`5592e05`](https://github.com/KhrulkovV/metaevolve/commit/5592e0565dadeaf8ef3cd6edf156d8d8923c3cfd))

- Fix license
  ([`1394967`](https://github.com/KhrulkovV/metaevolve/commit/1394967babb59912ddb82f71bf91a762c464367d))

- Remove emoji
  ([`7c4e0f5`](https://github.com/KhrulkovV/metaevolve/commit/7c4e0f5a332a21d5eaec880b853eb8121a8ef33a))

### Features

- Better stage scheduling
  ([`d2e35b0`](https://github.com/KhrulkovV/metaevolve/commit/d2e35b0364c14d2fd0db88e08962230d7c660119))


## v1.10.0 (2025-11-17)


## v1.9.1 (2025-11-17)


## v1.9.0 (2025-11-15)

### Chores

- Removed legacy fields, upd. FunctionSignature note
  ([`f7c832e`](https://github.com/KhrulkovV/metaevolve/commit/f7c832e2b4c2013db40d0ed5bfc0eae08272b1c5))

- Removed wizard example problem
  ([`41f2c77`](https://github.com/KhrulkovV/metaevolve/commit/41f2c77cac9646ade02a5677013f2917cbc1e903))

### Documentation

- Updated wizard documentation
  ([`a6d115e`](https://github.com/KhrulkovV/metaevolve/commit/a6d115e5281088bf3baab69b188b8dda5869ac67))

### Features

- Add problem scaffolding wizard
  ([`e12e566`](https://github.com/KhrulkovV/metaevolve/commit/e12e566d4340b072af4d317e9a871f4f8250f2c5))

### Refactoring

- Moved wizard configs, made wizard a module
  ([`e2af84b`](https://github.com/KhrulkovV/metaevolve/commit/e2af84b49e4630f72a1945802842b68a370adbd8))

- Updated wizard code functionality
  ([`43d0ab6`](https://github.com/KhrulkovV/metaevolve/commit/43d0ab686b5d695018d337bcd00ce1bdd85c1402))


## v1.8.1 (2025-11-14)


## v1.8.0 (2025-11-14)

### Features

- Remove memory leaks and small fixes
  ([`2746ea6`](https://github.com/KhrulkovV/metaevolve/commit/2746ea6c002c0c94d0e46ab6e43d44649dac062c))


## v1.7.1 (2025-11-12)

### Bug Fixes

- Small polish
  ([`e9abd09`](https://github.com/KhrulkovV/metaevolve/commit/e9abd0965a4e9385354f8d9a3aa5de52892bfbca))


## v1.7.0 (2025-11-12)

### Features

- Handling of langfuse errors
  ([`50adbc3`](https://github.com/KhrulkovV/metaevolve/commit/50adbc3abaf3c03712d7c89dc0a215e7b92eb242))

- Langfuse_tracing_less_comments
  ([`5910f96`](https://github.com/KhrulkovV/metaevolve/commit/5910f9665d8fcab50bd53ec9c22930040788a720))

- Simplifying_langfuse_tracing
  ([`c746de7`](https://github.com/KhrulkovV/metaevolve/commit/c746de7845b0610454a27561e230247ab0d9eeb2))

- Update README.md to work with langfuse
  ([`765e0e7`](https://github.com/KhrulkovV/metaevolve/commit/765e0e729afff42bcca5ba66d4ebba0e97996867))


## v1.6.1 (2025-11-12)

### Bug Fixes

- Follow-up on removing task-dependent text
  ([`a72968b`](https://github.com/KhrulkovV/metaevolve/commit/a72968bd1f29007865692bdee586a980cef2b571))

- Remove task-dependent text from evolution prompts
  ([`1c6acf3`](https://github.com/KhrulkovV/metaevolve/commit/1c6acf320c17d08ca428386bad2c4a2bff45e924))


## v1.6.0 (2025-11-11)


## v1.5.2 (2025-11-11)

### Bug Fixes

- Small fix problem name
  ([`6663ea4`](https://github.com/KhrulkovV/metaevolve/commit/6663ea4f50f8256589ae865a6e9022d7cc6a2a3f))

### Features

- More docs and stability for redis
  ([`dbcb856`](https://github.com/KhrulkovV/metaevolve/commit/dbcb85609002197d07c7b1e3ef58f274d788bba9))


## v1.5.1 (2025-11-11)

### Bug Fixes

- Small fix problem name
  ([`44ea85f`](https://github.com/KhrulkovV/metaevolve/commit/44ea85ff182cc835cd30f780d0ecf0d3a3a6555b))


## v1.5.0 (2025-11-11)

### Features

- Better config structure, examples, and polish
  ([`643331e`](https://github.com/KhrulkovV/metaevolve/commit/643331e2ef09c997da25cb55cfeacbcddab58653))


## v1.4.0 (2025-11-07)

### Features

- Wandb support; improve pythonpath passthrough
  ([`e408cec`](https://github.com/KhrulkovV/metaevolve/commit/e408ceca5b45a447d74c24c085bdf3c814c836e8))


## v1.3.0 (2025-11-06)

### Bug Fixes

- Remove now unused runner
  ([`a18749b`](https://github.com/KhrulkovV/metaevolve/commit/a18749b7ae231041504f484b8a5c79081f58f380))

- Remove now unused runner
  ([`0311c27`](https://github.com/KhrulkovV/metaevolve/commit/0311c27db9454f4f6f6120d593c342e4085433b0))

- Unify log dir
  ([`3097d4d`](https://github.com/KhrulkovV/metaevolve/commit/3097d4d38dbfb7ce4f38b62f8bbc5005b9f1a5e3))

### Features

- 1) add metrics logging with tensorboard 2) fix execution ordering in evolution engine 3) fix
  island api 4) add proper cancelation handling for async method
  ([`a0e7d8e`](https://github.com/KhrulkovV/metaevolve/commit/a0e7d8ebb7e391e4d986e62a2ce427b75e45886b))


## v1.2.0 (2025-11-06)

### Chores

- **prompts**: Removed unused prompt constants, moved task hints to description
  ([`b6db207`](https://github.com/KhrulkovV/metaevolve/commit/b6db207386179e62e9b665d9fcbda140e824c3d3))

### Features

- **prompts**: Centralize mutation prompts and remove task hints
  ([`6bd971f`](https://github.com/KhrulkovV/metaevolve/commit/6bd971fd1bf7dac06d6cf796e9bcdc002412f759))

### Refactoring

- **prompts**: Add task-independent mutation prompts
  ([`e56dd49`](https://github.com/KhrulkovV/metaevolve/commit/e56dd49fe79dedf08a78683f36935e32db886e9e))


## v1.1.0 (2025-10-31)

### Features

- Changed pickle serialization to cloudpickle for classes and lambdas over network
  ([`6524340`](https://github.com/KhrulkovV/metaevolve/commit/652434069eaf07dda49b29d2a64338539c448760))


## v1.0.3 (2025-10-31)


## v1.0.2 (2025-10-31)

### Bug Fixes

- Better error handling and logging in dag
  ([`5973d7b`](https://github.com/KhrulkovV/metaevolve/commit/5973d7bfa6b76e1f96ccefc1da13606024b03d4f))


## v1.0.1 (2025-10-31)

### Bug Fixes

- Add missing dep
  ([`acecda2`](https://github.com/KhrulkovV/metaevolve/commit/acecda25cf697398f2f13c377641f5e4e168df2a))

- Minor fixes to simplify hydra and fix prompt for insights
  ([`60aa776`](https://github.com/KhrulkovV/metaevolve/commit/60aa7764d920233d95ca689ffcd8d363dfb3f5f6))

### Chores

- **deps**: Move hydra dependencies to main requirements
  ([`6e574d9`](https://github.com/KhrulkovV/metaevolve/commit/6e574d90ed220192a4df0087a71ed59e61f49679))

### Refactoring

- Migrate MetaEvolve to GigaEvo
  ([`fc8c2ca`](https://github.com/KhrulkovV/metaevolve/commit/fc8c2ca64e0a1323a0101ea16ac3b3647df60892))


## v1.0.0 (2025-10-31)


## v0.9.0 (2025-09-26)


## v0.8.0 (2025-09-22)


## v0.7.0 (2025-09-21)

### Chores

- **release**: V0.7.0
  ([`5f102b4`](https://github.com/KhrulkovV/metaevolve/commit/5f102b4918de8343bbc50b80d27c93a64efc1f71))


## v0.6.0 (2025-09-20)

- Initial Release
