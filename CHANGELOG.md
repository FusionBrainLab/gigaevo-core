# CHANGELOG

<!-- version list -->

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
