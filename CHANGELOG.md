# CHANGELOG

<!-- version list -->

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
