# `dag_tab` research report

Updated: 2026-07-20

## Executive summary

The latest California and Adult experiments show that the quality plateau after roughly 300 mutation attempts is not primarily caused by large graphs failing validation. Large persisted California graphs remained mostly valid, and the strongest California candidates reached the configured node boundary. The dominant limitations were mutation transport reliability, accidental dependency flattening, prompt bias against graph depth, a full-child mutation representation that becomes harder as parents grow, weak late-run search diversification, and CV noise larger than the final Adult gains.

This revision fixes the implementation defects that could be addressed without replacing the search algorithm:

- Gemini structured-output transport is selected by the router instead of being forced to `json_schema`;
- function-calling schemas are translated to the correct `parameters` wire format;
- Gemini concurrency and retry settings are applied;
- intermittent mutation failures no longer terminate a useful run after only a few errors;
- retained nodes preserve dependencies when the field is omitted, while `[]` explicitly rewires them;
- mutations declare and verify `structural_intent` and optional `minimum_child_depth`;
- the default graph budget is 12 nodes, with a hard safety limit of 16;
- graph size and depth remain diagnostics but are removed from LLM optimization context;
- node code normalization and AST input/dependency synchronization accept supported full-function and DataFrame patterns;
- final test scoring delegates to `problems/tabular`'s own `score_on_test`, so dag_tab and tabular candidates share one protocol.

## Experiment evidence

### California

The completed Gemini 3 run produced a best CV \(R^2\) of 0.864554 from an eight-node graph. Under the shared single-refit test protocol it reached test RMSE 0.407899 and test \(R^2\) 0.872402, versus baseline RMSE 0.429330 and \(R^2\) 0.858642. This is a meaningful transfer of the engineered spatial and density features to the holdout.

The matched Gemini 3.5 run completed all 300 persisted mutants with `completion_reason=max_mutants_reached`; the engine snapshot ended at mutation attempt 1350. Its best observed CV \(R^2\) was 0.873174. KNN aggregate features were among the strongest candidates, and valid examples also composed KNN-produced local statistics into depth-two downstream features.

The historical Gemini 3.5 run exposed 671 structured-output parse failures before completion. These failures were transport/schema incompatibilities, not evidence that FeatureGraph validation rejected large graphs. The transport adapter is fixed in this revision, so old failure counts should not be used as an estimate for new runs.

### Adult

The best completed Adult candidate improved CV accuracy from 0.875537 to 0.876113, but the single-refit holdout result was effectively tied with baseline: accuracy changed from 0.874393 to 0.874271 while AUC changed from 0.928163 to 0.928379. The late CV gain is too small to distinguish reliably from fold noise. Adult therefore remains the useful stress dataset for selection, repeated-CV, and effect-threshold experiments rather than evidence of a robust quality improvement.

## Why previous graphs were mostly flat

The previous mutation contract made a flat graph easier to express than a composed one. Most new nodes read raw columns directly, retained-node mutations could silently replace omitted dependencies with an empty list, and graph depth/node count were shown with a lower-is-better orientation. This produced wide collections of parallel feature functions even when generated intermediates were available.

The current contract distinguishes four topological intentions:

- `local_edit` may keep the same depth;
- `extend_chain` must increase the selected parent depth;
- `compose_chain` must produce a dependency path of at least depth two;
- `simplify_graph` may deliberately remove structure.

Python verifies the resulting depth. Omitted dependencies on a retained node preserve surviving parent edges; an explicit empty list removes them. This makes depth greater than one representable and testable without optimizing depth for its own sake.

## KNN graph interpretation

A KNN algorithm can be internally multi-stage while remaining one FeatureGraph node. An aggregate node can fit `NearestNeighbors` on `df_fit`, query neighbors for `df`, and emit local means, standard deviations, or density proxies. Its structural depth is still one when no later FeatureNode consumes those outputs.

A true depth-two KNN DAG has a producer aggregate node that emits local neighborhood features and a consumer rowwise or aggregate node that declares the producer as a dependency and reads those generated columns. Internal algorithmic complexity and graph dependency depth should therefore be reported separately.

## Final test protocol

Final test evaluation delegates to the selected `problems/tabular` problem object: `score_on_test` builds that problem and hands it the FeatureGraph model factory. Splits, fold contract, and metric set therefore come from the same code path a `problems/tabular` program is scored by, and no part of the protocol is reimplemented in `problems/dag_tab`.

The consequence for this report is that dag_tab test numbers are directly comparable with the published `problems/tabular` baselines for the same dataset, without a protocol-conversion caveat.

## Remaining limitations

1. **Search budgets are incomplete.** `max_mutants` counts persisted mutants. A 300-mutant run can require far more mutation attempts and LLM calls. Attempts, parsed diffs, valid evaluations, accepted candidates, tokens, and wall time need independent limits and telemetry.
2. **The full-child slot schema scales with parent size.** Twelve slots permit deeper graphs but also enlarge prompts and increase invalid-reference risk. A compact edit-script representation should be compared against the current deterministic transcription/repair path.
3. **Selection plateaus.** Adult improves very rarely, and California often finds its best candidate early. Stagnation restarts, parent age, operator bandits, semantic novelty, and mutation-family islands require controlled ablations.
4. **Late gains are noisy.** Finalists should use repeated paired folds or confidence-adjusted selection before opening the test set.
5. **Trusted Python remains trusted.** Node execution is correctness-validated but not sandboxed. Hostile-code isolation requires a subprocess or container backend.
6. **Evolutionary CV still uses the shared generic tabular orchestration.** Final test is stratified for classification, but the generic evolutionary splitter currently uses plain K-fold behavior and repeatedly uses the canonical validation split for early stopping. This should be redesigned separately rather than silently changed in this PR.

## Recommended next experiments

### Preflight

Run at least 50 structured mutation calls across parent sizes 0, 4, 8, and 12. Require at least 95% parse success and report diff-apply, execution, and model-fit yield separately.

### Composition smoke

Use 30–50 persisted mutants on California with `max_nodes=12`. Record the share of `extend_chain` and `compose_chain` proposals that successfully produce depth 2, 3, and 4 graphs, and verify that retained parent edges survive local edits.

### Paired search study

Use identical dataset, model preset, seed policy, estimator, MAP-Elites configuration, and valid-candidate budget. Compare:

- previous eight-node/current-slot behavior;
- corrected twelve-node structural-intent behavior;
- later, current slots versus compact edit script.

Run California, Adult, and Otto with at least three seeds. Compare valid yield, accepted yield, frontier improvement timing, archive coverage, depth distribution, generated-feature reuse, token cost, and wall time.

### Finalist evaluation

Freeze candidate IDs before reading holdout metrics, then score baseline and all frozen finalists in the same revision through `problems.dag_tab.test`, which routes to the shared `problems/tabular` scorer.

## Verification status

The focused `tests/dag_tab` suite is green. It covers test-scorer delegation to `problems/tabular`, pinned estimator hyperparameters, full-class probability alignment, early stopping, aggregate preprocessing, target transforms, graph depth contracts that ignore declared-but-unread edges, dependency preservation, structured-output negotiation, and dynamic dataset configuration.
