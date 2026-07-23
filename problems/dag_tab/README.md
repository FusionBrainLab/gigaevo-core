# DAG Tab: FeatureGraph evolution

`dag_tab` evolves a JSON DAG of pandas feature transformations while reusing the standard GigaEvo engine, JSON-document pipeline, structured-diff mutation agent, storage, lineage, mutation context, and MAP-Elites archive. It also reuses dataset arrays and metadata, cross-validation protocol, metrics, behavior descriptors, and the final-test split from `problems/tabular`. It is the CatBoost control in the broader [`tabular_dag_baselines`](../tabular_dag_baselines/README.md) suite.

The current architecture is dataset-parameterized. `problem.dataset=<name>` selects the data and semantic context, while `loader=dag_tab_seed` creates a neutral raw-feature graph with the exact dataset width at runtime. No separate evolution runtime, LLM client, or dataset-specific FeatureGraph seed lives in this problem. Experimental evidence, resolved defects, remaining limitations, and the next study design are summarized in [`docs/dag_tab_research_report.md`](../../docs/dag_tab_research_report.md).

## Architecture status

The current implementation includes:

- dynamic dataset context assembled from this universal ABI plus `problems/tabular/<dataset>/task_description.txt`;
- dynamic neutral seeds with `nodes=[]` and raw columns `x0...xN-1`;
- first-mutation support for neutral parents, followed by keep/new/omit/edit/rewire mutations for non-empty graphs;
- exact successful `max_mutants` budgeting even when mutation attempts fail;
- structured validator failure reason, stage, and node telemetry;
- stable terminal `completion_reason` and final metrics/ingestion drains before shared storage closure.

Real neutral-seed smoke checks passed for regression (`california`), binary classification (`adult`), and multiclass classification (`otto`). The focused `tests/dag_tab` suite is green, including fold-local preprocessing and delegation of untouched-test scoring to `problems/tabular`.

## Genome

`Program.code` is the complete JSON FeatureGraph and is the only source of truth:

```json
{
  "schema_version": 1,
  "dataset": "california",
  "raw_columns": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
  "dropped_raw_columns": [],
  "target": null,
  "nodes": [
    {
      "id": "income_per_age",
      "kind": "rowwise",
      "input_cols": ["x0", "x1"],
      "output_cols": ["fe_income_per_age"],
      "output_types": {"fe_income_per_age": "numerical"},
      "code": "df['fe_income_per_age'] = df['x0'] / (df['x1'].abs() + 1.0)\nreturn df",
      "rationale": "Relate income to housing age.",
      "dependencies": [],
      "is_output": true
    }
  ]
}
```

Nodes are stored in topological order. `dependencies` name earlier nodes; generated inputs must come from those dependencies. Nodes with `is_output=true` export their generated columns to a fixed CatBoost estimator. Raw features are retained alongside generated outputs. Generated columns explicitly declare semantic types (`numerical`, `categorical`, `binary`, or `ordinal`); old genomes default omitted types to `numerical`. CatBoost categorical positions are derived from the final ordered feature schema.

There are two node ABIs. `kind=rowwise` wraps code as `transform(df)` for per-row arithmetic and composition. `kind=aggregate` wraps code as `transform(df_fit, y_fit, df)`: `df_fit` is a pandas DataFrame, while `y_fit` is an aligned one-dimensional NumPy array. Code that needs pandas alignment must explicitly construct `pd.Series(np.asarray(y_fit), index=df_fit.index)`. Aggregate code may fit statistics or estimators only on fold-fit inputs and then apply them to `df`. Supervised fit-row outputs must use leave-one-out or deterministic out-of-fold logic; an own-target perturbation probe enforces that a row's feature cannot depend on its own target. Both frames expose only declared inputs. Runtime checks preserve rows/index and inputs, require exactly the declared outputs, validate semantic output types, allow numerical NaN, and reject infinities. Reading a column the node did not declare fails at execution, where the interpreter has resolved the name. The commonest ABI mistakes — `df_fit`/`y_fit` in rowwise code, pandas-only calls such as `y_fit.groupby(...)` — are reported with a corrective hint on the raised exception rather than rejected from the syntax tree, because whether a name holds the ABI object depends on scope and control flow.

Node code is trusted Python, not a security sandbox. Complete `transform` modules are compiled as written, so helper functions and module-level setup are legal and module state persists across calls within that compile; bare bodies are wrapped in the ABI-appropriate `transform` signature. Imports are legal, while `compile()` provides syntax diagnostics. Security isolation, if required, belongs in a replaceable subprocess/container execution backend rather than an AST vocabulary denylist. Correctness remains behaviorally enforced: frozen-fit batch-purity rejects outputs that depend on companion transform rows, and repeated identical execution rejects unseeded stochastic transforms.

For each evolutionary evaluation fold, train, validation, and query are transformed under one graph contract while aggregate `df_fit`/`y_fit` contain only fold-fit rows. Validation and query never contribute to fitted state. CatBoost first uses transformed train/validation for early stopping. The normal `fit_predict` path then fits the graph again from scratch on train+validation and refits the estimator at the selected iteration count before predicting the evolutionary query.

Final untouched-test scoring delegates to `problems/tabular`'s own `score_on_test`, so a dag_tab genome and a `problems/tabular` program are scored by byte-identical protocol code: one fit on the stored train/validation split, one prediction of the untouched test matrix, and the metric set that dataset's task type defines. Nothing about the test protocol is reimplemented here.

Three graph-level completeness controls share that refit contract. `dropped_raw_columns` hides selected raw inputs only at the estimator boundary, so nodes may still use them. A generated `sample_weight` column supplies finite non-negative fit weights and is removed from every feature matrix. An optional regression `target` supplies `transform(y_fit, y)` and `inverse(y_fit, predictions)` bodies; a deterministic round-trip probe validates the pair, fitting uses the transformed scale, and returned predictions are inverted to the original scale.

Example aggregate node body:

```python
mean = df_fit['x0'].mean()
std = df_fit['x0'].std()
df['fe_income_z'] = (df['x0'] - mean) / (std + 1e-8)
return df
```

A later rowwise node can consume `fe_income_z` through an explicit dependency.

## Mutation

`mutation=structured_diff_dag_tab` instantiates the existing `StructuredDiffMutationOperator` with `AllowedDagTabChanges`. The generic operator and agent are unchanged.

A mutation emits the complete child as a topologically ordered `nodes` array (up to the hard safety limit of 16):

- `kind=keep` retains a rendered parent node and may edit its feature contract;
- `kind=new_rowwise` creates a `transform(df)` node, while `kind=new_aggregate` creates a `transform(df_fit, y_fit, df)` node;
- keep edits may override the retained node ABI with `kind=rowwise|aggregate`;
- `output_types` exposes generated categorical/binary/ordinal features to CatBoost; if a retained node changes `output_cols` without updating types, transcription preserves types for retained names, drops stale names, and defaults new outputs to `numerical`;
- omitted parent nodes are deleted;
- array order and node-id `dependencies` define rewiring; omitted/null dependencies on a retained node preserve its surviving parent edges, while an explicit empty list removes them;
- multiple nodes may be added or changed in one mutation;
- array order is topological; dependencies resolve against earlier entry ids, unresolved references are dropped, and repeated ids receive deterministic unique child ids;
- transcription scans literal `df[...]`/`df_fit[...]` reads and appends missing raw or earlier-generated columns to `input_cols`, then restores the corresponding dependency edge; a read it cannot place against a real column is left alone, so execution reports it against the column that is actually missing, and runtime checks remain authoritative;
- graph size, generated feature count, and depth remain recorded diagnostics but are no longer shown to the LLM as objectives to minimize;
- every diff declares `structural_intent=local_edit|extend_chain|compose_chain|simplify_graph`; `extend_chain` must increase the selected parent's depth, `compose_chain` must produce depth at least 2, and optional `minimum_child_depth` is verified after deterministic transcription;
- composition guidance asks the model to reuse genuine intermediate features without manufacturing depth for its own sake;
- `dropped_raw_columns` edits feature selection without removing node inputs, while `target_change` keeps, drops, or sets an invertible regression target transform;
- `dataset`, `raw_columns`, graph validation, and child JSON assembly remain Python-owned.

The diff also carries standard GigaEvo mutation evidence (`archetype`, `justification`, insights, cards, and changes), so lineage and memory attribution continue to work normally.

Invalid evaluations return sentinel numeric metrics plus structured artifact fields: `validation_failure_reason`, `validation_failure_stage`, and, when recoverable from the error, `validation_failure_node`. Stable reason values distinguish schema, execution, non-finite output, batch-purity, determinism, own-target invariance, target round-trip, model fit, sample-weight, and dataset-contract failures without parsing free-form exception text.

Engine shutdown persists a machine-readable `completion_reason` in `engine:snapshot` (`max_mutants_reached`, `wall_time_limit`, `fitness_plateau`, `external_signal`, or `engine_error`). Engine, DAG-runner, and per-program metric collectors each perform a final collection/drain before the run-level finalizer closes their shared storage, so terminal counters include work completed between the last periodic poll and shutdown without component-close races.

## Data

Set `GIGAEVO_TABULAR_DATA` to the same TabM data root as `problems/tabular`.

Select a dataset with `problem.dataset=<name>` and use `loader=dag_tab_seed`. The loader creates a neutral raw-feature FeatureGraph at runtime with the exact `x0...xN-1` schema reported by that dataset, so no California-specific seed is reused. `DagTabProblemContext` combines this file's universal FeatureGraph ABI with the selected dataset's `TASK`, `DATASET`, and `COLUMNS` sections from `problems/tabular/<name>/task_description.txt`. Its implementation lives in `problem_context.py`: the reserved `context.py` filename is intentionally absent because GigaEvo interprets it as a runtime `build_context` hook and would otherwise add an incompatible `AddContext` stage. The existing train/validation/test splits remain authoritative; test data is never used as an evolution fitness signal.

## Launches

Required environment: `GIGAEVO_TABULAR_DATA` and `OPENAI_API_KEY`. Qwen-backed
Memory V2 additionally requires `LOCAL_LLM_PROXY` and `LITELLM_MASTER_KEY`.

The preferred entry point is the same one-switch interface as every other
evaluator. California is the default:

```bash
python run.py experiment=tabular_dag/catboost
```

These are single 100-mutation S4-style runs on `adult`; replace
`problem.dataset=adult` as needed. Existing defaults provide three-fold mean CV,
one parent, disk storage, live Memory V2 writes, and per-run Hydra output paths.

### Without memory

```bash
python run.py \
  experiment=tabular_dag/catboost \
  problem.dataset=adult \
  pipeline=guided \
  memory=none \
  llm=gemini35_flash \
  algorithm=tabular/2d_local_ood \
  mutation_operator.allowed_changes.max_nodes=10 \
  max_mutants=100
```

### With Memory V2

The base config already selects `pipeline=memory_guided memory=v2`.

```bash
python run.py \
  experiment=tabular_dag/catboost \
  problem.dataset=adult \
  llm=gemini35_flash \
  memory/llm=qwen_instruct \
  algorithm=tabular/2d_local_ood \
  mutation_operator.allowed_changes.max_nodes=10 \
  max_mutants=100
```

Use `llm=gemini3_flash` instead for Gemini 3 Flash Preview. Evaluator-provided
`_evaluation_measurements` are stored in program metadata and used by Memory V2
as outcome SE; absent measurements remain `se=None`.

## MAP-Elites archive

Without an `algorithm` override the engine composes the default single-island archive whose behavior space is one axis over *fitness itself* — no diversity pressure, and the two tabular behavior descriptors are computed and stored but never used as archive coordinates. Runs converge within a handful of generations.

`algorithm=tabular/2d_local_ood` is the recommended setting: it bins the same `problems/tabular` descriptors this problem already emits, `local_lipschitz_p95` × `ood_delta_slope` (15×10 cells, dynamic bounds), so structurally different graphs at similar fitness occupy different cells and survive.

## Baseline evaluation and final test

Evolutionary evaluation and final test both run the selected `problems/tabular` problem object, so fold counts, splits, and metrics are whatever that dataset already declares — dag_tab supplies only the model factory. The fixed CatBoost recipe is shared across datasets (lr 0.05, depth 6, seed 0, early stopping, then refit on the evaluation train+validation split at the selected iteration count) and is pinned by `tests/dag_tab/test_validator.py::test_fixed_estimator_hyperparameters_are_pinned`, because it is the control variable a feature-DAG campaign holds constant. Do not set `GIGAEVO_TABULAR_CV_FOLDS` unless you also change it for the tabular runs you compare against.

The neutral seed is evaluated automatically by the normal run. A zero-mutation seed smoke for another dataset is:

```bash
python run.py \
  experiment=tabular_dag/catboost \
  problem.dataset=adult \
  max_mutants=0
```

Score a selected JSON genome on the untouched test split, under the same protocol a `problems/tabular` program would get:

```bash
python -m problems.dag_tab.test /path/to/program.json
```

The JSON result carries exactly the metric keys that dataset's task type emits in `problems/tabular`, so dag_tab and tabular test numbers are directly comparable.

For graph-by-estimator transfer after evolution, use the suite's
[`compare`](../tabular_dag_baselines/compare.py) and
[`compare_matrix`](../tabular_dag_baselines/compare_matrix.py) commands. They
keep each saved graph fixed and report repeated-seed mean and sample standard
deviation without creating another evolution run.

## Tests

```bash
python -m pytest tests/dag_tab -q
```

The focused suite covers graph invariants, neutral and non-empty parent mutation schemas, node-code execution, invalid-candidate handling, portable parent-specific schemas, keep/new/omit/rewire behavior, dynamic dataset context and seeds, Hydra composition, and the generic structured-diff operator end to end.

## Recommended generalization experiment

Run paired 300-mutant experiments on `california`, `adult`, and `otto`, keeping `llm_max_concurrent=1` and comparing `max_in_flight=1` against `2`. Compare each run against its own neutral seed using fitness delta, valid yield, structured failure reasons, MAP-Elites coverage, token cost, and wall-clock throughput. Raw scores are not comparable across regression, binary accuracy, and multiclass accuracy.

## Current scope

- One to twelve nodes per generated child by default, configurable up to sixteen.
- One selected parent is recommended for the first experiments; the schema can render multiple parents, but cross-parent feature contracts remain the LLM's responsibility and are validated after transcription.
- The estimator is fixed; only the FeatureGraph evolves.
- Supervised aggregate nodes receive fold-fit `y_fit`; own-target invariance requires leave-one-out or out-of-fold training features.
- Feature selection, fit-only sample weighting, and invertible regression target transforms are evolvable graph contracts.
- Data-inferred categorical vocabularies fit on training rows only; unseen values map to a stable unknown representation at the estimator boundary.
- Node code is trusted Python. The in-process executor is not a security boundary for hostile code; future subprocess/container isolation can implement the same graph ABI.
