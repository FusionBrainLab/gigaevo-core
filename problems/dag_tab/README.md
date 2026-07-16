# DAG Tab: FeatureGraph evolution

`dag_tab` evolves a JSON DAG of pandas feature transformations while reusing the standard GigaEvo engine, JSON-document pipeline, structured-diff mutation agent, storage, lineage, mutation context, and MAP-Elites archive. It also reuses the dataset loader, cross-validation protocol, metrics, behavior descriptors, and final-test split from `problems/tabular`.

No separate evolution runtime or LLM client lives in this problem.

## Genome

`Program.code` is the complete JSON FeatureGraph and is the only source of truth:

```json
{
  "schema_version": 1,
  "dataset": "california",
  "raw_columns": ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
  "nodes": [
    {
      "id": "income_per_age",
      "input_cols": ["x0", "x1"],
      "output_cols": ["fe_income_per_age"],
      "code": "df['fe_income_per_age'] = df['x0'] / (df['x1'].abs() + 1.0)\nreturn df",
      "rationale": "Relate income to housing age.",
      "dependencies": [],
      "is_output": true
    }
  ]
}
```

Nodes are stored in topological order. `dependencies` name earlier nodes; generated inputs must come from those dependencies. Nodes with `is_output=true` export their generated columns to a fixed CatBoost estimator. Raw features are retained alongside generated outputs. The estimator uses early stopping on each protocol validation split, then refits on the combined train and validation data with the selected iteration count.

Node code is a function body over a pandas DataFrame named `df`. `np` and `pd` are available. Every declared output must be assigned explicitly with `df['name'] = ...`, and the body must end with `return df`. An AST blocklist rejects unsafe syntax and APIs such as imports, private attributes, `eval`, and `exec`. The restricted execution frame exposes only declared inputs, not targets; execution must preserve rows and index, may create only declared outputs, and must leave declared inputs unchanged.

The current node ABI is intentionally stateless and split-invariant. The AST blocklist rejects known aggregate and order-dependent operations such as mean/rank/quantile/groupby/rolling, while row-wise reductions across columns with explicit `axis=1` are allowed. Validation also runs a batch-invariance probe that rejects graphs whose outputs change when the same rows are evaluated in a different batch composition. Supporting train-fitted aggregate features safely requires a future `fit`/`transform` ABI whose learned statistics are reused on validation and query data.

## Mutation

`mutation=structured_diff_dag_tab` instantiates the existing `StructuredDiffMutationOperator` with `AllowedDagTabChanges`. The generic operator and agent are unchanged.

Like the CARL chain vocabulary, a mutation emits a complete child as positional `slot_1...slot_8` values:

- `kind=keep` retains a rendered parent node and may edit its feature contract;
- `kind=new` creates a new pandas transformation;
- omitted parent nodes are deleted;
- slot order and `dependencies` define rewiring;
- multiple nodes may be added or changed in one mutation;
- dependencies can reference only earlier slots, so cycles and forward references are unrepresentable in the generated schema;
- `dataset`, `raw_columns`, graph validation, and child JSON assembly remain Python-owned.

The diff also carries standard GigaEvo mutation evidence (`archetype`, `justification`, insights, cards, and changes), so lineage and memory attribution continue to work normally.

## Data

Use the same TabM data root as `problems/tabular`:

```bash
export GIGAEVO_TABULAR_DATA=/path/to/tabm-data/data
```

The genome's `dataset` field can name an existing dataset from that root. The first supplied seed uses `california`. Its raw columns must be exactly `x0...x7`, matching the eight assembled California columns. The existing train/validation/test splits remain authoritative; test data is never used as an evolution fitness signal.

## Install

From the repository root:

```bash
conda run -n documents python -m pip install -e ".[test]"
```

## Config-only check

This resolves the complete engine without making an LLM call:

```bash
conda run -n documents python run.py \
  problem.name=dag_tab \
  program_format=json_document \
  mutation=structured_diff_dag_tab \
  num_parents=1 \
  max_mutants=3 \
  --cfg job
```

Confirm that the resolved config contains:

- `program_loader.pattern: '*.json'`;
- `JsonDocumentEvaluationFeature`;
- `StructuredDiffMutationOperator`;
- `AllowedDagTabChanges`.

## Qwen experiment

```bash
conda activate documents
export OPENAI_API_KEY=sk-gigaevo
export GIGAEVO_TABULAR_DATA=/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/datasets/tabm-data/data
export NO_PROXY="INTERNAL_IP,localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

python -u run.py \
  problem.name=dag_tab \
  program_format=json_document \
  pipeline=guided \
  memory=none \
  mutation=structured_diff_dag_tab \
  mutation_operator.allowed_changes.max_nodes=3 \
  algorithm=tabular/2d_local_ood \
  llm=qwen_thinking \
  llm_base_url=http://localhost:8000/v1 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  num_parents=1 \
  max_mutants=10 \
  max_in_flight=1 \
  llm_max_concurrent=1 \
  thinking_token_budget=64000 \
  max_tokens=72000 \
  request_timeout=600 \
  stage_timeout=3600 \
  dag_timeout=7200
```

`python -u` keeps progress visible in `tmux`.

A Qwen thinking run is slow enough to expose two engine defaults that this problem does not tune: the parent-refresh budget (600 s) is well under the 7200 s `dag_timeout`, so a refresh of an expensive parent can be killed mid-flight, and the 5 s final-ingestion sweep drops metrics for mutants still in flight at shutdown. Both are engine-wide concerns, not dag_tab ones — raise them in the engine rather than working around them here.

## Gemini 3 Flash via OpenRouter

`config/llm/gemini3_flash.yaml` uses `google/gemini-3-flash-preview`, OpenRouter, and `structured_output_method: function_calling`. Keep the API key only in the environment:

```bash
export OPENAI_API_KEY=<openrouter-api-key>
export GIGAEVO_TABULAR_DATA=/path/to/tabm-data/data

python -u run.py \
  problem.name=dag_tab \
  program_format=json_document \
  pipeline=guided \
  memory=none \
  mutation=structured_diff_dag_tab \
  mutation_operator.allowed_changes.max_nodes=10 \
  algorithm=tabular/2d_local_ood \
  llm=gemini3_flash \
  num_parents=1 \
  max_mutants=100
```

Engine defaults handle concurrency and timeouts — no extra overrides needed. For a first endpoint/schema smoke test set `max_mutants=1`, then scale up. Verified 2026-07-16: 100/100 mutations schema-valid in under 9 minutes (~1.7M tokens).

## MAP-Elites archive

Without an `algorithm` override the engine composes the default single-island archive whose behavior space is one axis over *fitness itself* — no diversity pressure, and the two tabular behavior descriptors are computed and stored but never used as archive coordinates. Runs converge within a handful of generations.

`algorithm=tabular/2d_local_ood` is the recommended setting: it bins the same `problems/tabular` descriptors this problem already emits, `local_lipschitz_p95` × `ood_delta_slope` (15×10 cells, dynamic bounds), so structurally different graphs at similar fitness occupy different cells and survive.

## Baseline evaluation and final test

Evaluation is exactly the `problems/tabular/california` protocol: the same `TabularProblem.validate` (default 3-fold CV, `KFold(shuffle=True, random_state=0)`, mean fold R² as fitness) and the same fixed CatBoost recipe as `problems/tabular/california/initial_programs/prog5.py` (lr 0.05, depth 6, seed 0, 2000 iterations with 50-round early stopping on the protocol validation split, then refit on train+val at the selected iteration count). `tests/dag_tab/test_validator.py::test_fixed_estimator_predictions_match_california_prog5` pins the estimators to byte-identical predictions. Do not set `GIGAEVO_TABULAR_CV_FOLDS` unless you also change it for the tabular runs you compare against.

Evaluate the seed with that protocol:

```bash
GIGAEVO_TABULAR_DATA=/path/to/tabm-data/data \
conda run -n documents python -c \
'import json; from pathlib import Path; from problems.dag_tab.validate import validate; p=json.loads(Path("problems/dag_tab/initial_programs/baseline.json").read_text()); print(validate(p))'
```

Score a selected JSON genome on the untouched test split:

```bash
GIGAEVO_TABULAR_DATA=/path/to/tabm-data/data \
conda run -n documents python -m problems.dag_tab.test /path/to/program.json
```

## Tests

```bash
conda run -n documents python -m pytest tests/dag_tab -q
```

The focused suite covers graph invariants, node-code execution, invalid-candidate handling, portable parent-specific schemas, keep/new/omit/rewire behavior, Hydra composition, and the generic structured-diff operator end to end.

## Current scope

- One to eight nodes per generated child, configurable up to sixteen.
- One selected parent is recommended for the first experiments; the schema can render multiple parents, but cross-parent feature contracts remain the LLM's responsibility and are validated after transcription.
- The estimator is fixed; only the FeatureGraph evolves.
- Node code cannot access targets, so target encoding is intentionally outside this first version.
- This executor is a constrained research executor, not a security boundary for hostile code.
