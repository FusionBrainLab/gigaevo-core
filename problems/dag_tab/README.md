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

Nodes are stored in topological order. `dependencies` name earlier nodes; generated inputs must come from those dependencies. Nodes with `is_output=true` export their generated columns to a fixed histogram-gradient-boosting estimator. Raw features are retained alongside generated outputs.

Node code is a function body over a pandas DataFrame named `df`. `np` and `pd` are available. Every declared output must be assigned explicitly with `df['name'] = ...`, and the body must end with `return df`. Imports, target access, file/network/process operations, private attributes, `eval`, and `exec` are rejected. Execution must preserve rows and index and create no undeclared columns.

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
  llm=qwen_thinking \
  llm_base_url=http://localhost:8000/v1 \
  model_name=Qwen3-235B-A22B-Thinking-2507 \
  num_parents=1 \
  max_mutants=10 \
  max_in_flight=1 \
  llm_max_concurrent=1 \
  max_consecutive_mutation_failures=3 \
  thinking_token_budget=64000 \
  max_tokens=72000 \
  request_timeout=600 \
  stage_timeout=3600 \
  dag_timeout=7200
```

`python -u` keeps progress visible in `tmux`. `max_consecutive_mutation_failures` bounds failed LLM/schema attempts that do not increment `max_mutants`; every successfully persisted mutation resets the failure streak. Set it to `0` only when intentionally disabling this guard.

This command intentionally uses the Thinking model. On compatible endpoint versions, `thinking_token_budget=64000` caps reasoning, while total `max_tokens=72000` leaves up to 8000 tokens for the final structured diff. Keep `max_in_flight=1` and `llm_max_concurrent=1` for the first run; increase concurrency only after several valid mutations. For a quick smoke, set `max_mutants=1`. Node code remains a simple structured string so guided decoding can terminate reliably; Python deterministically appends a missing final `return df`, then the AST validator enforces safety and declared outputs.

## Baseline evaluation and final test

Evaluate the seed with the existing tabular CV protocol:

```bash
GIGAEVO_TABULAR_DATA=/path/to/tabm-data/data \
GIGAEVO_TABULAR_CV_FOLDS=2 \
conda run -n documents python -c \
'import json; from pathlib import Path; from problems.dag_tab.validate import validate; p=json.loads(Path("problems/dag_tab/initial_programs/baseline.json").read_text()); print(validate(p))'
```

Score a selected JSON genome on the untouched test split:

```bash
GIGAEVO_TABULAR_DATA=/path/to/tabm-data/data \
conda run -n documents python problems/dag_tab/test.py /path/to/program.json
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
