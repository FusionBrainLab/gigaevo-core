# Tabular FeatureGraph estimator baselines

This suite evolves the same JSON FeatureGraph with several fixed tabular
estimators. It is the estimator-controlled counterpart of
[`problems/dag_tab`](../dag_tab/README.md): data splits, graph execution,
leakage probes, mutation schema, CV metrics, behavior descriptors, and the
untouched test protocol stay shared. Only estimator-specific preprocessing and
model fitting change.

The implementations are grouped as follows:

| Family | Preset | Evaluator | Compute |
|---|---|---|---|
| Boosting control | `catboost` | Canonical CatBoost evaluator | CPU |
| Classical deep learning | `tabm` | TabM-mini + PLE | GPU |
| Classical deep learning | `realmlp` | RealMLP-TD through PyTabKit | GPU |
| Foundation model | `tabicl` | TabICLv2 | GPU |
| Foundation model | `tabpfn` | TabPFN v3 | GPU, non-commercial licensed weights |
| Foundation model | `tabfm` | TabFM 1.0 PyTorch | GPU, non-commercial licensed weights |
| Boosting | `lightgbm` | LightGBM 4.6 | CPU |
| Boosting | `xgboost` | XGBoost 2.1 | CPU |

The `catboost` preset resolves directly to the control implementation in
[`problems/dag_tab`](../dag_tab); no duplicate CatBoost problem directory exists.
All eight presets default to the true no-memory control:
`pipeline=guided memory=none`. Estimator choice and memory policy are
independent.

## Environment

Heavy ML libraries are intentionally kept out of the main GigaEvo environment.
The tested environment on this host is:

```bash
source ~/venvs/evo_torch/bin/activate
export GIGAEVO_TABULAR_DATA=~/tabm-data/data
```

Its tested package versions are:

| Package | Version |
|---|---:|
| PyTorch | 2.13.0 |
| TabM | 0.0.3 |
| rtdl-num-embeddings | 0.0.12 |
| PyTabKit | 1.7.3 |
| TabICL | 2.1.1 |
| TabPFN | 8.1.0 |
| TabFM | 1.0.1 |
| LightGBM | 4.6.0 |
| XGBoost | 2.1.4 |

To reproduce the isolated environment from the repository root:

```bash
uv venv --python 3.12 ~/venvs/evo_torch
source ~/venvs/evo_torch/bin/activate
uv pip install -e ".[test]"
uv pip install \
  torch==2.13.0 \
  tabm==0.0.3 \
  rtdl-num-embeddings==0.0.12 \
  pytabkit==1.7.3 \
  tabicl==2.1.1 \
  tabpfn==8.1.0 \
  'tabfm[pytorch]==1.0.1' \
  lightgbm==4.6.0 \
  xgboost==2.1.4
uv pip check
```

Normal evolution also needs the credentials used by the selected LLM configs.
The standard Gemini mutation plus Qwen memory recipe expects
`OPENAI_API_KEY`, `LOCAL_LLM_PROXY`, and `LITELLM_MASTER_KEY`.

## Running a baseline

California is the shared default. Baseline recipes live under the existing
Hydra experiment group:

```bash
# Canonical control
python run.py experiment=tabular_dag/catboost

# Classical deep learning
python run.py experiment=tabular_dag/tabm
python run.py experiment=tabular_dag/realmlp

# Foundation models
python run.py experiment=tabular_dag/tabicl
python run.py experiment=tabular_dag/tabpfn
python run.py experiment=tabular_dag/tabfm

# Boosting
python run.py experiment=tabular_dag/lightgbm
python run.py experiment=tabular_dag/xgboost
```

The preset supplies all FeatureGraph plumbing: problem path, JSON
program format, dynamic raw-feature seed, dataset-aware task context, and the
canonical structured-diff mutation operator. The eight small model presets all
inherit that one shared configuration; there is no config file per dataset.
The bare commands above use no memory reader, writer, or memory LLM.
Each problem prompt also names and explains its fixed estimator before the
shared FeatureGraph ABI. These descriptions are factual only: they contain no
estimator-specific feature advice and no validation, refit, or test mechanics.
The shared ABI and selected dataset context otherwise remain the same across
models.

For a Memory V2 treatment matching the completed California campaign, select
both the reading pipeline and memory recipe explicitly:

```bash
python run.py \
  experiment=tabular_dag/realmlp \
  llm=gemini35_flash \
  pipeline=memory_guided \
  memory=v2 \
  memory/llm=qwen_instruct \
  algorithm=tabular/2d_local_ood \
  mutation_operator.allowed_changes.max_nodes=10 \
  max_mutants=100
```

Replace only `realmlp` with another estimator preset. Other compatible memory
recipes, such as `memory=v2_multitask`, can be selected the same way. A
no-memory campaign omits the three memory-treatment overrides because
`pipeline=guided memory=none` is already the tabular default. No
`max_in_flight=4` override is required; independent evaluations can use all
available GPUs through the shared allocator.

### Selecting a dataset

Set the dataset directly. For example:

```bash
python run.py experiment=tabular_dag/xgboost problem.dataset=adult
```

`problem.dataset=california` is defined once in the shared baseline config and
is merely the default. Any dataset already supported by `problems/tabular` can
be selected the same way. The seed loader determines `x0...xN-1` dynamically,
and the mutation prompt appends the selected dataset's `TASK`, `DATASET`, and
`COLUMNS` sections at runtime.

### Choosing an output directory

Hydra creates its normal timestamped output directory unless one is supplied.
For a named research run:

```bash
python run.py \
  experiment=tabular_dag/lightgbm \
  hydra.run.dir=experiments/dag_tabular/lightgbm_california
```

Do not point concurrent runs at the same directory.
In particular, do not background several bare commands within the same second:
the default timestamp has second-level resolution. Assign a distinct
`hydra.run.dir` to each model, as the campaign launcher does.

## Model recipes

| Evaluator | Fixed recipe | Validation selection and final fit | Row weights |
|---|---|---|---|
| CatBoost | Symmetric depth-6 trees, 2,000-tree ceiling, learning rate 0.05 | Early stopping on validation; fresh train+validation refit for selected rounds | Yes |
| TabM | TabM-mini + PLE, `k=32`, California-tuned paper recipe | Early stopping on validation; fresh train+validation refit for the selected epoch count | Yes |
| RealMLP | RealMLP-TD, one ensemble member, 256-epoch ceiling | Package stopping epoch from validation; fresh train+validation refit at that epoch | No |
| TabICLv2 | Frozen 2026-02-12 checkpoint, eight estimators | No optimizer or early stopping; train+validation is labeled context | No |
| TabPFN v3 | Default classifier or official medium-data regressor checkpoint, eight estimators, automatic estimator scaling disabled | No optimizer or early stopping; train+validation is labeled context | No |
| TabFM 1.0 | Frozen PyTorch checkpoint, plain 32-estimator recipe, at most 500 features per estimator | No optimizer or early stopping; train+validation is labeled context | No |
| LightGBM | 2,000-tree ceiling, 63 leaves, learning rate 0.05 | Early stopping on validation; fresh train+validation refit for selected rounds | Yes |
| XGBoost | Histogram trees, depth 6, 2,000-tree ceiling, learning rate 0.05 | Early stopping on validation; fresh train+validation refit for selected rounds | Yes |

The recipes are intentionally model-native rather than forcing unrelated models
to share hyperparameters. Every evaluation artifact records the selected
estimator and its full fixed configuration.

More detailed model notes and supported environment overrides live in each
model directory:

- [`tabm`](tabm/README.md)
- [`realmlp`](realmlp/README.md)
- [`tabicl`](tabicl/README.md)
- [`tabpfn`](tabpfn/README.md)
- [`tabfm`](tabfm/README.md)
- [`lightgbm`](lightgbm/README.md)
- [`xgboost`](xgboost/README.md)

### Verified California neutral scores

These raw-feature checks used the production three-fold evaluator in
`evo_torch` on 2026-07-23. They are runtime sanity checks, not claims about the
eventual evolved winners.

| Evaluator | Fitness / R² | CV fold SD | RMSE | Validation wall time |
|---|---:|---:|---:|---:|
| RealMLP-TD | 0.823277 | 0.000716 | 0.487801 | 247.0 s |
| TabICLv2 | 0.872979 | 0.001994 | 0.413508 | 6.8 s |
| TabPFN v3 medium-data | 0.881001 | 0.000379 | 0.400275 | 10.1 s |
| LightGBM 4.6 | 0.845503 | 0.003358 | 0.456065 | 10.8 s |
| XGBoost 2.1 | 0.840245 | 0.002758 | 0.463810 | 10.3 s |

RealMLP is consequently the throughput bottleneck in the current campaign;
the randomized GPU allocator prevents that slower run from pinning a specific
GPU while the foundation-model runs are active.

## Evaluation and test protocol

For TabM, RealMLP, LightGBM, and XGBoost, one evolutionary fold works as
follows:

1. Fit the FeatureGraph and estimator on the fold fitting rows.
2. Use only the stored validation rows to select an epoch or boosting-round
   count.
3. Discard the search model.
4. Rebuild the graph, preprocessing, and estimator from scratch on fitting rows
   plus validation rows at the selected fixed training length.
5. Predict the held-out evolutionary query.

TabICL, TabPFN, and TabFM are frozen in-context models. They have no
dataset-specific optimizer or early stopping, so fitting rows plus validation
rows form their labeled context directly.

Final test scoring repeats the corresponding clean fit once with
`X_train + X_val`, predicts `X_test`, and only then reads test labels to compute
the canonical task metrics. It never reuses an evolutionary model. Evolutionary
fitness and its CV standard deviation are returned by the unchanged
`problems/tabular` protocol.

### Score a saved graph on the untouched test split

Use the module matching the evaluator that should consume the frozen graph:

```bash
python -m problems.dag_tab.test program.json
python -m problems.tabular_dag_baselines.tabm.test program.json
python -m problems.tabular_dag_baselines.realmlp.test program.json
python -m problems.tabular_dag_baselines.tabicl.test program.json
python -m problems.tabular_dag_baselines.tabpfn.test program.json
python -m problems.tabular_dag_baselines.tabfm.test program.json
python -m problems.tabular_dag_baselines.lightgbm.test program.json
python -m problems.tabular_dag_baselines.xgboost.test program.json
```

Running several of these commands on the same JSON graph gives a direct
cross-evaluator comparison without evolving a new feature set.

## Two-phase experiment: evolve, then compare

Phase 1 evolves a separate FeatureGraph against each fixed estimator using the
short commands above. Freeze each selected winner JSON before looking at the
test split.

Phase 2 evaluates frozen graphs under other estimators. For one cell and one
model seed:

```bash
python -m problems.tabular_dag_baselines.compare \
  --evaluator tabpfn --graph winner.json --phase test --seed 0
```

For a graph-by-estimator matrix, the matrix command defaults to seeds 0–4,
reports the sample standard deviation (`ddof=1`), and runs up to four cells at
once:

```bash
python -m problems.tabular_dag_baselines.compare_matrix \
  --graph catboost=catboost-winner.json \
  --graph tabpfn=tabpfn-winner.json \
  --evaluator catboost --evaluator tabpfn \
  --output cross_eval.json
```

Repeat `--graph NAME=PATH` and `--evaluator MODEL` to extend the matrix. The
graph JSON and dataset split stay fixed within a cell; only the evaluator seed
changes. Use the default `--phase test` only for a preregistered finalist panel.
Use `--phase cv` while developing or screening so the untouched test split
does not become a search signal.

## GPU allocation

TabM, RealMLP, TabICL, TabPFN, and TabFM lease one randomly selected visible GPU
for an entire evaluation. Cross-process file locks are shared across all five
models, so concurrent runs do not silently select the same physical GPU. With
the four GPUs used by the standard campaign, no launch override is needed.

To restrict the shared pool explicitly:

```bash
export GIGAEVO_TABULAR_DAG_GPU_DEVICES=0,1,2,3
```

`CUDA_VISIBLE_DEVICES` is respected. Shared controls are
`GIGAEVO_TABULAR_DAG_DEVICE`, `GIGAEVO_TABULAR_DAG_GPU_DEVICES`,
`GIGAEVO_TABULAR_DAG_GPU_LOCK_DIR`, and
`GIGAEVO_TABULAR_DAG_GPU_LOCK_TIMEOUT`. A model-specific prefix such as
`GIGAEVO_REALMLP_` takes precedence. LightGBM and XGBoost use four CPU threads
by default and do not take GPU leases.

## Foundation-model checkpoints

TabICL downloads its public checkpoint on first real fit and caches it.
`GIGAEVO_TABICL_MODEL_PATH` can point to a pre-downloaded checkpoint.

TabPFN v3 checkpoints are downloaded directly from
[`Prior-Labs/tabpfn_3`](https://huggingface.co/Prior-Labs/tabpfn_3) on first use.
Downloading and using them is subject to the repository's non-commercial
license. If Hugging Face requires authentication for the repository, set an
account token first:

```bash
export HF_TOKEN=...
python run.py experiment=tabular_dag/tabpfn
```

The preflight resolves the exact task-specific checkpoint before taking a GPU.
If direct Hugging Face download fails, TabPFN's `TABPFN_TOKEN` flow is retained
as a fallback. An opposite-task checkpoint does not satisfy the check.
California uses the official medium-data v3 regressor because the final labeled
context contains 16,512 rows; pretraining-limit checks stay enabled. An
existing v3 checkpoint may instead be supplied through
`GIGAEVO_TABPFN_MODEL_PATH`.

TabFM downloads only the selected task subfolder from the pinned
[`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
revision before taking a GPU. Each task checkpoint is about 6.6 GB. The weights
use the TabFM Non-Commercial License; check that license before use. The model
is loaded once per evaluation, shared across its CV folds and behavior probes,
and dropped before the GPU lease is released. Set `GIGAEVO_TABFM_MODEL_PATH` to
a complete local checkpoint directory to run without downloading.

## Verification

Run the focused suite in `evo_torch`:

```bash
python -m pytest tests/tabular_dag_baselines tests/dag_tabm -q
```

The suite covers Hydra composition and runtime construction for all eight
presets, shared train/validation/test routing, randomized GPU locks, real
LightGBM and XGBoost fits, the RealMLP search/refit protocol, TabM preprocessing
and training, comparison-result aggregation, and checkpoint-free
TabICL/TabPFN/TabFM adapter checks. Tests that require optional research
packages skip cleanly in the normal lightweight environment.
