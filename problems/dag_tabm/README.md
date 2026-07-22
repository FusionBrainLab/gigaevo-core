# DAG TabM: FeatureGraph evolution with a fixed TabM evaluator

`dag_tabm` is the controlled TabM counterpart of `problems/dag_tab`. It reuses
the exact FeatureGraph schema, execution and leakage probes, dynamic dataset
seed, shared tabular folds, metrics, behavior descriptors, and untouched-test
protocol. Only the estimator boundary changes from CatBoost to TabM.

The default estimator is the paper's tuned California configuration:

- TabM-mini with `k=32`;
- three 576-wide blocks and dropout `0.2405049535`;
- noisy fold-local quantile normalization;
- 30-bin, 16-dimensional piecewise-linear numerical embeddings;
- AdamW at `2.992624e-4`, batch size 256, gradient clipping at 1;
- independent batches for ensemble members;
- validation patience 16, followed by train+validation refit for the selected
  epoch count.

This is a California-tuned fixed evaluator. Regression, binary-classification,
and multiclass paths have been smoke-tested on California, Adult, and Otto,
respectively, but model hyperparameters have not been tuned outside California.

## Environment

```bash
uv venv --python 3.12 ~/venvs/evo_torch
source ~/venvs/evo_torch/bin/activate
uv pip install -e ".[tabm-eval,dev,test]"
export GIGAEVO_TABULAR_DATA=~/tabm-data/data
```

The evaluator lazily imports Torch, `tabm`, and `rtdl-num-embeddings`, so the
ordinary CatBoost problem remains usable without these dependencies. The
dedicated extra does not install `vllm`.

The launch recipe below also needs `OPENAI_API_KEY` for Gemini/OpenRouter and
`LOCAL_LLM_PROXY` plus `LITELLM_MASTER_KEY` for Qwen Instruct. If the shell uses
an outbound HTTP proxy, include the local proxy host in both `NO_PROXY` and
`no_proxy`.

## GPU allocation

Each candidate leases one randomly selected visible GPU for its entire CV and
behavior-descriptor evaluation. Advisory file locks prevent concurrent workers
from selecting the same device, and are released automatically if a worker
exits. With all four local GPUs visible, no extra configuration is required.

Useful controls:

```bash
# Restrict the random pool to logical CUDA devices 0, 2, and 3.
export GIGAEVO_TABM_GPU_DEVICES=0,2,3

# Pin instead of randomizing (mainly for debugging).
export GIGAEVO_TABM_DEVICE=cuda:1

# Override the shared lock location when workers do not share /tmp.
export GIGAEVO_TABM_GPU_LOCK_DIR=/shared/path/tabm-gpu-locks
```

`CUDA_VISIBLE_DEVICES` is respected. Lock identities use its physical tokens,
so workers with the same visible-device list coordinate correctly.

## Launch

With Memory V2, Gemini 3.5 Flash mutations, and Qwen Instruct memory:

```bash
python run.py \
  problem.name=dag_tabm \
  problem.dataset=california \
  loader=dag_tabm_seed \
  program_format=json_document \
  mutation=structured_diff_dag_tabm \
  memory/llm=qwen_instruct \
  llm=gemini35_flash \
  algorithm=tabular/2d_local_ood \
  mutation_operator.allowed_changes.max_nodes=10 \
  max_mutants=100
```

Score a frozen graph on the untouched test split with:

```bash
python -m problems.dag_tabm.test /path/to/program.json
```

## Runtime profiles

The full defaults preserve the CatBoost problem's search-then-refit contract.
One candidate therefore performs three CV `fit_predict` calls plus one behavior
descriptor call, and every call contains an early-stopped search fit and a
fixed-epoch refit: eight neural-network fits in total.

For an inexpensive screening evolution, use a smaller ensemble and omit the
refit, then rerank finalists under the full defaults:

```bash
export GIGAEVO_TABM_K=8
export GIGAEVO_TABM_SHARE_TRAINING_BATCHES=true
export GIGAEVO_TABM_REFIT=false
export GIGAEVO_TABM_MAX_EPOCHS=256
```

Every override is explicit and recorded in the evaluation artifact where
relevant. Full comparisons should restore `k=32`, independent batches, and
refitting.

Additional controls include `GIGAEVO_TABM_N_BLOCKS`, `D_BLOCK`, `DROPOUT`,
`LEARNING_RATE`, `WEIGHT_DECAY`, `N_BINS`, `D_EMBEDDING`, `BATCH_SIZE`,
`PATIENCE`, `MAX_EPOCHS`, `AMP`, `SEED`, and `EVAL_BATCH_SIZE` (all prefixed by
`GIGAEVO_TABM_`).

## Evaluation protocol parity

There is no TabM-specific split or scoring implementation. Both `dag_tab` and
`dag_tabm` delegate to the same shared tabular protocol: identical deterministic
folds, fixed validation data for early stopping, fold metrics, behavior
descriptors, optional mean/LCB fitness aggregation, and untouched-test route.
`cv_score_std` is the sample standard deviation of the same fold scores
(`ddof=1`), and its `_evaluation_measurements` entry reports the same value and
fold count. Only the fitted estimator and its internal preprocessing differ.

## Local baseline timing

Measured on 2026-07-22 on one NVIDIA H100 80GB with Torch 2.13/CUDA 13 and the
default BF16 path, the raw California TabM-mini+PLE search selected 232 epochs
and stopped after 249. The actual search-only `fit_predict` took 29.5 seconds,
reached test RMSE 0.4306, and peaked at 0.50 GiB allocated GPU memory.

The complete search-plus-train+validation-refit path took 56.1 seconds, reached
test RMSE 0.4222, and also peaked at 0.50 GiB. A candidate has four such calls
(three CV folds plus one behavior-descriptor fit), so approximately 3.7
GPU-minutes remains an extrapolated candidate-level estimate rather than a
measured full-candidate run. Four leased GPUs should sustain approximately one
candidate per minute once the pipeline is full.
