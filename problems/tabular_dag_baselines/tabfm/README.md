# TabFM 1.0 FeatureGraph baseline

TabFM is Google's pretrained in-context foundation model for numerical and
categorical tables. This evaluator uses `tabfm==1.0.1` with the released
TabFM 1.0 PyTorch weights, bfloat16 compute, the official plain 32-estimator
recipe, batch size 1, all context rows, at most 500 features per estimator, and
seed 0. It does not use the separate, heavier TabFM-Ensemble recipe.

The FeatureGraph is fitted on each labeled context. TabFM then receives
train+validation as in-context examples and predicts only the held-out query.
There is no optimizer or early stopping, classification supports at most ten
classes, and row weights are rejected.

```bash
python run.py experiment=tabular_dag/tabfm
```

The public checkpoint is downloaded task-specifically from
[`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch)
at a pinned revision before a GPU is leased. The checkpoint weights use the
TabFM Non-Commercial License; the Python source is Apache-2.0. A local parent
directory containing `regression/` and/or `classification/`, or a selected
task directory itself, can be supplied with `GIGAEVO_TABFM_MODEL_PATH`.

The model is loaded once per validation or test evaluation, shared across its
folds, then released before the shared GPU lock. Optional fixed-recipe
overrides are `GIGAEVO_TABFM_N_ESTIMATORS`, `GIGAEVO_TABFM_BATCH_SIZE`,
`GIGAEVO_TABFM_MAX_NUM_FEATURES`, `GIGAEVO_TABFM_MAX_NUM_ROWS`, and
`GIGAEVO_TABFM_SEED`.

```bash
python -m problems.tabular_dag_baselines.tabfm.test program.json
```

See the [suite README](../README.md) for installation, protocol, GPU, and
frozen-graph comparison details.
