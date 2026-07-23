# XGBoost FeatureGraph baseline

This evaluator uses XGBoost 2.1.4 histogram trees with learning rate 0.05,
depth 6, a 2000-tree ceiling, 50-round early stopping, four CPU threads, and
seed 0. Categorical features use native handling with train-local pandas
categories; numerical NaNs remain native missing values. A new graph and model
refit on train+validation for the selected rounds. Generated row weights are
supported.

```bash
python run.py \
  experiment=tabular_dag/xgboost \
  llm=gemini35_flash \
  memory/llm=qwen_instruct \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

```bash
python -m problems.tabular_dag_baselines.xgboost.test program.json
```

See the [suite README](../README.md) for the exact shared protocol. Overrides
use the `GIGAEVO_XGBOOST_` prefix.
