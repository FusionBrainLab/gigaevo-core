# LightGBM FeatureGraph baseline

This evaluator uses LightGBM 4.6.0 with learning rate 0.05, 63 leaves, a
2000-tree ceiling, 50-round early stopping, four CPU threads, and seed 0.
Categorical features use train-local pandas categories; numerical NaNs remain
native missing values. A new graph and model refit on train+validation for the
selected round count. Valid generated row weights are supported.

```bash
python run.py \
  experiment=tabular_dag/lightgbm \
  llm=gemini35_flash \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

This is the no-memory baseline. To run Memory V2, also add
`pipeline=memory_guided memory=v2 memory/llm=qwen_instruct`.

```bash
python -m problems.tabular_dag_baselines.lightgbm.test program.json
```

See the [suite README](../README.md) for the exact shared protocol. Overrides
use the `GIGAEVO_LIGHTGBM_` prefix.
