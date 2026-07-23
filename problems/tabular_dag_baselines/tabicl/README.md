# TabICLv2 FeatureGraph baseline

TabICLv2 is a pretrained transformer that treats labeled rows as an inference
context; its checkpoint predicts new rows without dataset-specific gradients.

This evaluator uses the frozen `tabicl-*-v2-20260212.ckpt` checkpoint with eight
inference estimators, batch size 8, and seed 0. There is no early stopping or
dataset-specific training: the graph and category vocabulary fit on the labeled
context, and TabICL performs in-context inference. Final test context is
train+validation. Row weights are rejected.

```bash
python run.py \
  experiment=tabular_dag/tabicl \
  llm=gemini35_flash \
  memory/llm=qwen_instruct \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

The public checkpoint downloads on the first real fit and is cached by
Hugging Face. `GIGAEVO_TABICL_MODEL_PATH` can point to a pre-downloaded file.

```bash
python -m problems.tabular_dag_baselines.tabicl.test program.json
```

See the [suite README](../README.md) for protocol, GPU, and frozen-graph
comparison details.
