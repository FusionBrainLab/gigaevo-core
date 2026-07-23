# RealMLP FeatureGraph baseline

RealMLP-TD is a tuned, regularized tabular MLP recipe; it is trained on each
dataset and is not a pretrained foundation model.

This evaluator uses PyTabKit 1.7.3's RealMLP-TD recipe with one ensemble member,
a 256-epoch ceiling, batch size 256, and seed 0. Numeric missing values are
median-imputed from the current fitting rows; categorical vocabularies are also
fit-local. During search, validation-only levels map to a training-observed
fallback so PyTabKit cannot use validation to enlarge its categorical embedding
architecture. RealMLP selects its stopping epoch on validation, then a new graph,
preprocessor, and model fit on train+validation at that package-native stopping
epoch. Row weights are rejected rather than silently ignored.

```bash
python run.py \
  experiment=tabular_dag/realmlp \
  llm=gemini35_flash \
  memory/llm=qwen_instruct \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

```bash
python -m problems.tabular_dag_baselines.realmlp.test program.json
```

See the [suite README](../README.md) for the exact CV/test protocol and shared
GPU allocation, including frozen-graph comparison commands. Environment
overrides use the `GIGAEVO_REALMLP_` prefix.
