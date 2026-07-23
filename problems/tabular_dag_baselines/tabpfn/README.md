# TabPFN v3 FeatureGraph baseline

TabPFN v3 is a pretrained tabular foundation model: labeled rows condition a
frozen transformer whose synthetic-task prior is used for prediction.

This evaluator uses TabPFN 8.1.0 with eight inference estimators, automatic
estimator-count scaling disabled, fit-time preprocessor caching, and seed 0.
Classification uses the official default v3 checkpoint; regression uses the
official `20260417_mediumdata` v3 checkpoint so California's 16,512-row final
context remains within the intended regime. Pretraining-limit checks remain
enabled. There is no early stopping: the final labeled context is
train+validation. Row weights are rejected.

```bash
python run.py \
  experiment=tabular_dag/tabpfn \
  llm=gemini35_flash \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

This is the no-memory baseline. To run Memory V2, also add
`pipeline=memory_guided memory=v2 memory/llm=qwen_instruct`.

The adapter downloads the selected checkpoint directly from
[`Prior-Labs/tabpfn_3`](https://huggingface.co/Prior-Labs/tabpfn_3), subject to
its non-commercial license, and resolves it before taking a GPU. An `HF_TOKEN`
can be supplied if the repository requires authentication; TabPFN's
`TABPFN_TOKEN` flow remains a fallback. An opposite-task cache does not pass
the check. A pre-downloaded v3 checkpoint can be set with
`GIGAEVO_TABPFN_MODEL_PATH`. It never falls back to another TabPFN version.

```bash
python -m problems.tabular_dag_baselines.tabpfn.test program.json
```

See the [suite README](../README.md) for protocol, GPU, and frozen-graph
comparison details.
