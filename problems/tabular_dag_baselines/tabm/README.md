# TabM FeatureGraph baseline

TabM is a parameter-efficient MLP ensemble that jointly produces several
predictions per row instead of training many fully independent networks.

This is the TabM estimator variant in the shared
[`tabular_dag_baselines`](../README.md) suite. It replaces only CatBoost at the
estimator boundary; graph execution, leakage probes, data splits, CV metrics,
and final test scoring remain canonical.

The fixed California recipe is TabM-mini with `k=32`, three 576-wide blocks,
dropout 0.2405049535, noisy train-local quantile normalization, 30-bin
16-dimensional piecewise-linear embeddings, AdamW at 2.992624e-4, batch size
256, gradient clipping at 1, and independent ensemble-member batches.
Validation patience is 16. The search model is discarded and a fresh model is
trained on train+validation for the selected epoch count. CUDA autocast uses
BF16 when supported (including these H100s), with scaled FP16 as a fallback.

```bash
python run.py \
  experiment=tabular_dag/tabm \
  llm=gemini35_flash \
  memory/llm=qwen_instruct \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

```bash
python -m problems.tabular_dag_baselines.tabm.test program.json
```

The default is the fair final protocol. For explicitly labeled screening runs,
`GIGAEVO_TABM_K=8`, `GIGAEVO_TABM_SHARE_TRAINING_BATCHES=true`, and
`GIGAEVO_TABM_REFIT=false` reduce cost, but finalists must be reranked with
those variables unset.

On 2026-07-22, one H100 California BF16 search-only fit took 29.5 seconds,
selected 232 epochs, and gave test RMSE 0.4306. The complete
search-plus-train+validation-refit path took 56.1 seconds, gave test RMSE
0.4222, and peaked at 0.50 GiB allocated GPU memory. These are local timing
measurements, not promised throughput for every evolved graph.
