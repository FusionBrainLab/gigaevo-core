# TabM FeatureGraph baseline

TabM is a parameter-efficient MLP ensemble that jointly produces several
predictions per row instead of training many fully independent networks.

This is the TabM estimator variant in the shared
[`tabular_dag_baselines`](../README.md) suite. It replaces only CatBoost at the
estimator boundary; graph execution, leakage probes, data splits, CV metrics,
and final test scoring remain canonical.

The default is one dataset-independent recipe copied from the official TabM
[`example.ipynb`](https://github.com/yandex-research/tabm/blob/main/example.ipynb):
full TabM with `k=32`, two 512-wide blocks, dropout 0.1,
noisy train-local quantile normalization, 48-bin 16-dimensional piecewise-linear
embeddings, AdamW at learning rate 0.002 and weight decay 0.0003, batch size
256, gradient clipping at 1, and the recommended shared training batches.
Validation patience is 16. The adapter adds a fixed 512-epoch safety ceiling;
the upstream example uses an effectively unbounded loop governed only by
patience. The search model is discarded and a fresh model is trained on
train+validation for the selected epoch count. CUDA autocast uses BF16 when
supported (including these H100s), with scaled FP16 as a fallback.

No architecture or optimizer setting is selected per dataset. Input and output
dimensions, categorical cardinalities, train-local quantile statistics and PLE
bin boundaries necessarily follow the data; validation selects only the
training length. Explicit `GIGAEVO_TABM_*` overrides remain available for
reproducing named paper configurations, but such runs must be labeled as tuned
rather than compared as the default baseline.

This also follows the package's
[`arch_type` guidance](https://github.com/yandex-research/tabm#arch_type):
full `tabm` is the recommended default, whereas `tabm-mini` can need more
careful width/depth selection for a fixed `k`.

```bash
python run.py \
  experiment=tabular_dag/tabm \
  llm=gemini35_flash \
  algorithm=tabular/2d_local_ood \
  max_mutants=100
```

This is the no-memory baseline. To run Memory V2, also add
`pipeline=memory_guided memory=v2 memory/llm=qwen_instruct`.

```bash
python -m problems.tabular_dag_baselines.tabm.test program.json
```

The default is the fair final protocol. For explicitly labeled screening runs,
`GIGAEVO_TABM_K=8` and `GIGAEVO_TABM_REFIT=false` reduce cost, but finalists
must be reranked with those variables unset.

On 2026-07-23, the new fixed default completed the production California
three-fold raw-feature CV in 49.5 seconds: R² 0.839009, fold SD 0.006085, and
RMSE 0.464086. This was a post-selection runtime sanity check of the
repository-derived recipe, not dataset-specific hyperparameter selection.

The historical measurements below used the old paper-tuned California
TabM-mini recipe; they are retained only as provenance and do not benchmark the
new fixed default. On 2026-07-22, one H100 California BF16 search-only fit took
29.5 seconds, selected 232 epochs, and gave test RMSE 0.4306. The complete
search-plus-train+validation-refit path took 56.1 seconds, gave test RMSE
0.4222, and peaked at 0.50 GiB allocated GPU memory. These are local timing
measurements, not promised throughput for every evolved graph.
