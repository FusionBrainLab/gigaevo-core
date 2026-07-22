# Tabular problem suite

A mixed regression + classification tabular suite. All datasets share the
machinery in `_common/`; each dataset dir holds one real `task_description.txt`
plus symlinks to the shared `validate.py`, `score_test.py`, the task-type
`metrics.yaml`, and the seed set.

## Data setup (out-of-band, gitignored)

Data lives outside the repo, under `$GIGAEVO_TABULAR_DATA`:

```bash
wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar
tar xf data.tar -C /some/dir            # creates /some/dir/data/<dataset>/...
export GIGAEVO_TABULAR_DATA=/some/dir/data
```

Each dataset folder (tabm format) holds `info.json`, `X_num_{train,val,test}.npy`
(always), optional `X_cat_*`/`X_bin_*`, and `Y_{train,val,test}.npy`.

TabReD 0.1.x uses a different native layout. Import its official default
temporal splits into the same data root with:

```bash
uvx tabred download all --output-path /some/dir/tabred-native
python problems/tabular/_common/import_tabred.py /some/dir/tabred-native /some/dir/data
```

The TabArena leaderboard datasets are the public OpenML suite
`tabarena-v0.1`. Download its official tasks and splits, then import the
TabArena-Lite r0f0 view: repeat 0 fold 2 is train, fold 1 is validation, and
fold 0 is the untouched test set. Train plus validation therefore exactly
matches the official r0f0 training set.

Evolution validation merges train and validation, then rotates the configured
deterministic folds through fit, early-stopping validation, and scored-query
roles.

```bash
uvx --from openml python problems/tabular/_common/import_tabarena.py /some/dir/data \
  --openml-cache /some/dir/tabarena-openml
GIGAEVO_TABULAR_DATA=/some/dir/data \
  python problems/tabular/_common/gen_task_descriptions.py --prefix tabarena-
```

Imported IDs are prefixed `tabarena-`. Full TabArena scores aggregate 9–30
outer splits per dataset; these single-view datasets must be reported as
`TabArena-Lite-r0f0`. Comparable final metrics are RMSE (regression), ROC-AUC
(binary), and log-loss (multiclass).

## Datasets

| problem.name          | task        | features (num/cat/bin) | classes | eval   |
|-----------------------|-------------|------------------------|---------|--------|
| tabular/california    | regression  | 8/0/0                  | —       | CV     |
| tabular/house         | regression  | 16/0/0                 | —       | CV     |
| tabular/diamond       | regression  | 6/3/0                  | —       | CV     |
| tabular/black-friday  | regression  | 4/4/1                  | —       | holdout|
| tabular/microsoft     | regression  | 131/0/5                | —       | holdout|
| tabular/adult         | binclass    | 6/7/1                  | 2       | CV     |
| tabular/churn         | binclass    | 7/1/3                  | 2       | CV     |
| tabular/higgs-small   | binclass    | 28/0/0                 | 2       | CV     |
| tabular/otto          | multiclass  | 93/0/0                 | 9       | CV     |
| tabular/covtype2      | multiclass  | 10/1/4                 | 7       | holdout|

CV uses `GIGAEVO_TABULAR_CV_FOLDS` (default 3) when the combined train and
validation size is at most `GIGAEVO_TABULAR_CV_MAX` (default 100000), else a
single 80/20 holdout.

## Program contract

```python
class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query) -> np.ndarray: ...

def entrypoint() -> type:
    return Model
```

- `X` is one float64 matrix `[X_num | X_bin | X_cat]`; categoricals are integer
  codes (see each `task_description.txt` for the per-column value vocabulary).
- regression → return 1D float; classification → return `(n_query, n_classes)`
  probabilities (column j = P(class j)), or 1D int labels (BD/AUC/log-loss degrade).

## Fitness (scale-free)

- regression: mean per-fold R² clamped at −1 (`[-1, 1]`, 1.0 = perfect).
- classification: mean per-fold accuracy (`[0, 1]`).

Robustness archive axes `local_lipschitz_p95 × ood_delta_slope` are reported but
NOT optimised (`include_in_prompts: false`).

## Run

```bash
GIGAEVO_TABULAR_DATA=/some/dir/data \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python run.py problem.name=tabular/california algorithm=tabular/2d_local_ood
```

For example, replace the problem with
`problem.name=tabular/tabarena-airfoil-self-noise` to run its r0f0 view.

End-of-evolution test scoring (never a search signal):

```bash
GIGAEVO_TABULAR_DATA=/some/dir/data \
python problems/tabular/california/score_test.py problems/tabular/california/initial_programs/prog1.py
```

## Materializing problem directories

Problem wrappers and `task_description.txt` files are generated from the data:

```bash
GIGAEVO_TABULAR_DATA=/some/dir/data \
python problems/tabular/_common/gen_task_descriptions.py
```

Use `--prefix tabarena-` for all imported TabArena datasets.

(`california`'s numeric feature names are hand-added on top of the generated block.)
