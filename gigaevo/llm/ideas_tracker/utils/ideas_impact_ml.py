"""
Impact analysis pipeline for IdeaTracker.

Estimates the reliable impact of individual ideas — and optionally their
pairwise interactions — on program fitness, using regularized linear models
(ElasticNet) with bootstrap confidence intervals.

Designed for robustness under small-sample / high-dimensional regimes where
overfitting and spurious correlations are the primary risks.

Known constraints addressed
----------------------------
* **Few data points**  — adaptive CV folds (LOO for n <= 10), bootstrap
  iteration count scaled to sample size, aggressive L1 regularisation.
* **Feature interactions** — selective pairwise interaction terms are added
  only when they co-occur frequently enough and correlate with the target,
  keeping dimensionality under control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.utils import resample
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ImpactResult:
    """Structured result of the impact analysis pipeline.

    Attributes
    ----------
    summary : pd.DataFrame
        Per-idea impact summary.  When ``idea_ids`` are provided the
        index contains RecordCard UUIDs and a ``Description`` column
        holds the human-readable text.  Columns: ``Description``
        (when IDs are used), ``Mean_Weight``, ``Std_Dev``, ``CI_Low``,
        ``CI_High``, ``Is_Reliable``, ``Frequency``.
    interactions : pd.DataFrame | None
        Same schema as *summary* but for pairwise interaction terms.
        ``None`` when interactions are disabled or not detected.
    metadata : dict[str, Any]
        Pipeline diagnostics — sample/feature counts, CV folds used,
        and any warnings raised.
    """

    summary: pd.DataFrame
    interactions: pd.DataFrame | None
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(
    programs: list[Any],
    ideas: list[Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build a binary feature matrix and fitness target from IdeaTracker data.

    Parameters
    ----------
    programs : list
        ``ProgramRecord`` instances (need ``.id`` and ``.fitness``).
    ideas : list
        ``RecordCard`` instances (need ``.id``, ``.description``,
        ``.linked_programs``).

    Returns
    -------
    X : np.ndarray, shape (n_programs, n_ideas)
        Binary indicator — 1 if the idea was used by the program.
    y : np.ndarray, shape (n_programs,)
        Fitness values.
    feature_names : list[str]
        Idea descriptions (used as human-readable column labels).
    idea_ids : list[str]
        Idea UUIDs (for downstream mapping back to IdeaTracker).
    """
    idea_ids = [idea.id for idea in ideas]
    feature_names = [idea.description for idea in ideas]
    n_programs = len(programs)
    n_ideas = len(ideas)

    X = np.zeros((n_programs, n_ideas), dtype=np.float64)
    y = np.zeros(n_programs, dtype=np.float64)

    prog_id_to_row: dict[str, int] = {prog.id: row for row, prog in enumerate(programs)}

    for col, idea in enumerate(ideas):
        for linked_id in idea.linked_programs:
            row = prog_id_to_row.get(linked_id)
            if row is not None:
                X[row, col] = 1.0

    for row, prog in enumerate(programs):
        y[row] = prog.fitness

    return X, y, feature_names, idea_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _adaptive_cv(n_samples: int) -> int:
    """Choose the number of CV folds based on dataset size.

    * n <= 10  → leave-one-out (= n folds)
    * n <= 30  → min(5, n)
    * n > 30   → 5
    """
    if n_samples <= 10:
        return max(n_samples, 2)  # need at least 2 folds
    if n_samples <= 30:
        return min(5, n_samples)
    return 5


def _filter_features(
    X: np.ndarray,
    feature_names: list[str],
    idea_ids: list[str],
    min_pos: int = 2,
    min_neg: int = 2,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """Remove features lacking variation.

    A binary feature present in fewer than *min_pos* samples (or absent from
    fewer than *min_neg* samples) carries too little signal and only inflates
    dimensionality.

    Returns
    -------
    X_filtered, filtered_names, filtered_ids, keep_mask
    """
    n = X.shape[0]
    col_sums = X.sum(axis=0)
    keep = (col_sums >= min_pos) & (col_sums <= n - min_neg)
    return (
        X[:, keep],
        [name for name, k in zip(feature_names, keep) if k],
        [iid for iid, k in zip(idea_ids, keep) if k],
        np.asarray(keep),
    )


def _select_interactions(
    X: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    max_pairs: int = 10,
    min_cooccurrence: int = 3,
    idea_ids: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str] | None]:
    """Add the most promising pairwise interaction columns to *X*.

    Strategy (to avoid the O(p²) explosion):

    1. Only consider pairs that co-occur in >= *min_cooccurrence* samples
       **and** have enough non-co-occurrence samples.
    2. Rank candidates by ``|corr(x_i · x_j, y)|``.
    3. Keep at most *max_pairs*, capped so that total feature count stays
       below ``n_samples / 2`` (to leave room for regularisation).

    Returns
    -------
    X_augmented, augmented_feature_names, augmented_ids (or None)
    """
    n_samples, n_features = X.shape
    if n_features < 2:
        return X, feature_names, idea_ids

    # Budget: total features must stay well below n to avoid rank deficiency.
    max_total = max(n_samples // 2, n_features + 1)
    budget = min(max_pairs, max_total - n_features)
    if budget <= 0:
        return X, feature_names, idea_ids

    y_centered = y - y.mean()
    y_norm = float(np.linalg.norm(y_centered))
    if y_norm < 1e-12:
        return X, feature_names, idea_ids

    candidates: list[tuple[int, int, float, np.ndarray]] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            product: np.ndarray = X[:, i] * X[:, j]
            cooccurrence = int(product.sum())
            if cooccurrence < min_cooccurrence:
                continue
            # Need enough variation — at least some without co-occurrence
            if cooccurrence >= n_samples - 1:
                continue
            prod_centered = product - product.mean()
            prod_norm = float(np.linalg.norm(prod_centered))
            if prod_norm < 1e-12:
                continue
            corr: float = abs(
                float(np.dot(prod_centered, y_centered)) / (prod_norm * y_norm)
            )
            candidates.append((i, j, corr, product))
    if not candidates:
        return X, feature_names, idea_ids

    candidates.sort(key=lambda c: c[2], reverse=True)
    selected = candidates[:budget]

    interaction_cols = np.column_stack([c[3] for c in selected])
    interaction_names = [
        f"{feature_names[c[0]]} × {feature_names[c[1]]}" for c in selected
    ]
    augmented_ids: list[str] | None = None
    if idea_ids is not None:
        interaction_ids = [f"{idea_ids[c[0]]} × {idea_ids[c[1]]}" for c in selected]
        augmented_ids = idea_ids + interaction_ids
    return (
        np.hstack([X, interaction_cols]),
        feature_names + interaction_names,
        augmented_ids,
    )


def _bootstrap_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int,
    cv_folds: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Run bootstrap ElasticNetCV and collect coefficient vectors.

    Returns
    -------
    coefs : np.ndarray, shape (n_iterations, n_features)
    """
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    n_features = X.shape[1]
    coefs = np.zeros((n_iterations, n_features))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for i in tqdm(
            range(n_iterations),
            desc="Bootstrap impact analysis",
            leave=False,
        ):
            seed = int(rng.randint(0, 2**31))
            resampled = resample(X, y, random_state=seed)
            X_boot: np.ndarray = resampled[0]  # type: ignore[index]
            y_boot: np.ndarray = resampled[1]  # type: ignore[index]
            # Adaptive: ensure cv <= available samples in resample
            effective_cv = min(cv_folds, int(X_boot.shape[0]))
            effective_cv = max(effective_cv, 2)
            model = ElasticNetCV(
                l1_ratio=l1_ratios,  # type: ignore[arg-type]
                cv=effective_cv,
                max_iter=10_000,
                random_state=seed,
            )
            model.fit(X_boot, y_boot)
            coefs[i] = model.coef_

    return coefs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

_Z_SCORES: dict[float, float] = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def impact_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    idea_ids: list[str] | None = None,
    n_iterations: int = 200,
    include_interactions: bool = True,
    max_interaction_pairs: int = 10,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> ImpactResult:
    """Core impact estimation on a pre-built feature matrix.

    Steps
    -----
    1. Standardise the target (zero-mean, unit-variance) for stable
       regularisation, then rescale coefficients back afterwards.
    2. *(optional)* Detect and append pairwise interaction features.
    3. Run bootstrap ElasticNetCV to produce a distribution of
       coefficient vectors.
    4. Summarise: mean, std, CI, reliability flag, selection frequency.

    Parameters
    ----------
    X : np.ndarray
        Binary feature matrix (n_samples, n_features).
    y : np.ndarray
        Continuous fitness target (n_samples,).
    feature_names : list[str]
        Human-readable label for each column of *X*.
    idea_ids : list[str] | None
        RecordCard UUIDs for each column. When provided the summary
        DataFrame is indexed by these IDs and descriptions become a
        separate ``Description`` column.  When ``None`` descriptions
        are used as the index (legacy behaviour).
    n_iterations : int
        Bootstrap repetitions (higher → tighter CIs, slower).
    include_interactions : bool
        Whether to detect and include pairwise interaction terms.
    max_interaction_pairs : int
        Maximum number of interaction columns to add.
    confidence_level : float
        Width of the confidence interval (0.90 / 0.95 / 0.99).
    random_state : int
        Seed for full reproducibility.

    Returns
    -------
    ImpactResult
    """
    n_samples, n_features_orig = X.shape
    pipeline_warnings: list[str] = []
    rng = np.random.RandomState(random_state)

    # --- Edge: no features ---------------------------------------------------
    if n_features_orig == 0:
        return ImpactResult(
            summary=pd.DataFrame(),
            interactions=None,
            metadata={
                "n_samples": n_samples,
                "n_features_original": 0,
                "n_features_total": 0,
                "warnings": ["No features provided."],
            },
        )

    # --- Edge: very few samples ----------------------------------------------
    if n_samples < 5:
        pipeline_warnings.append(
            f"Very few samples ({n_samples}). Results will be unreliable."
        )

    # --- Standardise target ---------------------------------------------------
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std < 1e-12:
        pipeline_warnings.append("Target has zero variance — cannot estimate impacts.")
        empty_data: dict[str, Any] = {
            "Mean_Weight": 0.0,
            "Std_Dev": 0.0,
            "CI_Low": 0.0,
            "CI_High": 0.0,
            "Is_Reliable": False,
            "Frequency": 0.0,
        }
        if idea_ids is not None:
            empty_data["Description"] = feature_names
            empty = pd.DataFrame(empty_data, index=pd.Index(idea_ids))
        else:
            empty = pd.DataFrame(empty_data, index=pd.Index(feature_names))
        return ImpactResult(
            summary=empty,
            interactions=None,
            metadata={
                "n_samples": n_samples,
                "n_features_original": n_features_orig,
                "n_features_total": n_features_orig,
                "warnings": pipeline_warnings,
            },
        )

    y_scaled = (y - y_mean) / y_std

    # --- Interaction detection ------------------------------------------------
    n_main = X.shape[1]
    if include_interactions and n_features_orig >= 2:
        X_aug, all_names, all_ids = _select_interactions(
            X,
            feature_names,
            y_scaled,
            max_pairs=max_interaction_pairs,
            min_cooccurrence=3,
            idea_ids=idea_ids,
        )
    else:
        X_aug = X
        all_names = list(feature_names)
        all_ids = list(idea_ids) if idea_ids is not None else None

    interaction_names = all_names[n_main:]
    n_features_total = X_aug.shape[1]

    if n_features_total >= n_samples:
        pipeline_warnings.append(
            f"More features ({n_features_total}) than samples ({n_samples}). "
            "Relying heavily on L1 regularisation."
        )

    # --- Adaptive sizing ------------------------------------------------------
    cv_folds = _adaptive_cv(n_samples)
    # Reduce iterations proportionally for tiny datasets — CIs won't tighten
    # much beyond ~50 iterations when n is very small.
    effective_iterations = min(n_iterations, max(50, n_samples * 10))

    # --- Bootstrap ElasticNet -------------------------------------------------
    coefs = _bootstrap_elasticnet(
        X_aug,
        y_scaled,
        effective_iterations,
        cv_folds,
        rng,
    )

    # Rescale coefficients back to the original y scale
    coefs *= y_std

    # --- Summary statistics ---------------------------------------------------
    z = _Z_SCORES.get(confidence_level, 1.960)
    means = coefs.mean(axis=0)
    stds = coefs.std(axis=0)
    ci_low = means - z * stds
    ci_high = means + z * stds
    # Selection frequency: fraction of bootstrap iterations with non-zero coeff
    frequency = (np.abs(coefs) > 1e-10).mean(axis=0)

    # Build index: use idea IDs when available, otherwise descriptions
    if all_ids is not None:
        df_index = pd.Index(all_ids)
    else:
        df_index = pd.Index(all_names)

    summary_data: dict[str, Any] = {
        "Mean_Weight": means,
        "Std_Dev": stds,
        "CI_Low": ci_low,
        "CI_High": ci_high,
        "Is_Reliable": (
            (np.sign(ci_low) == np.sign(ci_high)) & (np.abs(means) > 1e-10)
        ),
        "Frequency": frequency,
    }
    if all_ids is not None:
        summary_data["Description"] = all_names

    summary_full = pd.DataFrame(summary_data, index=df_index)

    # Split main effects vs interactions
    main_summary = summary_full.iloc[:n_main].sort_values(
        by=["Is_Reliable", "Mean_Weight"],
        ascending=[False, False],
    )
    if interaction_names:
        interaction_summary: pd.DataFrame | None = summary_full.iloc[
            n_main:
        ].sort_values(
            by=["Is_Reliable", "Mean_Weight"],
            ascending=[False, False],
        )
    else:
        interaction_summary = None

    metadata: dict[str, Any] = {
        "n_samples": n_samples,
        "n_features_original": n_features_orig,
        "n_features_total": n_features_total,
        "n_interactions_added": len(interaction_names),
        "n_iterations": effective_iterations,
        "cv_folds": cv_folds,
        "confidence_level": confidence_level,
        "y_mean": y_mean,
        "y_std": y_std,
        "warnings": pipeline_warnings,
    }

    return ImpactResult(
        summary=main_summary,
        interactions=interaction_summary,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# High-level entry point (accepts IdeaTracker objects directly)
# ---------------------------------------------------------------------------


def run_impact_pipeline(
    programs: list[Any],
    ideas: list[Any],
    *,
    n_iterations: int = 200,
    include_interactions: bool = True,
    max_interaction_pairs: int = 10,
    min_idea_programs: int = 2,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> ImpactResult:
    """End-to-end entry point: build features from IdeaTracker data, filter,
    and run the full impact analysis.

    Parameters
    ----------
    programs : list
        ``ProgramRecord`` instances.
    ideas : list
        ``RecordCard`` instances.
    n_iterations : int
        Bootstrap repetitions.
    include_interactions : bool
        Whether to detect pairwise interaction effects.
    max_interaction_pairs : int
        Cap on interaction terms added.
    min_idea_programs : int
        An idea must be linked to at least this many programs (and be
        *absent* from at least this many) to be included.
    confidence_level : float
        CI width (0.90 / 0.95 / 0.99).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ImpactResult
        ``.summary`` — main-effect impacts per idea.
        ``.interactions`` — interaction-effect impacts (or ``None``).
        ``.metadata`` — diagnostics, warnings, id mappings.
    """
    X, y, feature_names, idea_ids = build_feature_matrix(programs, ideas)

    # Filter ideas with insufficient variation
    X_filt, names_filt, ids_filt, keep_mask = _filter_features(
        X,
        feature_names,
        idea_ids,
        min_pos=min_idea_programs,
        min_neg=min_idea_programs,
    )

    if X_filt.shape[1] == 0:
        return ImpactResult(
            summary=pd.DataFrame(),
            interactions=None,
            metadata={
                "n_samples": len(programs),
                "n_features_original": len(ideas),
                "n_features_total": 0,
                "warnings": [
                    f"No ideas had >= {min_idea_programs} linked programs "
                    f"and >= {min_idea_programs} programs without them. "
                    "Cannot estimate impacts.",
                ],
                "idea_id_map": {},
            },
        )

    result = impact_analysis(
        X_filt,
        y,
        names_filt,
        idea_ids=ids_filt,
        n_iterations=n_iterations,
        include_interactions=include_interactions,
        max_interaction_pairs=max_interaction_pairs,
        confidence_level=confidence_level,
        random_state=random_state,
    )

    # Attach id ↔ description mapping so callers can resolve back to UUIDs
    result.metadata["idea_id_map"] = dict(zip(names_filt, ids_filt))

    return result


if __name__ == "__main__":
    import json
    from pathlib import Path

    from gigaevo.llm.ideas_tracker.components.data_components import (
        ProgramRecord,
        RecordCard,
    )

    p = Path(__file__).resolve().parent / "st_input"
    p_out = Path(__file__).resolve().parent / "st_output"
    programs_path = p / "programs.json"
    ideas_path = p / "banks.json"
    programs = [
        ProgramRecord(**p) for p in json.load(open(programs_path))[0]["programs"]
    ]
    inactive_ideas = [
        RecordCard(**i) for i in json.load(open(ideas_path))[0]["inactive_bank"]
    ]
    activeideas = [
        RecordCard(**i) for i in json.load(open(ideas_path))[0]["active_bank"]
    ]
    ideas = inactive_ideas + activeideas
    result = run_impact_pipeline(
        programs,
        ideas,
        n_iterations=50,
    )
    result.summary.to_csv(p_out / "impact_summary.csv")
    if result.interactions is not None:
        result.interactions.to_csv(p_out / "impact_interactions.csv")
    with open(p_out / "metadata.json", "w") as f:
        json.dump(result.metadata, f, indent=4)
