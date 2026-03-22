from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from loguru import logger
import numpy as np
import pandas as pd
import redis

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


class OutlierMethod(StrEnum):
    """Outlier detection methods."""

    IQR = "iqr"  # Interquartile Range (Tukey's method)
    MAD = "mad"  # Median Absolute Deviation (more robust)
    ZSCORE = "zscore"  # Z-score with robust estimators
    PERCENTILE = "percentile"  # Simple percentile-based clipping


@dataclass
class RedisRunConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_prefix: str = ""
    label: str = ""

    def url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def display_label(self) -> str:
        return self.label or f"{self.redis_prefix}@{self.redis_db}"


async def fetch_evolution_dataframe(
    config: RedisRunConfig, add_stage_results: bool = False
) -> pd.DataFrame:
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=config.url(),
            key_prefix=config.redis_prefix,
            max_connections=50,
            connection_pool_timeout=30.0,
            health_check_interval=60,
            read_only=True,
        )
    )

    try:
        programs = await storage.get_all()
    finally:
        await storage.close()

    if not programs:
        logger.warning(
            f"No programs found for prefix='{config.redis_prefix}' at {config.url()}"
        )
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for program in programs:
        row: dict[str, Any] = {
            "program_id": program.id,
            "name": program.name or "unnamed",
            "code": program.code,
            "created_at": program.created_at,
            "atomic_counter": program.atomic_counter,
            "state": program.state.value,
            "is_complete": program.is_complete,
            "generation": program.generation,
            "is_root": program.is_root,
            "parent_ids": (program.lineage.parents),
            "children_ids": (program.lineage.children),
        }
        if add_stage_results:
            row["stage_results"] = program.stage_results
        # metrics
        metrics = program.metrics
        for mname, mval in metrics.items():
            row[f"metric_{mname}"] = mval

        # lineage
        lineage = program.lineage
        row["lineage_num_parents"] = len(lineage.parents)
        row["lineage_num_children"] = len(lineage.children)
        row["lineage_mutation"] = lineage.mutation
        row["lineage_generation"] = lineage.generation

        # metadata
        metadata = program.metadata
        for k, v in metadata.items():
            row[f"metadata_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def _outlier_mask_iqr(
    values: pd.Series,
    multiplier: float = 1.5,
) -> tuple[pd.Series, float, float]:
    """IQR-based outlier detection (Tukey's method).

    Args:
        values: Series of values to check
        multiplier: IQR multiplier (1.5 = standard, 3.0 = conservative)

    Returns:
        (mask, lower_bound, upper_bound) where mask is True for outliers
    """
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    if IQR <= 0:
        # All values are essentially the same; no outliers
        return pd.Series(False, index=values.index), Q1, Q3

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    mask = (values < lower) | (values > upper)
    return mask, lower, upper


def _outlier_mask_mad(
    values: pd.Series,
    multiplier: float = 3.5,
) -> tuple[pd.Series, float, float]:
    """MAD-based outlier detection (Median Absolute Deviation).

    More robust than IQR for highly skewed or heavy-tailed distributions.
    Uses the consistency constant 1.4826 to make MAD comparable to std dev.

    Args:
        values: Series of values to check
        multiplier: Number of scaled MADs from median (3.5 is common)

    Returns:
        (mask, lower_bound, upper_bound) where mask is True for outliers
    """
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad <= 0:
        # All values are essentially the same; no outliers
        return pd.Series(False, index=values.index), median, median

    # Scale MAD to be comparable to standard deviation
    scaled_mad = 1.4826 * mad
    lower = median - multiplier * scaled_mad
    upper = median + multiplier * scaled_mad
    mask = (values < lower) | (values > upper)
    return mask, lower, upper


def _outlier_mask_zscore(
    values: pd.Series,
    threshold: float = 3.0,
) -> tuple[pd.Series, float, float]:
    """Modified Z-score outlier detection using robust estimators.

    Uses median and MAD instead of mean and std for robustness.

    Args:
        values: Series of values to check
        threshold: Z-score threshold (3.0 is common)

    Returns:
        (mask, lower_bound, upper_bound) where mask is True for outliers
    """
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad <= 0:
        return pd.Series(False, index=values.index), median, median

    # Modified z-score
    modified_z = 0.6745 * (values - median) / mad
    mask = np.abs(modified_z) > threshold

    # Approximate bounds for reporting
    scaled_mad = 1.4826 * mad
    lower = median - threshold * scaled_mad
    upper = median + threshold * scaled_mad
    return mask, lower, upper


def _outlier_mask_percentile(
    values: pd.Series,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[pd.Series, float, float]:
    """Simple percentile-based outlier detection.

    Args:
        values: Series of values to check
        lower_percentile: Lower percentile cutoff (default 1%)
        upper_percentile: Upper percentile cutoff (default 99%)

    Returns:
        (mask, lower_bound, upper_bound) where mask is True for outliers
    """
    lower = values.quantile(lower_percentile / 100.0)
    upper = values.quantile(upper_percentile / 100.0)
    mask = (values < lower) | (values > upper)
    return mask, lower, upper


def detect_outliers(
    values: pd.Series,
    method: OutlierMethod | str = OutlierMethod.MAD,
    multiplier: float | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    side: Literal["both", "low", "high"] = "both",
) -> tuple[pd.Series, float, float, int]:
    """Detect outliers using the specified method.

    Args:
        values: Series of values to check for outliers
        method: Detection method (iqr, mad, zscore, percentile)
        multiplier: Method-specific multiplier/threshold:
            - IQR: multiplier for IQR (default 1.5, use 3.0 for conservative)
            - MAD: multiplier for scaled MAD (default 3.5)
            - ZSCORE: z-score threshold (default 3.0)
            - PERCENTILE: ignored, uses lower/upper_percentile instead
        lower_percentile: For percentile method, lower bound (default 1%)
        upper_percentile: For percentile method, upper bound (default 99%)
        side: Which side to detect outliers on:
            - "both": detect outliers on both sides (default)
            - "low": only detect low outliers (values below lower bound)
            - "high": only detect high outliers (values above upper bound)

    Returns:
        (mask, lower_bound, upper_bound, n_outliers) where mask is True for outliers
    """
    if isinstance(method, str):
        method = OutlierMethod(method.lower())

    # Handle NaN values - they're not outliers, just missing
    valid_mask = values.notna()
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        return pd.Series(False, index=values.index), float("nan"), float("nan"), 0

    if method == OutlierMethod.IQR:
        mult = multiplier if multiplier is not None else 1.5
        _, lower, upper = _outlier_mask_iqr(valid_values, mult)
    elif method == OutlierMethod.MAD:
        mult = multiplier if multiplier is not None else 3.5
        _, lower, upper = _outlier_mask_mad(valid_values, mult)
    elif method == OutlierMethod.ZSCORE:
        thresh = multiplier if multiplier is not None else 3.0
        _, lower, upper = _outlier_mask_zscore(valid_values, thresh)
    elif method == OutlierMethod.PERCENTILE:
        _, lower, upper = _outlier_mask_percentile(
            valid_values, lower_percentile, upper_percentile
        )
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    # Apply one-sided or two-sided masking based on 'side' parameter
    if side == "low":
        # Only flag values below lower bound (bad performers in maximization)
        mask = valid_values < lower
    elif side == "high":
        # Only flag values above upper bound (bad performers in minimization)
        mask = valid_values > upper
    else:  # "both"
        mask = (valid_values < lower) | (valid_values > upper)

    # Expand mask back to original index
    full_mask = pd.Series(False, index=values.index)
    full_mask.loc[mask.index] = mask

    n_outliers = int(full_mask.sum())
    return full_mask, lower, upper, n_outliers


def prepare_iteration_dataframe(
    df: pd.DataFrame,
    *,
    iteration_rolling_window: int = 5,
    remove_outliers: bool = True,
    outlier_method: OutlierMethod | str = OutlierMethod.PERCENTILE,
    outlier_multiplier: float | None = None,
    outlier_lower_percentile: float = 5.0,
    outlier_upper_percentile: float = 95.0,
    extreme_value_cutoff: float | None = None,
    fitness_col: str = "metric_fitness",
    iteration_col: str = "metadata_iteration",
    minimize: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame sorted by iteration with rolling mean/std columns.

    Args:
        df: Input DataFrame with fitness and iteration columns
        iteration_rolling_window: Window size for rolling statistics
        remove_outliers: Whether to remove outliers before computing stats
        outlier_method: Method for outlier detection (iqr, mad, zscore, percentile)
        outlier_multiplier: Method-specific multiplier (None = use method default)
        outlier_lower_percentile: For percentile method, lower bound
        outlier_upper_percentile: For percentile method, upper bound
        extreme_value_cutoff: Absolute value cutoff for extreme failures.
            For minimize: removes values >= cutoff (default: 1000)
            For maximize: removes values <= cutoff (default: -1000)
            Set to None to disable.
        fitness_col: Name of the fitness column
        iteration_col: Name of the iteration column
        minimize: If True, lower fitness is better (for frontier calculation)

    Returns:
        DataFrame with iteration, fitness, and computed statistics columns
    """
    if fitness_col not in df.columns:
        logger.warning("No fitness metric found in dataframe")
        return pd.DataFrame()

    if iteration_col not in df.columns:
        logger.warning("No iteration metadata found in dataframe")
        return pd.DataFrame()

    # Coerce iteration to numeric
    df = df.copy()
    df[iteration_col] = pd.to_numeric(df[iteration_col], errors="coerce")

    valid = df[fitness_col].notna()
    df = df[valid]
    if df.empty:
        return pd.DataFrame()

    n_before = len(df)

    # Step 1: Remove extreme values by absolute cutoff (e.g., hard-coded failure values)
    if extreme_value_cutoff is None:
        # Set sensible defaults based on optimization direction
        extreme_value_cutoff = 1000.0 if minimize else -1000.0

    if extreme_value_cutoff is not None:
        if minimize:
            # Remove values >= cutoff (extremely bad in minimization)
            extreme_mask = df[fitness_col] >= extreme_value_cutoff
        else:
            # Remove values <= cutoff (extremely bad in maximization)
            extreme_mask = df[fitness_col] <= extreme_value_cutoff

        n_extreme = int(extreme_mask.sum())
        if n_extreme > 0:
            df = df[~extreme_mask]
            pct_extreme = 100.0 * n_extreme / n_before
            direction = ">=" if minimize else "<="
            logger.info(
                f"Extreme value removal: removed {n_extreme}/{n_before} "
                f"({pct_extreme:.1f}%) points with fitness {direction} {extreme_value_cutoff}"
            )
            n_before = len(df)  # Update count for next stage
            if df.empty:
                logger.warning("All data points were extreme values")
                return pd.DataFrame()

    # Step 2: Outlier removal using robust methods (one-sided based on optimization direction)
    if remove_outliers:
        outlier_side = "high" if minimize else "low"
        mask, lower, upper, n_outliers = detect_outliers(
            df[fitness_col],
            method=outlier_method,
            multiplier=outlier_multiplier,
            lower_percentile=outlier_lower_percentile,
            upper_percentile=outlier_upper_percentile,
            side=outlier_side,
        )
        df = df[~mask]
        if n_outliers > 0:
            pct = 100.0 * n_outliers / n_before
            bound_desc = f"above {upper:.4g}" if minimize else f"below {lower:.4g}"
            logger.info(
                f"Outlier removal ({outlier_method}, side={outlier_side}): "
                f"removed {n_outliers}/{n_before} ({pct:.1f}%) points {bound_desc}"
            )
        if df.empty:
            logger.warning("All data points were classified as outliers")
            return pd.DataFrame()

    # Keep only rows with an iteration value
    df = df[df[iteration_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Sort by iteration to compute running stats in iteration order
    df = df.sort_values(iteration_col).reset_index(drop=True)

    # Rolling statistics over the sequence (not grouped by unique iteration)
    df["running_mean_fitness"] = (
        df[fitness_col]
        .rolling(window=iteration_rolling_window, min_periods=1, center=False)
        .mean()
    )
    df["running_std_fitness"] = (
        df[fitness_col]
        .rolling(window=iteration_rolling_window, min_periods=1, center=False)
        .std()
    )
    df["running_mean_plus_std"] = df["running_mean_fitness"] + df["running_std_fitness"]
    df["running_mean_minus_std"] = (
        df["running_mean_fitness"] - df["running_std_fitness"]
    )

    # Frontier: per-iteration best and its cumulative best across iterations
    # For minimization problems, we want the minimum; for maximization, the maximum
    if minimize:
        per_iter_best = (
            df.groupby(iteration_col, as_index=False)[fitness_col]
            .min()
            .sort_values(iteration_col)
            .reset_index(drop=True)
        )
        per_iter_best["frontier_fitness"] = per_iter_best[fitness_col].cummin()
    else:
        per_iter_best = (
            df.groupby(iteration_col, as_index=False)[fitness_col]
            .max()
            .sort_values(iteration_col)
            .reset_index(drop=True)
        )
        per_iter_best["frontier_fitness"] = per_iter_best[fitness_col].cummax()
    df = df.merge(
        per_iter_best[[iteration_col, "frontier_fitness"]],
        on=iteration_col,
        how="left",
    )

    return df[
        [
            iteration_col,
            fitness_col,
            "running_mean_fitness",
            "running_std_fitness",
            "running_mean_plus_std",
            "running_mean_minus_std",
            "frontier_fitness",
        ]
    ]


def fetch_frontier_from_redis(
    redis_host: str, redis_port: int, redis_db: int, redis_prefix: str, metric_key: str
) -> dict[int, float] | None:
    """Fetch the authoritative frontier series from Redis metrics history.

    When NO_CACHE stages re-evaluate programs, the frontier written to Redis is
    the single source of truth. This function retrieves it directly.

    Args:
        redis_host: Redis server hostname
        redis_port: Redis server port
        redis_db: Redis database number
        redis_prefix: Redis key prefix for the run
        metric_key: Metric name (e.g., "fitness")

    Returns:
        dict mapping iteration -> frontier_value, or None if not available
    """
    import json

    try:
        r = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        # Format: program_metrics:valid:frontier:{metric_key}
        history_key = f"{redis_prefix}:metrics:history:program_metrics:valid/frontier/{metric_key}"
        entries = r.lrange(history_key, 0, -1)
        if not entries:
            return None

        # Each entry is JSON: {"s": step, "v": value, "t": wall_time, "k": "scalar"}
        frontier: dict[int, float] = {}
        for entry_json in entries:
            try:
                entry = json.loads(entry_json)
                iteration = int(entry.get("s", 0))
                value = float(entry.get("v", 0))
                frontier[iteration] = value
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning(
                    f"Failed to parse frontier entry for {redis_prefix}/{metric_key}: {entry_json}"
                )
                continue

        return frontier if frontier else None
    except Exception as e:
        logger.warning(
            f"Failed to fetch frontier from Redis {redis_prefix}@{redis_db}:{metric_key}: {e}"
        )
        return None


def add_frontier_from_redis_to_dataframe(
    df: pd.DataFrame,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    redis_prefix: str,
    metric_key: str,
    iteration_col: str = "metadata_iteration",
) -> pd.DataFrame:
    """Add the authoritative frontier column from Redis to a prepared dataframe.

    Replaces the frontier_fitness column computed from program data with the
    actual frontier series from MetricsTracker (which is correct even after
    NO_CACHE metric re-evaluation).

    Args:
        df: Prepared dataframe from prepare_iteration_dataframe()
        redis_host, redis_port, redis_db, redis_prefix: Redis connection details
        metric_key: Metric name (e.g., "fitness")
        iteration_col: Column name for iterations

    Returns:
        DataFrame with updated frontier_fitness column (or unchanged if fetch fails)
    """
    frontier_series = fetch_frontier_from_redis(
        redis_host, redis_port, redis_db, redis_prefix, metric_key
    )
    if frontier_series is None:
        logger.info(
            f"Frontier not available in Redis for {redis_prefix}; using computed frontier"
        )
        return df

    # Map iteration -> frontier value
    df_copy = df.copy()
    df_copy["frontier_fitness"] = df_copy[iteration_col].map(frontier_series)

    # Forward-fill any missing iterations (should not happen if frontier is complete)
    df_copy["frontier_fitness"] = df_copy["frontier_fitness"].fillna(method="ffill")

    logger.info(
        f"Loaded frontier from Redis for {redis_prefix}: {len(frontier_series)} iterations"
    )
    return df_copy
