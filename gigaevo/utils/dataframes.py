"""DataFrame preparation and outlier detection.

Ported from tools/utils.py — provides iteration-level DataFrame processing,
outlier detection with multiple methods, and Redis frontier integration.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from loguru import logger
import numpy as np
import pandas as pd
import redis

from gigaevo.utils.redis import RedisRunConfig  # noqa: F401


class OutlierMethod(StrEnum):
    IQR = "iqr"
    MAD = "mad"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"


def _outlier_mask_iqr(
    values: pd.Series,
    multiplier: float = 1.5,
) -> tuple[pd.Series, float, float]:
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    if IQR <= 0:
        return pd.Series(False, index=values.index), Q1, Q3

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    mask = (values < lower) | (values > upper)
    return mask, lower, upper


def _outlier_mask_mad(
    values: pd.Series,
    multiplier: float = 3.5,
) -> tuple[pd.Series, float, float]:
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad <= 0:
        return pd.Series(False, index=values.index), median, median

    scaled_mad = 1.4826 * mad
    lower = median - multiplier * scaled_mad
    upper = median + multiplier * scaled_mad
    mask = (values < lower) | (values > upper)
    return mask, lower, upper


def _outlier_mask_zscore(
    values: pd.Series,
    threshold: float = 3.0,
) -> tuple[pd.Series, float, float]:
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad <= 0:
        return pd.Series(False, index=values.index), median, median

    modified_z = 0.6745 * (values - median) / mad
    mask = np.abs(modified_z) > threshold

    scaled_mad = 1.4826 * mad
    lower = median - threshold * scaled_mad
    upper = median + threshold * scaled_mad
    return mask, lower, upper


def _outlier_mask_percentile(
    values: pd.Series,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[pd.Series, float, float]:
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
    if isinstance(method, str):
        method = OutlierMethod(method.lower())

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

    if side == "low":
        mask = valid_values < lower
    elif side == "high":
        mask = valid_values > upper
    else:
        mask = (valid_values < lower) | (valid_values > upper)

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
    iteration_col: str = "iteration",
    minimize: bool = False,
    compute_frontier: bool = True,
    sentinel_value: float | None = None,
) -> pd.DataFrame:
    if fitness_col not in df.columns:
        logger.warning("No fitness metric found in dataframe")
        return pd.DataFrame()

    if iteration_col not in df.columns:
        logger.warning("No iteration metadata found in dataframe")
        return pd.DataFrame()

    df = df.copy()
    df[iteration_col] = pd.to_numeric(df[iteration_col], errors="coerce")

    valid = df[fitness_col].notna()
    df = df[valid]
    if df.empty:
        return pd.DataFrame()

    n_before = len(df)

    if sentinel_value is not None:
        sentinel_mask = df[fitness_col] == sentinel_value
        n_sentinel = int(sentinel_mask.sum())
        if n_sentinel > 0:
            df = df[~sentinel_mask]
            pct_sentinel = 100.0 * n_sentinel / n_before
            logger.info(
                f"Sentinel removal: removed {n_sentinel}/{n_before} "
                f"({pct_sentinel:.1f}%) points with fitness == {sentinel_value}"
            )
            n_before = len(df)
            if df.empty:
                logger.warning("All data points were sentinel values")
                return pd.DataFrame()

    if extreme_value_cutoff is None:
        extreme_value_cutoff = 1000.0 if minimize else -1000.0

    if extreme_value_cutoff is not None:
        if minimize:
            extreme_mask = df[fitness_col] >= extreme_value_cutoff
        else:
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
            n_before = len(df)
            if df.empty:
                logger.warning("All data points were extreme values")
                return pd.DataFrame()

    if remove_outliers:
        outlier_side: Literal["both", "low", "high"] = "high" if minimize else "low"
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

    df = df[df[iteration_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(iteration_col).reset_index(drop=True)

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

    if compute_frontier:
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

    output_cols = [
        iteration_col,
        fitness_col,
        "running_mean_fitness",
        "running_std_fitness",
        "running_mean_plus_std",
        "running_mean_minus_std",
    ]
    if compute_frontier:
        output_cols.append("frontier_fitness")

    return df[output_cols]


def fetch_frontier_from_redis(
    redis_host: str, redis_port: int, redis_db: int, redis_prefix: str, metric_key: str
) -> dict[int, float] | None:
    import json

    try:
        r = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        history_key = f"{redis_prefix}:metrics:history:program_metrics:valid/frontier/{metric_key}"
        entries = r.lrange(history_key, 0, -1)
        if not entries:
            return None

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
    iteration_col: str = "iteration",
) -> pd.DataFrame:
    frontier_series = fetch_frontier_from_redis(
        redis_host, redis_port, redis_db, redis_prefix, metric_key
    )
    if frontier_series is None:
        logger.info(
            f"Frontier not available in Redis for {redis_prefix}; using computed frontier"
        )
        return df

    df_copy = df.copy()
    df_copy["frontier_fitness"] = df_copy[iteration_col].map(frontier_series)
    df_copy["frontier_fitness"] = df_copy["frontier_fitness"].fillna(method="ffill")

    logger.info(
        f"Loaded frontier from Redis for {redis_prefix}: {len(frontier_series)} iterations"
    )
    return df_copy
