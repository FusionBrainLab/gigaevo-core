from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
import pandas as pd

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


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


def _outlier_mask(
    values: pd.Series,
    extreme_threshold: float,
    outlier_multiplier: float,
) -> tuple[pd.Series, float, float]:
    """Return boolean mask for outliers (True = outlier) and the bounds used.

    Uses extreme_threshold and IQR method, consistent with the analyzer logic.
    """
    values = values.copy()
    extreme_outliers = abs(values) < extreme_threshold
    non_extreme = values[~extreme_outliers]
    if len(non_extreme) > 0:
        Q1 = non_extreme.quantile(0.25)
        Q3 = non_extreme.quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - outlier_multiplier * IQR
            upper = Q3 + outlier_multiplier * IQR
            stat_mask = (values < lower) | (values > upper)
        else:
            lower, upper = extreme_threshold, float("inf")
            stat_mask = pd.Series(False, index=values.index)
    else:
        lower, upper = extreme_threshold, float("inf")
        stat_mask = pd.Series(False, index=values.index)

    return (extreme_outliers | stat_mask), lower, upper


def prepare_iteration_dataframe(
    df: pd.DataFrame,
    *,
    iteration_rolling_window: int = 5,
    remove_outliers: bool = True,
    extreme_threshold: float = 100.0,
    outlier_multiplier: float = 3.0,
    fitness_col: str = "metric_fitness",
    iteration_col: str = "metadata_iteration",
) -> pd.DataFrame:
    """Return a DataFrame sorted by iteration with rolling mean/std columns."""

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

    # Outlier removal
    if remove_outliers:
        mask, lower, upper = _outlier_mask(
            df[fitness_col], extreme_threshold, outlier_multiplier
        )
        df = df[~mask]
        if df.empty:
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

    # Frontier: per-iteration maximum and its cumulative maximum across iterations
    per_iter_max = (
        df.groupby(iteration_col, as_index=False)[fitness_col]
        .max()
        .sort_values(iteration_col)
        .reset_index(drop=True)
    )
    per_iter_max["frontier_fitness"] = per_iter_max[fitness_col].cummax()
    df = df.merge(
        per_iter_max[[iteration_col, "frontier_fitness"]],
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
