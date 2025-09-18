#!/usr/bin/env python3
"""
Shared utilities for extracting and preparing evolution data from Redis.

This module provides:
- fetch_evolution_dataframe: asynchronously load all programs for a given Redis run
- prepare_iteration_dataframe: filter, coerce, and compute iteration-ordered rolling stats
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


ITERATION_COL = "meta_iteration"
FITNESS_COL = "metric_fitness"


@dataclass
class RedisRunConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_prefix: str = ""
    label: Optional[str] = None

    def url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def display_label(self) -> str:
        return self.label or f"{self.redis_prefix}@{self.redis_db}"


async def fetch_evolution_dataframe(config: RedisRunConfig) -> pd.DataFrame:
    """Extract all programs and flattened metrics/metadata from Redis for one run.

    Returns a DataFrame with at least: program_id, created_at, updated_at, state,
    metric_* columns, meta_* columns (including meta_iteration, when present).
    """

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
        try:
            conn = await storage._conn()
            if conn and hasattr(conn, "connection_pool"):
                await conn.connection_pool.disconnect()
            if conn and hasattr(conn, "close"):
                await conn.close()
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")

    if not programs:
        logger.warning(
            f"No programs found for prefix='{config.redis_prefix}' at {config.url()}"
        )
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for program in programs:
        row: Dict[str, Any] = {
            "program_id": getattr(program, "id", None),
            "name": getattr(program, "name", None) or "unnamed",
            "created_at": getattr(program, "created_at", None),
            "updated_at": getattr(program, "updated_at", None),
            "state": getattr(program, "state", None).value if getattr(program, "state", None) else None,
            "is_complete": getattr(program, "is_complete", None),
            "generation": getattr(program, "generation", 0) or 0,
            "parent_count": getattr(program, "parent_count", 0),
            "is_root": getattr(program, "is_root", False),
        }

        # metrics
        metrics = getattr(program, "metrics", None) or {}
        for mname, mval in metrics.items():
            row[f"metric_{mname}"] = mval

        # lineage
        lineage = getattr(program, "lineage", None)
        if lineage:
            row["lineage_parents"] = len(getattr(lineage, "parents", []) or [])
            row["lineage_mutation"] = getattr(lineage, "mutation", None)
            row["lineage_generation"] = getattr(lineage, "generation", 0) or 0
        else:
            row["lineage_parents"] = 0
            row["lineage_mutation"] = None
            row["lineage_generation"] = 0

        # metadata
        metadata = getattr(program, "metadata", None) or {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                row[f"meta_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def _outlier_mask(
    values: pd.Series,
    extreme_threshold: float,
    outlier_multiplier: float,
) -> Tuple[pd.Series, float, float]:
    """Return boolean mask for outliers (True = outlier) and the bounds used.

    Uses extreme_threshold and IQR method, consistent with the analyzer logic.
    """
    values = values.copy()
    extreme_outliers = values < extreme_threshold
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
    extreme_threshold: float = -10000.0,
    outlier_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Return a DataFrame sorted by iteration with rolling mean/std columns.

    Required columns in df: FITNESS_COL and ITERATION_COL.
    The output contains: ITERATION_COL, FITNESS_COL, running_mean, running_std.
    """

    if FITNESS_COL not in df.columns:
        logger.warning("No fitness metric found in dataframe")
        return pd.DataFrame()

    if ITERATION_COL not in df.columns:
        logger.warning("No iteration metadata found in dataframe")
        return pd.DataFrame()

    # Coerce iteration to numeric
    df = df.copy()
    df[ITERATION_COL] = pd.to_numeric(df[ITERATION_COL], errors="coerce")

    # Basic validity filter for fitness
    valid = df[FITNESS_COL].notna() & (df[FITNESS_COL] != -1000.0)
    df = df[valid]
    if df.empty:
        return pd.DataFrame()

    # Outlier removal
    if remove_outliers:
        mask, lower, upper = _outlier_mask(
            df[FITNESS_COL], extreme_threshold, outlier_multiplier
        )
        df = df[~mask]
        if df.empty:
            return pd.DataFrame()

    # Keep only rows with an iteration value
    df = df[df[ITERATION_COL].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Sort by iteration to compute running stats in iteration order
    df = df.sort_values(ITERATION_COL).reset_index(drop=True)

    # Rolling statistics over the sequence (not grouped by unique iteration)
    df["running_mean_fitness"] = df[FITNESS_COL].rolling(
        window=iteration_rolling_window, min_periods=1, center=False
    ).mean()
    df["running_std_fitness"] = df[FITNESS_COL].rolling(
        window=iteration_rolling_window, min_periods=1, center=False
    ).std()
    df["running_mean_plus_std"] = df["running_mean_fitness"] + df["running_std_fitness"]
    df["running_mean_minus_std"] = df["running_mean_fitness"] - df["running_std_fitness"]

    # Frontier: per-iteration maximum and its cumulative maximum across iterations
    per_iter_max = (
        df.groupby(ITERATION_COL, as_index=False)[FITNESS_COL]
        .max()
        .sort_values(ITERATION_COL)
        .reset_index(drop=True)
    )
    per_iter_max["frontier_fitness"] = per_iter_max[FITNESS_COL].cummax()
    df = df.merge(
        per_iter_max[[ITERATION_COL, "frontier_fitness"]],
        on=ITERATION_COL,
        how="left",
    )

    return df[[
        ITERATION_COL,
        FITNESS_COL,
        "running_mean_fitness",
        "running_std_fitness",
        "running_mean_plus_std",
        "running_mean_minus_std",
        "frontier_fitness",
    ]]





