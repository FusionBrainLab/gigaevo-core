import asyncio
from pathlib import Path
from typing import Optional

import pandas as pd

from gigaevo.memory.ideas_tracker.utils.helpers import normalize_dataframe
from tools.utils import RedisRunConfig, fetch_evolution_dataframe


def load_dataframe(
    redis_config: RedisRunConfig, path_to_database: Optional[str | Path]
) -> pd.DataFrame:
    if path_to_database is not None:
        if isinstance(path_to_database, str):
            path_to_database = Path(path_to_database)
        if path_to_database.is_file() and path_to_database.suffix == ".csv":
            df = pd.read_csv(path_to_database)
        else:
            raise ValueError(f"Invalid database file: {path_to_database}")
    else:
        df = asyncio.run(load_database(redis_config))

    df = normalize_dataframe(df)
    if df.empty:
        return pd.DataFrame()

    required_columns = {
        "program_id",
        "metric_fitness",
        "generation",
        "is_root",
        "parent_ids",
        "metadata_mutation_output",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Ideas tracker input is missing required columns: "
            + ", ".join(missing_columns)
        )
    return df


async def load_database(redis_config: RedisRunConfig) -> pd.DataFrame:
    """
    Load fresh copy of Redis database as Pandas DataFrame.

    Returns:
        DataFrame containing evolution data from Redis.
    """
    dataset = await fetch_evolution_dataframe(redis_config)
    return dataset
