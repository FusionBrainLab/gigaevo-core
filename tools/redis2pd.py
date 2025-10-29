import argparse
import asyncio

import pandas as pd

from tools.utils import RedisRunConfig, fetch_evolution_dataframe


async def main():
    parser = argparse.ArgumentParser(
        description="Convert Redis data to pandas DataFrame"
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, required=True, help="Redis database")
    parser.add_argument("--redis-prefix", type=str, required=True, help="Redis prefix")
    parser.add_argument("--output-file", type=str, required=True, help="Output file")
    args = parser.parse_args()
    config = RedisRunConfig(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_prefix=args.redis_prefix,
        label=args.output_file,
    )
    df: pd.DataFrame = await fetch_evolution_dataframe(config, add_stage_results=False)
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    asyncio.run(main())
