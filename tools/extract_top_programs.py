#!/usr/bin/env python3
"""
Extract Top Programs Script

This script extracts the top-N programs by fitness from a Redis database
and saves their code to individual files in a specified folder.

Usage:
    python extract_top_programs.py --output-folder ./top_programs --top-n 10 [--redis-host localhost] [--redis-port 6379] [--redis-db 0]

Arguments:
    --output-folder: Directory to save the program files
    --top-n: Number of top programs to extract (default: 10)
    --redis-host: Redis host (default: localhost)
    --redis-port: Redis port (default: 6379)
    --redis-db: Redis database number (default: 0)
    --fitness-metric: Name of the fitness metric to use (default: fitness)
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys
from typing import List

from loguru import logger

from src.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from src.programs.program import Program


class TopProgramExtractor:
    """Extract top programs by fitness from Redis and save to files."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db

        # Create Redis storage connection
        self.redis_storage = RedisProgramStorage(
            RedisProgramStorageConfig(
                redis_url=f"redis://{redis_host}:{redis_port}/{redis_db}",
                key_prefix=args.key_prefix,
                max_connections=50,
                connection_pool_timeout=30.0,
                health_check_interval=60,
            )
        )

        logger.info(
            f"Initialized extractor for Redis at {redis_host}:{redis_port}/{redis_db}"
        )

    async def extract_all_programs(self) -> List[Program]:
        """Extract all programs from Redis."""
        logger.info("🔍 Extracting programs from Redis...")

        try:
            all_programs = await self.redis_storage.get_all()
            logger.info(f"📊 Found {len(all_programs)} total programs")
            return all_programs
        except Exception as e:
            logger.error(f"❌ Error extracting programs: {e}")
            return []

    def filter_programs_by_fitness(
        self, programs: List[Program], fitness_metric: str = "fitness"
    ) -> List[Program]:
        """Filter programs that have valid fitness values."""
        (
            f"metric_{fitness_metric}"
            if not fitness_metric.startswith("metric_")
            else fitness_metric
        )

        valid_programs = []
        for program in programs:
            # Check if program has the fitness metric
            if not hasattr(program, "metrics") or not program.metrics:
                continue

            fitness_value = program.metrics.get(fitness_metric)
            if fitness_value is None:
                continue

            # Filter out invalid fitness values (like -1000.0 which indicates failure)
            if fitness_value == -1000.0:
                continue

            valid_programs.append(program)

        logger.info(
            f"📈 Found {len(valid_programs)} programs with valid '{fitness_metric}' metric out of {len(programs)} total"
        )
        return valid_programs

    def get_top_programs(
        self,
        programs: List[Program],
        top_n: int,
        fitness_metric: str = "fitness",
    ) -> List[Program]:
        """Get top N programs sorted by fitness."""
        if not programs:
            return []

        # Sort programs by fitness (descending - higher fitness is better)
        sorted_programs = sorted(
            programs,
            key=lambda p: p.metrics.get(fitness_metric, -float("inf")),
            reverse=True,
        )

        # Get top N
        top_programs = sorted_programs[:top_n]

        logger.info(
            f"🏆 Selected top {len(top_programs)} programs by '{fitness_metric}' metric"
        )

        # Log the fitness range
        if top_programs:
            best_fitness = top_programs[0].metrics.get(fitness_metric)
            worst_fitness = top_programs[-1].metrics.get(fitness_metric)
            logger.info(
                f"Fitness range: {worst_fitness:.4f} to {best_fitness:.4f}"
            )

        return top_programs

    def save_programs_to_folder(
        self,
        programs: List[Program],
        output_folder: Path,
        fitness_metric: str = "fitness",
    ) -> None:
        """Save program codes to individual files in the output folder."""
        if not programs:
            logger.warning("⚠️ No programs to save")
            return

        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save each program to a separate file
        for i, program in enumerate(programs, 1):
            fitness_value = program.metrics.get(fitness_metric, 0.0)

            # Create filename with ranking, fitness, and program ID
            filename = f"rank_{i:02d}_fitness_{fitness_value:.4f}_id_{program.id[:8]}.py"
            filepath = output_folder / filename

            # Prepare file content with metadata header
            content = f'''"""
Top Program #{i}
Program ID: {program.id}
Fitness: {fitness_value:.4f}
Created: {program.created_at}
Updated: {program.updated_at}
Generation: {program.generation or 'N/A'}
State: {program.state}
"""

{program.code}
'''

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"💾 Saved program {i}/{len(programs)}: {filename}")

        # Create summary file
        summary_file = output_folder / "summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Top {len(programs)} Programs Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Extracted at: {datetime.now()}\n")
            f.write(f"Fitness metric: {fitness_metric}\n")
            f.write(
                f"Redis: {self.redis_host}:{self.redis_port}/{self.redis_db}\n\n"
            )

            for i, program in enumerate(programs, 1):
                fitness_value = program.metrics.get(fitness_metric, 0.0)
                f.write(
                    f"{i:2d}. Program {program.id[:12]}... | Fitness: {fitness_value:.4f} | Gen: {program.generation or 'N/A'} | State: {program.state}\n"
                )

                # Add lineage info if available
                if program.lineage and program.lineage.parents:
                    f.write(
                        f"    Parents: {len(program.lineage.parents)} | Mutation: {program.lineage.mutation or 'N/A'}\n"
                    )

                # Add other metrics if available
                other_metrics = {
                    k: v
                    for k, v in program.metrics.items()
                    if k != fitness_metric
                }
                if other_metrics:
                    metrics_str = ", ".join(
                        [
                            f"{k}: {v:.3f}"
                            for k, v in list(other_metrics.items())[:3]
                        ]
                    )
                    f.write(f"    Other metrics: {metrics_str}\n")

                f.write("\n")

        logger.info(f"📄 Saved summary to {summary_file}")
        logger.info(f"✅ All programs saved to {output_folder}")

    async def extract_and_save_top_programs(
        self,
        output_folder: Path,
        top_n: int = 10,
        fitness_metric: str = "fitness",
    ) -> None:
        """Main method to extract and save top programs."""
        logger.info(f"🚀 Starting extraction of top {top_n} programs...")

        # Extract all programs
        all_programs = await self.extract_all_programs()
        if not all_programs:
            logger.error("❌ No programs found in Redis")
            return

        # Filter programs with valid fitness
        valid_programs = self.filter_programs_by_fitness(
            all_programs, fitness_metric
        )
        if not valid_programs:
            logger.error(
                f"❌ No programs found with valid '{fitness_metric}' metric"
            )
            return

        # Get top N programs
        top_programs = self.get_top_programs(
            valid_programs, top_n, fitness_metric
        )
        if not top_programs:
            logger.error("❌ No top programs selected")
            return

        # Save to folder
        self.save_programs_to_folder(
            top_programs, output_folder, fitness_metric
        )

        logger.info(
            f"🎉 Successfully extracted and saved {len(top_programs)} top programs!"
        )

    async def cleanup(self):
        """Clean up Redis connections."""
        try:
            redis_conn = await self.redis_storage._conn()
            if redis_conn:
                if hasattr(redis_conn, "connection_pool"):
                    await redis_conn.connection_pool.disconnect()
                if hasattr(redis_conn, "close"):
                    await redis_conn.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing Redis connection: {e}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract top programs by fitness from Redis"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Directory to save the program files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top programs to extract (default: 10)",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host (default: localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis database number (default: 0)",
    )
    parser.add_argument(
        "--key-prefix", type=str, required=True, help="Redis key prefix"
    )
    parser.add_argument(
        "--fitness-metric",
        type=str,
        default="fitness",
        help="Name of the fitness metric to use (default: fitness)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.top_n <= 0:
        logger.error("❌ --top-n must be positive")
        sys.exit(1)

    output_folder = Path(args.output_folder)

    # Create extractor
    extractor = TopProgramExtractor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
    )

    try:
        # Extract and save programs
        await extractor.extract_and_save_top_programs(
            output_folder=output_folder,
            top_n=args.top_n,
            fitness_metric=args.fitness_metric,
        )
    finally:
        # Clean up
        await extractor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
