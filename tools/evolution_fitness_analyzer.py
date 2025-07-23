#!/usr/bin/env python3
"""
Evolution Fitness Analyzer

This script extracts running programs from the MetaEvolve evolution system
and creates comprehensive visualizations including:
- Fitness evolution over time with running mean and standard deviation
- Program state statistics (failed/completed/running/etc.)
- Time since last update analysis (how long programs have been in their current state)
- DAG stage results statistics (individual stage execution analysis)
- Island statistics (distribution of programs across evolution islands)
- Top performing programs analysis
- Comprehensive JSON export for cross-run comparison

It connects to Redis, extracts all programs, analyzes their fitness data,
and creates comprehensive visualizations and reports. Running statistics
(mean, std dev) are calculated using a configurable rolling window and
exported to JSON for easy comparison between different evolution runs.

Usage:
    python evolution_fitness_analyzer.py --redis-prefix PREFIX --output-folder results [options]

Plot Options:
    --no-plots              Skip all plotting (just export data)
    --no-fitness-plots      Skip fitness evolution plots
    --no-stage-plots        Skip program state statistics plots
    --no-persistence-plots  Skip time since last update analysis plots
    --no-dag-stage-plots    Skip DAG stage results statistics plots
    --no-island-plots       Skip island statistics plots
    --no-metric-plots       Skip metric correlation plots
    --no-validity-plots     Skip validity distribution plots
    --no-metric-distribution-plots  Skip metric distribution plots

Outlier Detection Options:
    --extreme-threshold     Threshold for extreme outliers (default: -10000.0)
    --outlier-multiplier    IQR multiplier for outlier detection (default: 3.0)
    --no-outlier-removal    Skip outlier removal (keep all fitness values)

Running Statistics Options:
    --rolling-window        Rolling window size for running mean/std calculations (default: 50)

Export Format:
    - CSV files for raw data
    - JSON files (full and compact) for cross-run comparison
    - PNG and PDF plots for visualization
    - Text summaries and statistics
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # professional styling

from src.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


class EvolutionFitnessAnalyzer:
    """Analyzer for evolution fitness data from MetaEvolve system."""

    def __init__(
        self,
        redis_prefix: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        extreme_threshold: float = -10000.0,
        outlier_multiplier: float = 3.0,
        remove_outliers: bool = True,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_prefix = redis_prefix
        self.extreme_threshold = extreme_threshold
        self.outlier_multiplier = outlier_multiplier
        self.remove_outliers = remove_outliers

        # ------------------------------------------------------------------
        # Configure a CONSISTENT, polished plotting style once at init.
        # ------------------------------------------------------------------
        self._configure_plotting_style()

        # Create Redis storage connection
        self.redis_storage = RedisProgramStorage(
            RedisProgramStorageConfig(
                redis_url=f"redis://{redis_host}:{redis_port}/{redis_db}",
                key_prefix=redis_prefix,
                max_connections=50,
                connection_pool_timeout=30.0,
                health_check_interval=60,
            )
        )

        logger.info(
            f"Initialized analyzer for Redis at {redis_host}:{redis_port}/{redis_db}"
        )

    async def extract_evolution_data(self) -> pd.DataFrame:
        """Extract all programs and their fitness data from Redis."""

        logger.info("🔍 Extracting programs from Redis...")

        try:
            # Get all programs from Redis
            all_programs = await self.redis_storage.get_all()
            logger.info(f"📊 Found {len(all_programs)} total programs")

            if not all_programs:
                logger.warning("⚠️ No programs found in Redis database")
                return pd.DataFrame()

            # Extract data for each program
            data = []

            for program in all_programs:
                # Get basic program info
                program_data = {
                    "program_id": program.id,
                    "name": program.name or "unnamed",
                    "created_at": program.created_at,
                    "updated_at": program.updated_at,
                    "state": program.state.value,
                    "is_complete": program.is_complete,
                    "generation": program.generation or 0,
                    "parent_count": program.parent_count,
                    "is_root": program.is_root,
                }

                # Extract fitness and other metrics
                if program.metrics:
                    for metric_name, metric_value in program.metrics.items():
                        program_data[f"metric_{metric_name}"] = metric_value

                # Extract lineage information
                if program.lineage:
                    program_data["lineage_parents"] = len(
                        program.lineage.parents
                    )
                    program_data["lineage_mutation"] = program.lineage.mutation
                    program_data["lineage_generation"] = (
                        program.lineage.generation or 0
                    )
                else:
                    program_data["lineage_parents"] = 0
                    program_data["lineage_mutation"] = None
                    program_data["lineage_generation"] = 0

                # Extract metadata
                if program.metadata:
                    for meta_key, meta_value in program.metadata.items():
                        if isinstance(meta_value, (str, int, float, bool)):
                            program_data[f"meta_{meta_key}"] = meta_value

                # Extract stage results
                if program.stage_results:
                    for (
                        stage_name,
                        stage_result,
                    ) in program.stage_results.items():
                        stage_key = f"stage_{stage_name}"
                        program_data[f"{stage_key}_status"] = (
                            stage_result.status.value
                        )
                        program_data[f"{stage_key}_started_at"] = (
                            stage_result.started_at
                        )
                        program_data[f"{stage_key}_finished_at"] = (
                            stage_result.finished_at
                        )
                        program_data[f"{stage_key}_duration"] = (
                            stage_result.duration_seconds()
                        )
                        program_data[f"{stage_key}_has_error"] = (
                            stage_result.error is not None
                        )
                        if stage_result.error:
                            program_data[f"{stage_key}_error_type"] = str(
                                type(stage_result.error).__name__
                            )

                data.append(program_data)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Convert timestamps to datetime if they're strings
            for col in ["created_at", "updated_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            logger.info(f"✅ Extracted {len(df)} programs with data")

            return df

        except Exception as e:
            logger.error(f"❌ Error extracting data: {e}")
            return pd.DataFrame()

    def analyze_fitness_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fitness data and prepare for plotting."""

        if df.empty:
            return {}

        # Store the full dataset for state analysis
        full_df = df.copy()

        # Check if fitness metric exists
        fitness_col = "metric_fitness"
        if fitness_col not in df.columns:
            logger.warning(
                f"⚠️ No fitness metric found. Available metrics: {[col for col in df.columns if col.startswith('metric_')]}"
            )
            return {}

        # Calculate time since start using a more intelligent approach
        # Look for the first program created by evolution (has lineage/parents) rather than initial population
        full_df_sorted = full_df.sort_values("created_at").copy()

        # Find programs with lineage (created by evolution, not initial population)
        evolved_programs = full_df_sorted[full_df_sorted["lineage_parents"] > 0]

        if not evolved_programs.empty:
            # Use the first evolved program as the evolution start time
            evolution_start_time = evolved_programs["created_at"].min()
            logger.info(
                f"🎯 Evolution start time (first evolved program): {evolution_start_time}"
            )
        else:
            # Fallback to earliest program if no evolved programs found
            evolution_start_time = full_df_sorted["created_at"].min()
            logger.info(
                f"⚠️ No evolved programs found, using earliest program as start: {evolution_start_time}"
            )

        # Calculate time since evolution start
        full_df_sorted["time_since_start"] = (
            full_df_sorted["created_at"] - evolution_start_time
        ).dt.total_seconds()

        # Filter programs with valid fitness values (exclude failures and extreme outliers)
        # First, remove known failure values (-1000.0)
        valid_fitness_basic = full_df_sorted[
            full_df_sorted[fitness_col].notna()
            & (full_df_sorted[fitness_col] != -1000.0)
        ]

        if valid_fitness_basic.empty:
            logger.warning(
                "⚠️ No programs with valid fitness values found after basic filtering"
            )
            return {}

        # Apply outlier removal if enabled
        if self.remove_outliers:
            # Remove extreme outliers that disrupt plotting
            fitness_values = valid_fitness_basic[fitness_col]

            # Log fitness value distribution before outlier removal
            logger.info(f"📊 Fitness distribution before outlier removal:")
            logger.info(f"   Min: {fitness_values.min():.4f}")
            logger.info(f"   Max: {fitness_values.max():.4f}")
            logger.info(f"   Mean: {fitness_values.mean():.4f}")
            logger.info(f"   Median: {fitness_values.median():.4f}")
            logger.info(f"   Std: {fitness_values.std():.4f}")

            # 1. Remove extremely negative values (likely errors)
            extreme_outliers = fitness_values < self.extreme_threshold

            # 2. Use IQR method for additional outlier detection on remaining values
            non_extreme = fitness_values[
                fitness_values >= self.extreme_threshold
            ]

            if len(non_extreme) > 0:
                Q1 = non_extreme.quantile(0.25)
                Q3 = non_extreme.quantile(0.75)
                IQR = Q3 - Q1

                # Use configurable multiplier for outlier detection
                lower_bound = Q1 - self.outlier_multiplier * IQR
                upper_bound = Q3 + self.outlier_multiplier * IQR

                # Since fitness values are negative (smaller is worse), we're mainly concerned with the lower bound
                statistical_outliers = (fitness_values < lower_bound) | (
                    fitness_values > upper_bound
                )
            else:
                statistical_outliers = pd.Series(
                    [False] * len(fitness_values), index=fitness_values.index
                )
                lower_bound = self.extreme_threshold
                upper_bound = 0.0

            # Combine outlier detection methods
            all_outliers = extreme_outliers | statistical_outliers

            # Log outlier detection results
            num_extreme = extreme_outliers.sum()
            num_statistical = statistical_outliers.sum()
            num_total_outliers = all_outliers.sum()

            logger.info(f"🔍 Outlier detection results:")
            logger.info(f"   Extreme threshold: {self.extreme_threshold}")
            logger.info(f"   IQR multiplier: {self.outlier_multiplier}")
            logger.info(
                f"   Extreme outliers (< {self.extreme_threshold}): {num_extreme}"
            )
            logger.info(
                f"   Statistical outliers (IQR method): {num_statistical}"
            )
            logger.info(f"   Total outliers removed: {num_total_outliers}")
            logger.info(
                f"   Fitness range for analysis: {lower_bound:.4f} to {upper_bound:.4f}"
            )

            if num_total_outliers > 0:
                outlier_examples = fitness_values[all_outliers].head(5)
                logger.info(
                    f"   Examples of removed outliers: {list(outlier_examples.values)}"
                )

            # Filter out outliers
            valid_fitness = valid_fitness_basic[~all_outliers]

            if valid_fitness.empty:
                logger.warning(
                    "⚠️ No programs with valid fitness values found after outlier removal"
                )
                return {}

            # Log final fitness distribution
            final_fitness = valid_fitness[fitness_col]
            logger.info(f"📊 Final fitness distribution after outlier removal:")
            logger.info(
                f"   Programs remaining: {len(valid_fitness)} (removed {num_total_outliers} outliers)"
            )
            logger.info(f"   Min: {final_fitness.min():.4f}")
            logger.info(f"   Max: {final_fitness.max():.4f}")
            logger.info(f"   Mean: {final_fitness.mean():.4f}")
            logger.info(f"   Median: {final_fitness.median():.4f}")
            logger.info(f"   Std: {final_fitness.std():.4f}")
        else:
            logger.info("⚠️ Outlier removal disabled - using all fitness values")
            valid_fitness = valid_fitness_basic

        if valid_fitness.empty:
            logger.warning("⚠️ No programs with valid fitness values found")
            return {}

        logger.info(
            f"📊 Found {len(valid_fitness)} programs with valid fitness out of {len(full_df)} total programs"
        )
        logger.info(
            f"Fitness range: {valid_fitness[fitness_col].min():.4f} to {valid_fitness[fitness_col].max():.4f}"
        )
        logger.info(
            f"Timeline spans: {evolution_start_time} to {full_df_sorted['created_at'].max()}"
        )
        logger.info(
            f"Time range: {full_df_sorted['time_since_start'].min():.1f}s to {full_df_sorted['time_since_start'].max():.1f}s"
        )

        # Analyze gaps in the timeline
        time_gaps = []
        sorted_times = full_df_sorted["time_since_start"].sort_values().values
        for i in range(1, len(sorted_times)):
            gap = sorted_times[i] - sorted_times[i - 1]
            if gap > 60:  # Report gaps larger than 1 minute
                time_gaps.append((sorted_times[i - 1], sorted_times[i], gap))

        if time_gaps:
            # Sort gaps by size (largest first)
            time_gaps.sort(key=lambda x: x[2], reverse=True)

            logger.info(
                f"⚠️ Found {len(time_gaps)} significant time gaps (>60s):"
            )
            for i, (start, end, gap) in enumerate(
                time_gaps[:5]
            ):  # Show top 5 gaps
                gap_minutes = gap / 60
                logger.info(
                    f"   {i+1}. Gap: {start:.1f}s - {end:.1f}s (duration: {gap:.1f}s = {gap_minutes:.1f} minutes)"
                )

                # Check what programs exist in this gap
                gap_programs = full_df_sorted[
                    (full_df_sorted["time_since_start"] >= start)
                    & (full_df_sorted["time_since_start"] <= end)
                ]
                if len(gap_programs) > 0:
                    logger.info(
                        f"      Programs in gap: {len(gap_programs)} (states: {dict(gap_programs['state'].value_counts())})"
                    )
                else:
                    logger.info(f"      No programs created during this gap")

            if len(time_gaps) > 5:
                logger.info(f"   ... and {len(time_gaps) - 5} more gaps")

            # Highlight the largest gap
            largest_gap = time_gaps[0]
            logger.warning(
                f"🚨 LARGEST GAP: {largest_gap[2]:.1f}s ({largest_gap[2]/60:.1f} minutes) - "
                f"Evolution may have been paused, stopped, or experiencing issues"
            )
        else:
            logger.info("✅ No significant time gaps found")

        # Create timeline for fitness analysis (only valid fitness programs)
        timeline_df = valid_fitness.copy()

        # Calculate running best fitness
        timeline_df["running_best_fitness"] = (
            timeline_df[fitness_col].expanding().max()
        )

        # Group by generation if available
        generation_stats = None
        if "generation" in timeline_df.columns:
            generation_stats = (
                timeline_df.groupby("generation")
                .agg(
                    {
                        fitness_col: ["count", "mean", "max", "min"],
                        "created_at": "min",
                    }
                )
                .round(4)
            )

            generation_stats.columns = [
                "program_count",
                "mean_fitness",
                "max_fitness",
                "min_fitness",
                "generation_start",
            ]
            generation_stats = generation_stats.reset_index()

            logger.info(f"📈 Generation Statistics:")
            logger.info(generation_stats)

        return {
            "timeline_df": timeline_df,
            "full_df": full_df_sorted,  # Include full dataset with proper timeline
            "generation_stats": generation_stats,
            "fitness_col": fitness_col,
            "start_time": evolution_start_time,
            "total_programs": len(valid_fitness),
            "total_all_programs": len(full_df),
            "best_fitness": valid_fitness[fitness_col].max(),
            "worst_fitness": valid_fitness[fitness_col].min(),
            "mean_fitness": valid_fitness[fitness_col].mean(),
        }

    def plot_fitness_evolution(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
        rolling_window: int = 50,  # Number of programs for rolling statistics
    ):
        """Create comprehensive fitness evolution plots including running mean and std."""

        if not fitness_analysis:
            logger.warning("No fitness data to plot")
            return

        timeline_df = fitness_analysis["timeline_df"].copy()
        fitness_col = fitness_analysis["fitness_col"]

        # Sort by time to ensure proper rolling calculations
        timeline_df = timeline_df.sort_values("time_since_start").reset_index(drop=True)

        # Calculate running statistics
        logger.info(f"📊 Calculating running statistics with window size: {rolling_window}")
        
        # Running best fitness (cumulative max)
        timeline_df["running_best_fitness"] = timeline_df[fitness_col].expanding().max()
        
        # Running mean and std with rolling window
        timeline_df["running_mean_fitness"] = timeline_df[fitness_col].rolling(
            window=rolling_window, min_periods=1, center=False
        ).mean()
        
        timeline_df["running_std_fitness"] = timeline_df[fitness_col].rolling(
            window=rolling_window, min_periods=1, center=False
        ).std()
        
        # Calculate confidence intervals (mean ± std)
        timeline_df["running_mean_plus_std"] = (
            timeline_df["running_mean_fitness"] + timeline_df["running_std_fitness"]
        )
        timeline_df["running_mean_minus_std"] = (
            timeline_df["running_mean_fitness"] - timeline_df["running_std_fitness"]
        )

        # Store running statistics in fitness_analysis for JSON export
        fitness_analysis["running_statistics"] = {
            "rolling_window": rolling_window,
            "timeline_data": timeline_df[[
                "time_since_start", 
                "running_best_fitness", 
                "running_mean_fitness", 
                "running_std_fitness",
                "running_mean_plus_std",
                "running_mean_minus_std",
                fitness_col
            ]].to_dict('records')
        }

        # Create main figure with subplots - increased size for better readability
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        fig.suptitle(
            "Evolution Fitness Analysis with Running Statistics", fontsize=18, fontweight="bold"
        )

        # 1. Best Fitness vs Time with Running Mean
        ax1 = axes[0, 0]
        ax1.plot(
            timeline_df["time_since_start"],
            timeline_df["running_best_fitness"],
            linewidth=3,
            color="green",
            label="Running Best Fitness",
        )
        ax1.plot(
            timeline_df["time_since_start"],
            timeline_df["running_mean_fitness"],
            linewidth=2,
            color="orange",
            label=f"Running Mean (n={rolling_window})",
        )
        ax1.scatter(
            timeline_df["time_since_start"],
            timeline_df[fitness_col],
            alpha=0.4,
            s=15,
            color="blue",
            label="Individual Fitness",
        )
        ax1.set_xlabel("Time Since Start (seconds)")
        ax1.set_ylabel("Fitness (negative enclosing hexagon side length)")
        ax1.set_title("Best Fitness and Running Mean vs Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Running Mean with Confidence Band
        ax2 = axes[0, 1]
        ax2.plot(
            timeline_df["time_since_start"],
            timeline_df["running_mean_fitness"],
            linewidth=2,
            color="orange",
            label=f"Running Mean (n={rolling_window})",
        )
        ax2.fill_between(
            timeline_df["time_since_start"],
            timeline_df["running_mean_minus_std"],
            timeline_df["running_mean_plus_std"],
            alpha=0.3,
            color="orange",
            label="Mean ± 1 Std Dev",
        )
        ax2.scatter(
            timeline_df["time_since_start"],
            timeline_df[fitness_col],
            alpha=0.2,
            s=10,
            color="blue",
            label="Individual Fitness",
        )
        ax2.set_xlabel("Time Since Start (seconds)")
        ax2.set_ylabel("Fitness")
        ax2.set_title("Running Mean with Standard Deviation Band")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Running Standard Deviation
        ax3 = axes[1, 0]
        ax3.plot(
            timeline_df["time_since_start"],
            timeline_df["running_std_fitness"],
            linewidth=2,
            color="red",
            label=f"Running Std Dev (n={rolling_window})",
        )
        ax3.set_xlabel("Time Since Start (seconds)")
        ax3.set_ylabel("Standard Deviation")
        ax3.set_title("Running Standard Deviation vs Time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Fitness Distribution
        ax4 = axes[1, 1]
        ax4.hist(
            timeline_df[fitness_col],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax4.axvline(
            timeline_df[fitness_col].mean(),
            color="red",
            linestyle="--",
            label=f"Overall Mean: {timeline_df[fitness_col].mean():.3f}",
        )
        ax4.axvline(
            timeline_df[fitness_col].max(),
            color="green",
            linestyle="--",
            label=f"Best: {timeline_df[fitness_col].max():.3f}",
        )
        ax4.axvline(
            timeline_df["running_mean_fitness"].iloc[-1],
            color="orange",
            linestyle="--",
            label=f"Final Running Mean: {timeline_df['running_mean_fitness'].iloc[-1]:.3f}",
        )
        ax4.set_xlabel("Fitness")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Fitness Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Generation-based analysis (if available)
        ax5 = axes[2, 0]
        if fitness_analysis["generation_stats"] is not None:
            gen_stats = fitness_analysis["generation_stats"]
            ax5.plot(
                gen_stats["generation"],
                gen_stats["max_fitness"],
                marker="o",
                linewidth=2,
                color="green",
                label="Best Fitness",
            )
            ax5.plot(
                gen_stats["generation"],
                gen_stats["mean_fitness"],
                marker="s",
                linewidth=2,
                color="orange",
                label="Mean Fitness",
            )
            ax5.fill_between(
                gen_stats["generation"],
                gen_stats["min_fitness"],
                gen_stats["max_fitness"],
                alpha=0.2,
                color="blue",
                label="Fitness Range",
            )
            ax5.set_xlabel("Generation")
            ax5.set_ylabel("Fitness")
            ax5.set_title("Fitness by Generation")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(
                0.5,
                0.5,
                "No generation data available",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=12,
            )
            ax5.set_title("Fitness by Generation")

        # 6. Program State Analysis
        ax6 = axes[2, 1]
        if "state" in timeline_df.columns:
            # Use the full dataset for state distribution, not just valid fitness programs
            full_df = fitness_analysis.get(
                "full_df", timeline_df
            )  # Fallback to timeline_df if full_df not available
            state_counts = full_df["state"].value_counts()
            ax6.pie(
                state_counts.values,
                labels=state_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=plt.cm.Set3.colors,
            )
            ax6.set_title("Program States Distribution")
        else:
            ax6.text(
                0.5,
                0.5,
                "No state data available",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=12,
            )
            ax6.set_title("Program States Distribution")

        plt.tight_layout(pad=2.0)  # Increased padding for better spacing

        if save_plots:
            self._save_fig(fig, output_folder / "evolution_analysis_overview")

        plt.show()

        # Additional detailed plot: Comprehensive fitness timeline
        plt.figure(figsize=(16, 10))

        # Create subplots for detailed timeline
        fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        # Top subplot: All fitness metrics
        ax_top.scatter(
            timeline_df["time_since_start"],
            timeline_df[fitness_col],
            alpha=0.3,
            s=20,
            color="lightblue",
            label="Individual Programs",
        )

        # Plot running best with thicker line
        ax_top.plot(
            timeline_df["time_since_start"],
            timeline_df["running_best_fitness"],
            linewidth=3,
            color="darkgreen",
            label="Running Best Fitness",
        )

        # Plot running mean
        ax_top.plot(
            timeline_df["time_since_start"],
            timeline_df["running_mean_fitness"],
            linewidth=2,
            color="orange",
            label=f"Running Mean (n={rolling_window})",
        )

        # Add confidence band
        ax_top.fill_between(
            timeline_df["time_since_start"],
            timeline_df["running_mean_minus_std"],
            timeline_df["running_mean_plus_std"],
            alpha=0.2,
            color="orange",
            label="Mean ± 1 Std Dev",
        )

        # Highlight improvements
        improvements = timeline_df[
            timeline_df[fitness_col] == timeline_df["running_best_fitness"]
        ]
        ax_top.scatter(
            improvements["time_since_start"],
            improvements[fitness_col],
            color="red",
            s=100,
            zorder=5,
            label="New Best Fitness",
        )

        ax_top.set_ylabel("Fitness (negative enclosing hexagon side length)")
        ax_top.set_title(
            f"Comprehensive Fitness Evolution Over Time\n(Rolling window: {rolling_window} programs)"
        )
        ax_top.legend()
        ax_top.grid(True, alpha=0.3)

        # Bottom subplot: Running standard deviation
        ax_bottom.plot(
            timeline_df["time_since_start"],
            timeline_df["running_std_fitness"],
            linewidth=2,
            color="red",
            label=f"Running Std Dev (n={rolling_window})",
        )
        ax_bottom.set_xlabel("Time Since Start (seconds)")
        ax_bottom.set_ylabel("Standard Deviation")
        ax_bottom.set_title("Running Standard Deviation Over Time")
        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3)

        plt.tight_layout(pad=2.0)

        if save_plots:
            self._save_fig(
                fig2, output_folder / "comprehensive_fitness_evolution_timeline"
            )

        plt.show()

        # Log running statistics summary
        logger.info(f"\n📊 RUNNING STATISTICS SUMMARY (window={rolling_window}):")
        logger.info("=" * 80)
        logger.info(f"  Final running mean: {timeline_df['running_mean_fitness'].iloc[-1]:.4f}")
        logger.info(f"  Final running std: {timeline_df['running_std_fitness'].iloc[-1]:.4f}")
        logger.info(f"  Overall best fitness: {timeline_df['running_best_fitness'].iloc[-1]:.4f}")
        logger.info(f"  Mean improvement rate: {(timeline_df['running_best_fitness'].iloc[-1] - timeline_df['running_best_fitness'].iloc[0]) / len(timeline_df):.6f} per program")

        # Calculate and log key milestones
        if len(timeline_df) > rolling_window:
            early_mean = timeline_df['running_mean_fitness'].iloc[rolling_window-1]
            late_mean = timeline_df['running_mean_fitness'].iloc[-1]
            mean_improvement = late_mean - early_mean
            logger.info(f"  Mean fitness improvement: {mean_improvement:.4f} (from {early_mean:.4f} to {late_mean:.4f})")

        # New plot: Full timeline showing all programs (including failed ones)
        # Get the full dataset
        full_df = fitness_analysis["full_df"]

        # Create subplots - increased size for better readability
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

        # Top subplot: All programs over time (by state)
        for state in full_df["state"].unique():
            state_data = full_df[full_df["state"] == state]
            ax1.scatter(
                state_data["time_since_start"],
                [1] * len(state_data),
                alpha=0.6,
                s=20,
                label=f"{state} ({len(state_data)} programs)",
            )

        ax1.set_ylabel("Programs")
        ax1.set_title("All Programs Timeline (by State)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Bottom subplot: Fitness programs with running statistics
        ax2.scatter(
            timeline_df["time_since_start"],
            timeline_df[fitness_col],
            alpha=0.4,
            s=30,
            color="lightblue",
            label="Individual Programs",
        )
        ax2.plot(
            timeline_df["time_since_start"],
            timeline_df["running_best_fitness"],
            linewidth=3,
            color="darkgreen",
            label="Running Best Fitness",
        )
        ax2.plot(
            timeline_df["time_since_start"],
            timeline_df["running_mean_fitness"],
            linewidth=2,
            color="orange",
            label=f"Running Mean (n={rolling_window})",
        )

        # Highlight improvements
        improvements = timeline_df[
            timeline_df[fitness_col] == timeline_df["running_best_fitness"]
        ]
        ax2.scatter(
            improvements["time_since_start"],
            improvements[fitness_col],
            color="red",
            s=100,
            zorder=5,
            label="New Best Fitness",
        )

        ax2.set_xlabel("Time Since Start (seconds)")
        ax2.set_ylabel("Fitness (negative enclosing hexagon side length)")
        ax2.set_title("Fitness Evolution with Running Mean")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(pad=2.0)  # Increased padding for better spacing

        if save_plots:
            self._save_fig(fig, output_folder / "full_timeline_analysis")

        plt.show()

    def plot_program_stage_statistics(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Create comprehensive program state statistics plots."""

        if not fitness_analysis:
            logger.warning("No data to plot")
            return

        full_df = fitness_analysis["full_df"]

        if "state" not in full_df.columns:
            logger.warning(
                "No state data available for program state statistics"
            )
            return

        # Create figure with subplots - increased size for better readability
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(
            "Program State Statistics Analysis", fontsize=18, fontweight="bold"
        )

        # 1. Program State Distribution (Bar Chart)
        ax1 = axes[0, 0]
        state_counts = full_df["state"].value_counts()
        colors = plt.cm.Set3.colors[: len(state_counts)]

        bars = ax1.bar(
            range(len(state_counts)),
            state_counts.values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_xlabel("Program State")
        ax1.set_ylabel("Number of Programs")
        ax1.set_title("Program State Distribution")
        ax1.set_xticks(range(len(state_counts)))
        ax1.set_xticklabels(
            state_counts.index, rotation=45, ha="right", fontsize=10
        )

        # Add value labels on bars
        for bar, count in zip(bars, state_counts.values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(state_counts.values) * 0.01,
                f"{count}\n({count/len(full_df)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Program State Distribution (Pie Chart)
        ax2 = axes[0, 1]
        ax2.pie(
            state_counts.values,
            labels=state_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.05] * len(state_counts),
        )
        ax2.set_title("Program State Distribution (Pie Chart)")

        # 3. Programs by State Over Time
        ax3 = axes[1, 0]

        # Create a timeline showing when programs in each state were created
        for i, state in enumerate(state_counts.index):
            state_data = full_df[full_df["state"] == state]
            if not state_data.empty:
                ax3.scatter(
                    state_data["time_since_start"],
                    [i] * len(state_data),
                    alpha=0.6,
                    s=20,
                    label=f"{state} ({len(state_data)} programs)",
                    color=colors[i % len(colors)],
                )

        ax3.set_xlabel("Time Since Start (seconds)")
        ax3.set_ylabel("Program State")
        ax3.set_title("Program States Timeline")
        ax3.set_yticks(range(len(state_counts)))
        ax3.set_yticklabels(state_counts.index)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(True, alpha=0.3, axis="x")

        # 4. State Persistence Analysis (how long programs stay in each state)
        ax4 = axes[1, 1]

        if "updated_at" in full_df.columns and "created_at" in full_df.columns:
            # Calculate current time since last update for each program
            # Since we don't track when programs entered their current state, we'll use
            # the time since last update as a proxy for how long they've been in current state
            current_time = datetime.now(timezone.utc)
            full_df["state_persistence_seconds"] = (
                current_time - full_df["updated_at"]
            ).dt.total_seconds()

            # Filter out negative persistence times (data issues)
            # Don't filter out long persistence times as programs can legitimately stay in states for hours
            valid_persistence = full_df[
                full_df["state_persistence_seconds"] >= 0
            ]

            if not valid_persistence.empty:
                # Box plot of persistence time by state
                state_persistence_data = [
                    valid_persistence[valid_persistence["state"] == state][
                        "state_persistence_seconds"
                    ].values
                    for state in state_counts.index
                    if len(
                        valid_persistence[valid_persistence["state"] == state]
                    )
                    > 0
                ]
                state_labels = [
                    state
                    for state in state_counts.index
                    if len(
                        valid_persistence[valid_persistence["state"] == state]
                    )
                    > 0
                ]

                if state_persistence_data:
                    bp = ax4.boxplot(
                        state_persistence_data,
                        labels=state_labels,
                        patch_artist=True,
                    )

                    # Color the boxes
                    for patch, color in zip(
                        bp["boxes"], colors[: len(state_persistence_data)]
                    ):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax4.set_xlabel("Program State")
                    ax4.set_ylabel("Time Since Last Update (seconds)")
                    ax4.set_title("Time Since Last Update Distribution")
                    ax4.tick_params(axis="x", rotation=45, labelsize=10)
                    ax4.grid(True, alpha=0.3, axis="y")
                else:
                    ax4.text(
                        0.5,
                        0.5,
                        "No valid persistence data available",
                        ha="center",
                        va="center",
                        transform=ax4.transAxes,
                        fontsize=12,
                    )
                    ax4.set_title("State Persistence Distribution")
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No valid persistence data available",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=12,
                )
                ax4.set_title("State Persistence Distribution")
        else:
            ax4.text(
                0.5,
                0.5,
                "No timestamp data available for persistence analysis",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title("State Persistence Distribution")

        plt.tight_layout(pad=3.0)  # Increased padding for better spacing

        if save_plots:
            self._save_fig(fig, output_folder / "program_stage_statistics")

        plt.show()

        # Additional detailed analysis: State statistics table
        logger.info("\n📊 PROGRAM STATE STATISTICS:")
        logger.info("=" * 80)

        for state in state_counts.index:
            state_data = full_df[full_df["state"] == state]
            count = len(state_data)
            percentage = count / len(full_df) * 100

            logger.info(f"\n{state.upper()}:")
            logger.info(f"  Count: {count} programs ({percentage:.1f}%)")

            # Show generation info if available
            if "generation" in state_data.columns:
                gen_stats = state_data["generation"].describe()
                logger.info(
                    f"  Generation range: {gen_stats['min']:.0f} - {gen_stats['max']:.0f}"
                )
                logger.info(f"  Mean generation: {gen_stats['mean']:.1f}")

            # Show time since update info if available
            if "state_persistence_seconds" in state_data.columns:
                time_data = state_data[
                    state_data["state_persistence_seconds"] >= 0
                ]["state_persistence_seconds"]
                if not time_data.empty:
                    time_stats = time_data.describe()
                    logger.info(
                        f"  Time since update: {time_stats['min']:.1f}s - {time_stats['max']:.1f}s"
                    )
                    logger.info(
                        f"  Mean time since update: {time_stats['mean']:.1f}s"
                    )
                    logger.info(
                        f"  Median time since update: {time_stats['50%']:.1f}s"
                    )
                else:
                    logger.info(f"  Time since update: No valid data")

    def plot_state_persistence_analysis(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Create detailed state persistence analysis plots for each program state."""

        if not fitness_analysis:
            logger.warning("No data to plot")
            return

        full_df = fitness_analysis["full_df"]

        if (
            "state" not in full_df.columns
            or "updated_at" not in full_df.columns
            or "created_at" not in full_df.columns
        ):
            logger.warning(
                "Missing required data for state persistence analysis"
            )
            return

        # Calculate current state persistence time for each program
        # Since we don't track when programs entered their current state, we'll use
        # the time since last update as a proxy for how long they've been in current state
        current_time = datetime.now(timezone.utc)
        full_df["state_persistence_seconds"] = (
            current_time - full_df["updated_at"]
        ).dt.total_seconds()

        # Filter out negative persistence times (data issues)
        # Don't filter out long persistence times as programs can legitimately stay in states for hours
        valid_persistence = full_df[full_df["state_persistence_seconds"] >= 0]

        # Log persistence data quality
        total_programs = len(full_df)
        valid_programs = len(valid_persistence)
        invalid_programs = total_programs - valid_programs

        if invalid_programs > 0:
            logger.info(
                f"⚠️ Persistence data quality: {valid_programs}/{total_programs} programs have valid persistence data"
            )
            logger.info(
                f"   {invalid_programs} programs excluded (negative persistence time)"
            )
        else:
            logger.info(
                f"✅ Persistence data quality: All {valid_programs} programs have valid persistence data"
            )

        if valid_persistence.empty:
            logger.warning("No valid persistence data available")
            return

        # Log data ranges for debugging
        logger.info(f"📊 Time since last update data ranges:")
        for state in valid_persistence["state"].unique():
            state_data = valid_persistence[valid_persistence["state"] == state][
                "state_persistence_seconds"
            ]
            logger.info(
                f"  {state}: {state_data.min():.1f}s - {state_data.max():.1f}s (mean: {state_data.mean():.1f}s)"
            )

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # Get unique states
        states = valid_persistence["state"].unique()
        colors = plt.cm.Set3.colors[: len(states)]

        # Create figure with subplots - increased size for better readability
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(
            "Time Since Last Update Analysis by Program State",
            fontsize=18,
            fontweight="bold",
        )

        # 1. Overall Persistence Distribution (All States Combined)
        ax1 = axes[0, 0]
        # ------------------------------------------------------------------
        # 1. Program counts by state (replaces overall histogram)
        # ------------------------------------------------------------------
        state_counts = valid_persistence["state"].value_counts()
        bars = ax1.bar(
            range(len(state_counts)),
            state_counts.values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_xlabel("Program State")
        ax1.set_ylabel("Number of Programs")
        ax1.set_title("Number of Programs by State")
        ax1.set_xticks(range(len(state_counts)))
        ax1.set_xticklabels(
            state_counts.index, rotation=45, ha="right", fontsize=10
        )
        for bar, count in zip(bars, state_counts.values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(state_counts.values) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Persistence Distribution by State (Box Plot)
        ax2 = axes[0, 1]
        state_persistence_data = [
            valid_persistence[valid_persistence["state"] == state][
                "state_persistence_seconds"
            ].values
            for state in states
        ]

        bp = ax2.boxplot(
            state_persistence_data, labels=states, patch_artist=True
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xlabel("Program State")
        ax2.set_ylabel("Time Since Last Update (seconds)")
        ax2.set_title("Time Since Last Update by State (Box Plot)")
        ax2.tick_params(axis="x", rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Persistence Distribution by State (Violin Plot – replaces per-state histograms)
        ax3 = axes[1, 0]
        vp = ax3.violinplot(
            state_persistence_data, positions=range(len(states)), showmeans=True
        )
        # Colour each violin body to match state colours
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        ax3.set_xlabel("Program State")
        ax3.set_ylabel("Time Since Last Update (seconds)")
        ax3.set_title("Time Since Last Update by State (Violin Plot)")
        ax3.set_xticks(range(len(states)))
        ax3.set_xticklabels(states, rotation=45, ha="right", fontsize=10)
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Persistence Statistics by State (Bar Chart)
        ax4 = axes[1, 1]

        # Calculate statistics for each state
        state_stats = []
        for state in states:
            state_data = valid_persistence[valid_persistence["state"] == state][
                "state_persistence_seconds"
            ]
            state_stats.append(
                {
                    "state": state,
                    "count": len(state_data),
                    "mean": state_data.mean(),
                    "median": state_data.median(),
                    "std": state_data.std(),
                    "min": state_data.min(),
                    "max": state_data.max(),
                }
            )

        # Create bar chart for mean persistence by state
        states_list = [stat["state"] for stat in state_stats]
        means = [stat["mean"] for stat in state_stats]
        counts = [stat["count"] for stat in state_stats]

        bars = ax4.bar(
            range(len(states_list)),
            means,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax4.set_xlabel("Program State")
        ax4.set_ylabel("Mean Time Since Last Update (seconds)")
        ax4.set_title("Mean Time Since Last Update by State")
        ax4.set_xticks(range(len(states_list)))
        ax4.set_xticklabels(states_list, rotation=45, ha="right", fontsize=10)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(means) * 0.01,
                f"n={count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(pad=3.0)  # Increased padding for better spacing

        if save_plots:
            self._save_fig(fig, output_folder / "state_persistence_analysis")

        plt.show()

        # Additional detailed analysis
        logger.info("\n⏱️ TIME SINCE LAST UPDATE ANALYSIS:")
        logger.info("=" * 80)

        for stat in state_stats:
            logger.info(f"\n{stat['state'].upper()}:")
            logger.info(f"  Count: {stat['count']} programs")
            logger.info(f"  Mean time since update: {stat['mean']:.1f}s")
            logger.info(f"  Median time since update: {stat['median']:.1f}s")
            logger.info(f"  Std dev: {stat['std']:.1f}s")
            logger.info(f"  Range: {stat['min']:.1f}s - {stat['max']:.1f}s")

            # Calculate percentiles
            state_data = valid_persistence[
                valid_persistence["state"] == stat["state"]
            ]["state_persistence_seconds"]
            p25 = state_data.quantile(0.25)
            p75 = state_data.quantile(0.75)
            logger.info(f"  25th percentile: {p25:.1f}s")
            logger.info(f"  75th percentile: {p75:.1f}s")

        # Overall statistics
        logger.info(f"\n📊 OVERALL TIME SINCE UPDATE STATISTICS:")
        logger.info(
            f"  Total programs with valid data: {len(valid_persistence)}"
        )
        logger.info(
            f"  Overall mean time since update: {valid_persistence['state_persistence_seconds'].mean():.1f}s"
        )
        logger.info(
            f"  Overall median time since update: {valid_persistence['state_persistence_seconds'].median():.1f}s"
        )
        logger.info(
            f"  Overall std dev: {valid_persistence['state_persistence_seconds'].std():.1f}s"
        )
        logger.info(
            f"  Overall range: {valid_persistence['state_persistence_seconds'].min():.1f}s - {valid_persistence['state_persistence_seconds'].max():.1f}s"
        )

        # Identify outliers (programs with time since update > 2 standard deviations from mean)
        mean_time = valid_persistence["state_persistence_seconds"].mean()
        std_time = valid_persistence["state_persistence_seconds"].std()
        outliers = valid_persistence[
            valid_persistence["state_persistence_seconds"]
            > mean_time + 2 * std_time
        ]

        if not outliers.empty:
            logger.info(f"\n🚨 TIME SINCE UPDATE OUTLIERS (>2σ from mean):")
            logger.info(f"  Found {len(outliers)} outliers")
            for _, outlier in outliers.head(
                10
            ).iterrows():  # Show top 10 outliers
                logger.info(
                    f"    Program {outlier['program_id'][:12]}...: {outlier['state_persistence_seconds']:.1f}s ({outlier['state']})"
                )
            if len(outliers) > 10:
                logger.info(f"    ... and {len(outliers) - 10} more outliers")

        # Save detailed statistics to file
        persistence_stats_path = (
            output_folder / "time_since_update_statistics.txt"
        )
        with open(persistence_stats_path, "w") as f:
            f.write("Time Since Last Update Statistics by Program State\n")
            f.write("=" * 50 + "\n\n")

            for stat in state_stats:
                f.write(f"{stat['state'].upper()}:\n")
                f.write(f"  Count: {stat['count']} programs\n")
                f.write(f"  Mean time since update: {stat['mean']:.1f}s\n")
                f.write(f"  Median time since update: {stat['median']:.1f}s\n")
                f.write(f"  Std dev: {stat['std']:.1f}s\n")
                f.write(f"  Range: {stat['min']:.1f}s - {stat['max']:.1f}s\n\n")

            f.write(f"OVERALL STATISTICS:\n")
            f.write(
                f"  Total programs with valid data: {len(valid_persistence)}\n"
            )
            f.write(
                f"  Overall mean time since update: {valid_persistence['state_persistence_seconds'].mean():.1f}s\n"
            )
            f.write(
                f"  Overall median time since update: {valid_persistence['state_persistence_seconds'].median():.1f}s\n"
            )
            f.write(
                f"  Overall std dev: {valid_persistence['state_persistence_seconds'].std():.1f}s\n"
            )
            f.write(
                f"  Overall range: {valid_persistence['state_persistence_seconds'].min():.1f}s - {valid_persistence['state_persistence_seconds'].max():.1f}s\n"
            )

        logger.info(
            f"✅ Saved detailed time since update statistics to {persistence_stats_path}"
        )

    def plot_island_statistics(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Create comprehensive island statistics analysis."""

        if not fitness_analysis:
            logger.warning("No data to plot")
            return

        full_df = fitness_analysis["full_df"]

        # Check if island data is available
        island_col = "meta_current_island"
        if island_col not in full_df.columns:
            logger.warning(
                f"⚠️ No island data found. Available metadata columns: {[col for col in full_df.columns if col.startswith('meta_')]}"
            )
            return

        # Filter out programs without island data
        island_data = full_df[full_df[island_col].notna()]

        if island_data.empty:
            logger.warning("No programs with island data found")
            return

        # IMPORTANT: For island visualization, only consider EVOLVING programs
        evolving_island_data = island_data[island_data["state"] == "evolving"]

        if evolving_island_data.empty:
            logger.warning("No EVOLVING programs with island data found")
            return

        logger.info(
            f"📊 Found {len(island_data)} total programs with island data out of {len(full_df)} total programs"
        )
        logger.info(
            f"📊 Found {len(evolving_island_data)} EVOLVING programs with island data"
        )
        logger.info(
            f"📊 Unique islands: {list(evolving_island_data[island_col].unique())}"
        )

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # Create figure with subplots - increased size for better readability
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(
            "Island Statistics Analysis (EVOLVING Programs Only)",
            fontsize=18,
            fontweight="bold",
        )

        # 1. Island Distribution (Bar Chart) - EVOLVING programs only
        ax1 = axes[0, 0]
        island_counts = evolving_island_data[island_col].value_counts()
        colors = plt.cm.Set3.colors[: len(island_counts)]

        bars = ax1.bar(
            range(len(island_counts)),
            island_counts.values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_xlabel("Island")
        ax1.set_ylabel("Number of EVOLVING Programs")
        ax1.set_title("EVOLVING Program Distribution by Island")
        ax1.set_xticks(range(len(island_counts)))
        ax1.set_xticklabels(
            island_counts.index, rotation=45, ha="right", fontsize=10
        )

        # Add value labels on bars
        for bar, count in zip(bars, island_counts.values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(island_counts.values) * 0.01,
                f"{count}\n({count/len(evolving_island_data)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Island Distribution (Pie Chart) - EVOLVING programs only
        ax2 = axes[0, 1]
        ax2.pie(
            island_counts.values,
            labels=island_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.05] * len(island_counts),
        )
        ax2.set_title("EVOLVING Program Distribution by Island (Pie Chart)")

        # 3. Island vs State Distribution - Show full state breakdown for context
        ax3 = axes[0, 2]
        island_state_pivot = (
            pd.crosstab(
                island_data[island_col], island_data["state"], normalize="index"
            )
            * 100
        )

        island_state_pivot.plot(
            kind="bar", stacked=True, ax=ax3, colormap="Set3"
        )
        ax3.set_xlabel("Island")
        ax3.set_ylabel("Percentage")
        ax3.set_title("All Program States by Island")
        ax3.tick_params(axis="x", rotation=45, labelsize=10)
        ax3.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Island Timeline (when EVOLVING programs were created)
        ax4 = axes[1, 0]

        # Consistent y-axis ordering for islands
        islands_sorted = sorted(evolving_island_data[island_col].unique())
        island_to_y = {name: idx for idx, name in enumerate(islands_sorted)}

        for island in islands_sorted:
            island_programs = evolving_island_data[
                evolving_island_data[island_col] == island
            ]
            y = np.full(len(island_programs), island_to_y[island])
            ax4.scatter(
                island_programs["time_since_start"],
                y,
                alpha=0.6,
                s=20,
                label=f"{island} ({len(island_programs)} programs)",
            )

        ax4.set_xlabel("Time Since Start (seconds)")
        ax4.set_ylabel("Island")
        ax4.set_title("EVOLVING Program Creation Timeline by Island")
        ax4.set_yticks(list(island_to_y.values()))
        ax4.set_yticklabels(islands_sorted)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax4.grid(True, alpha=0.3, axis="x")

        # 5. Fitness by Island (EVOLVING programs only)
        ax5 = axes[1, 1]
        fitness_col = fitness_analysis.get("fitness_col", "metric_fitness")

        if fitness_col in evolving_island_data.columns:
            # Filter programs with valid fitness
            valid_fitness_islands = evolving_island_data[
                evolving_island_data[fitness_col].notna()
                & (evolving_island_data[fitness_col] != -1000.0)
            ]

            if not valid_fitness_islands.empty:
                # Box plot of fitness by island
                island_fitness_data = [
                    valid_fitness_islands[
                        valid_fitness_islands[island_col] == island
                    ][fitness_col].values
                    for island in valid_fitness_islands[island_col].unique()
                ]
                island_names = valid_fitness_islands[island_col].unique()

                if island_fitness_data:
                    bp = ax5.boxplot(
                        island_fitness_data,
                        labels=island_names,
                        patch_artist=True,
                    )

                    # Color the boxes
                    for patch, color in zip(
                        bp["boxes"], colors[: len(island_fitness_data)]
                    ):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax5.set_xlabel("Island")
                    ax5.set_ylabel("Fitness")
                    ax5.set_title(
                        "EVOLVING Program Fitness Distribution by Island"
                    )
                    ax5.tick_params(axis="x", rotation=45, labelsize=10)
                    ax5.grid(True, alpha=0.3, axis="y")
                else:
                    ax5.text(
                        0.5,
                        0.5,
                        "No valid fitness data available",
                        ha="center",
                        va="center",
                        transform=ax5.transAxes,
                        fontsize=12,
                    )
                    ax5.set_title(
                        "EVOLVING Program Fitness Distribution by Island"
                    )
            else:
                ax5.text(
                    0.5,
                    0.5,
                    "No valid fitness data available",
                    ha="center",
                    va="center",
                    transform=ax5.transAxes,
                    fontsize=12,
                )
                ax5.set_title("EVOLVING Program Fitness Distribution by Island")
        else:
            ax5.text(
                0.5,
                0.5,
                "No fitness data available",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=12,
            )
            ax5.set_title("EVOLVING Program Fitness Distribution by Island")

        # 6. Island Statistics Summary - Show both EVOLVING and full statistics
        ax6 = axes[1, 2]

        # Calculate statistics for each island (both EVOLVING and full)
        island_stats = []
        for island in island_data[island_col].unique():
            island_programs = island_data[island_data[island_col] == island]
            evolving_island_programs = evolving_island_data[
                evolving_island_data[island_col] == island
            ]

            stats = {
                "island": island,
                "total_count": len(island_programs),
                "evolving_count": len(evolving_island_programs),
                "discarded_count": len(
                    island_programs[island_programs["state"] == "discarded"]
                ),
                "other_states_count": len(
                    island_programs[
                        ~island_programs["state"].isin(
                            ["evolving", "discarded"]
                        )
                    ]
                ),
            }

            # Add fitness stats for EVOLVING programs only
            if fitness_col in evolving_island_programs.columns:
                valid_fitness = evolving_island_programs[
                    evolving_island_programs[fitness_col].notna()
                    & (evolving_island_programs[fitness_col] != -1000.0)
                ]
                if not valid_fitness.empty:
                    stats["mean_fitness"] = valid_fitness[fitness_col].mean()
                    stats["best_fitness"] = valid_fitness[fitness_col].max()
                    stats["fitness_count"] = len(valid_fitness)
                else:
                    stats["mean_fitness"] = None
                    stats["best_fitness"] = None
                    stats["fitness_count"] = 0
            else:
                stats["mean_fitness"] = None
                stats["best_fitness"] = None
                stats["fitness_count"] = 0

            island_stats.append(stats)

        # Create summary table
        summary_data = []
        for stat in island_stats:
            summary_data.append(
                [
                    stat["island"],
                    stat["evolving_count"],  # Show EVOLVING count as primary
                    stat["total_count"],  # Show total count for reference
                    (
                        f"{stat['mean_fitness']:.3f}"
                        if stat["mean_fitness"] is not None
                        else "N/A"
                    ),
                    (
                        f"{stat['best_fitness']:.3f}"
                        if stat["best_fitness"] is not None
                        else "N/A"
                    ),
                    stat["fitness_count"],
                ]
            )

        # Create table
        table_data = [
            [
                "Island",
                "EVOLVING",
                "Total",
                "Mean Fitness",
                "Best Fitness",
                "Fitness Count",
            ]
        ] + summary_data
        table = ax6.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc="center",
            loc="center",
            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        ax6.set_title("Island Statistics Summary (EVOLVING Programs)")
        ax6.axis("off")

        plt.tight_layout(pad=3.0)  # Increased padding for better spacing

        if save_plots:
            self._save_fig(fig, output_folder / "island_statistics")

        plt.show()

        # Additional detailed analysis
        logger.info("\n🏝️ ISLAND STATISTICS (EVOLVING PROGRAMS):")
        logger.info("=" * 80)

        for stat in island_stats:
            logger.info(f"\n{stat['island'].upper()}:")
            logger.info(f"  Total programs: {stat['total_count']}")
            logger.info(
                f"  EVOLVING: {stat['evolving_count']} ({stat['evolving_count']/stat['total_count']*100:.1f}%)"
            )
            logger.info(
                f"  Discarded: {stat['discarded_count']} ({stat['discarded_count']/stat['total_count']*100:.1f}%)"
            )
            logger.info(
                f"  Other states: {stat['other_states_count']} ({stat['other_states_count']/stat['total_count']*100:.1f}%)"
            )

            if stat["mean_fitness"] is not None:
                logger.info(
                    f"  Mean fitness (EVOLVING): {stat['mean_fitness']:.3f}"
                )
                logger.info(
                    f"  Best fitness (EVOLVING): {stat['best_fitness']:.3f}"
                )
                logger.info(
                    f"  EVOLVING programs with fitness: {stat['fitness_count']} ({stat['fitness_count']/stat['evolving_count']*100:.1f}%)"
                )

        # Save detailed island statistics to file
        island_stats_file_path = output_folder / "island_statistics.txt"
        with open(island_stats_file_path, "w") as f:
            f.write("Island Statistics Analysis\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total programs with island data: {len(island_data)}\n")
            f.write(
                f"  Total EVOLVING programs with island data: {len(evolving_island_data)}\n"
            )
            f.write(
                f"  Total programs without island data: {len(full_df) - len(island_data)}\n"
            )
            f.write(
                f"  Unique islands: {len(island_data[island_col].unique())}\n\n"
            )

            f.write(f"ISLAND-BY-ISLAND ANALYSIS:\n")
            for stat in island_stats:
                f.write(f"\n{stat['island'].upper()}:\n")
                f.write(f"  Total programs: {stat['total_count']}\n")
                f.write(
                    f"  EVOLVING: {stat['evolving_count']} ({stat['evolving_count']/stat['total_count']*100:.1f}%)\n"
                )
                f.write(
                    f"  Discarded: {stat['discarded_count']} ({stat['discarded_count']/stat['total_count']*100:.1f}%)\n"
                )
                f.write(
                    f"  Other states: {stat['other_states_count']} ({stat['other_states_count']/stat['total_count']*100:.1f}%)\n"
                )

                if stat["mean_fitness"] is not None:
                    f.write(
                        f"  Mean fitness (EVOLVING): {stat['mean_fitness']:.3f}\n"
                    )
                    f.write(
                        f"  Best fitness (EVOLVING): {stat['best_fitness']:.3f}\n"
                    )
                    f.write(
                        f"  EVOLVING programs with fitness: {stat['fitness_count']} ({stat['fitness_count']/stat['evolving_count']*100:.1f}%)\n"
                    )

        logger.info(
            f"✅ Saved detailed island statistics to {island_stats_file_path}"
        )

    def plot_stage_results_statistics(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Create comprehensive analysis of individual DAG stage results."""

        if not fitness_analysis:
            logger.warning("No data to plot")
            return

        full_df = fitness_analysis["full_df"]

        # Extract stage results from programs
        stage_data = []

        # Find all stage-related columns
        stage_columns = [
            col
            for col in full_df.columns
            if col.startswith("stage_") and col.endswith("_status")
        ]
        stage_names = [
            col.replace("stage_", "").replace("_status", "")
            for col in stage_columns
        ]

        logger.info(f"🔍 Found stage columns: {stage_names}")

        for _, program_row in full_df.iterrows():
            program_id = program_row["program_id"]
            created_at = program_row["created_at"]
            time_since_start = program_row["time_since_start"]

            # Extract stage data for each stage
            for stage_name in stage_names:
                stage_key = f"stage_{stage_name}"
                status_col = f"{stage_key}_status"

                # Check if this stage has data
                if status_col in program_row and pd.notna(
                    program_row[status_col]
                ):
                    stage_data.append(
                        {
                            "program_id": program_id,
                            "stage_name": stage_name,
                            "stage_status": program_row[status_col],
                            "started_at": program_row.get(
                                f"{stage_key}_started_at"
                            ),
                            "finished_at": program_row.get(
                                f"{stage_key}_finished_at"
                            ),
                            "duration_seconds": program_row.get(
                                f"{stage_key}_duration"
                            ),
                            "has_error": program_row.get(
                                f"{stage_key}_has_error", False
                            ),
                            "error_type": program_row.get(
                                f"{stage_key}_error_type"
                            ),
                            "program_created_at": created_at,
                            "time_since_start": time_since_start,
                        }
                    )

        if not stage_data:
            logger.warning(
                "No stage results data found. This might be because:"
            )
            logger.warning("1. Programs don't have stage_results populated")
            logger.warning("2. Stage results are not being extracted properly")
            logger.warning(
                "3. Programs haven't been processed through DAG stages yet"
            )
            logger.warning("4. Stage results are stored in a different format")

            # Show what columns we actually have for debugging
            stage_related_cols = [
                col for col in full_df.columns if "stage" in col.lower()
            ]
            if stage_related_cols:
                logger.info(
                    f"Found stage-related columns: {stage_related_cols}"
                )
            else:
                logger.info("No stage-related columns found in the data")
            return

        stage_df = pd.DataFrame(stage_data)

        # Convert timestamps to datetime if they're strings
        for col in ["started_at", "finished_at", "program_created_at"]:
            if col in stage_df.columns:
                stage_df[col] = pd.to_datetime(stage_df[col], errors="coerce")

        # Clean up duration data
        if "duration_seconds" in stage_df.columns:
            # Filter out negative or extremely long durations
            stage_df["duration_seconds"] = stage_df["duration_seconds"].apply(
                lambda x: (
                    x if pd.notna(x) and 0 <= x <= 86400 else None
                )  # Max 24 hours
            )

        logger.info(
            f"📊 Found {len(stage_df)} stage results from {stage_df['program_id'].nunique()} programs"
        )
        logger.info(
            f"📊 Unique stages: {list(stage_df['stage_name'].unique())}"
        )
        logger.info(
            f"📊 Stage statuses: {list(stage_df['stage_status'].unique())}"
        )

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # Create figure with subplots - increased size for better readability with long stage names
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(
            "DAG Stage Results Analysis", fontsize=18, fontweight="bold"
        )

        # 1. Stage Status Distribution (Overall)
        ax1 = axes[0, 0]
        status_counts = stage_df["stage_status"].value_counts()
        colors = plt.cm.Set3.colors[: len(status_counts)]

        bars = ax1.bar(
            range(len(status_counts)),
            status_counts.values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_xlabel("Stage Status")
        ax1.set_ylabel("Number of Stage Results")
        ax1.set_title("Overall Stage Status Distribution")
        ax1.set_xticks(range(len(status_counts)))
        ax1.set_xticklabels(
            status_counts.index, rotation=45, ha="right", fontsize=10
        )

        # Add value labels on bars
        for bar, count in zip(bars, status_counts.values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(status_counts.values) * 0.01,
                f"{count}\n({count/len(stage_df)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Stage Status Distribution by Stage Name
        ax2 = axes[0, 1]
        stage_status_pivot = (
            pd.crosstab(
                stage_df["stage_name"],
                stage_df["stage_status"],
                normalize="index",
            )
            * 100
        )

        stage_status_pivot.plot(
            kind="bar", stacked=True, ax=ax2, colormap="Set3"
        )
        ax2.set_xlabel("Stage Name")
        ax2.set_ylabel("Percentage")
        ax2.set_title("Stage Status Distribution by Stage Name")
        ax2.tick_params(axis="x", rotation=45, labelsize=9)
        ax2.legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Stage Duration Analysis (if available)
        ax3 = axes[0, 2]
        if (
            "duration_seconds" in stage_df.columns
            and stage_df["duration_seconds"].notna().any()
        ):
            valid_duration = stage_df[stage_df["duration_seconds"].notna()]

            # Box plot of duration by stage name
            stage_duration_data = [
                valid_duration[valid_duration["stage_name"] == stage][
                    "duration_seconds"
                ].values
                for stage in valid_duration["stage_name"].unique()
            ]
            stage_names = valid_duration["stage_name"].unique()

            if stage_duration_data:
                bp = ax3.boxplot(
                    stage_duration_data, labels=stage_names, patch_artist=True
                )

                # Color the boxes
                for patch, color in zip(
                    bp["boxes"], colors[: len(stage_duration_data)]
                ):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax3.set_xlabel("Stage Name")
                ax3.set_ylabel("Duration (seconds)")
                ax3.set_title("Stage Duration Distribution")
                ax3.tick_params(axis="x", rotation=45, labelsize=9)
                ax3.grid(True, alpha=0.3, axis="y")
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No valid duration data available",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=12,
                )
                ax3.set_title("Stage Duration Distribution")
        else:
            ax3.text(
                0.5,
                0.5,
                "No duration data available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title("Stage Duration Distribution")

        # 4. Stage Timeline (when stages were executed)
        ax4 = axes[1, 0]
        if (
            "started_at" in stage_df.columns
            and stage_df["started_at"].notna().any()
        ):
            valid_timeline = stage_df[stage_df["started_at"].notna()]

            for stage_name in valid_timeline["stage_name"].unique():
                stage_data = valid_timeline[
                    valid_timeline["stage_name"] == stage_name
                ]
                ax4.scatter(
                    stage_data["time_since_start"],
                    [stage_name] * len(stage_data),
                    alpha=0.6,
                    s=20,
                    label=f"{stage_name} ({len(stage_data)} executions)",
                )

            ax4.set_xlabel("Time Since Start (seconds)")
            ax4.set_ylabel("Stage Name")
            ax4.set_title("Stage Execution Timeline")
            ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax4.grid(True, alpha=0.3, axis="x")
        else:
            ax4.text(
                0.5,
                0.5,
                "No timeline data available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title("Stage Execution Timeline")

        # 5. Error Rate by Stage
        ax5 = axes[1, 1]
        if "has_error" in stage_df.columns:
            error_rates = (
                stage_df.groupby("stage_name")["has_error"]
                .agg(["count", "sum"])
                .reset_index()
            )
            error_rates["error_rate"] = (
                error_rates["sum"] / error_rates["count"] * 100
            )

            bars = ax5.bar(
                range(len(error_rates)),
                error_rates["error_rate"],
                color="red",
                alpha=0.7,
                edgecolor="black",
            )
            ax5.set_xlabel("Stage Name")
            ax5.set_ylabel("Error Rate (%)")
            ax5.set_title("Error Rate by Stage")
            ax5.set_xticks(range(len(error_rates)))
            ax5.set_xticklabels(
                error_rates["stage_name"], rotation=45, ha="right", fontsize=9
            )

            # Add value labels on bars
            for bar, rate, count in zip(
                bars, error_rates["error_rate"], error_rates["count"]
            ):
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{rate:.1f}%\n(n={count})",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax5.grid(True, alpha=0.3, axis="y")
        else:
            ax5.text(
                0.5,
                0.5,
                "No error data available",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=12,
            )
            ax5.set_title("Error Rate by Stage")

        # 6. Success Rate by Stage
        ax6 = axes[1, 2]
        success_rates = (
            stage_df.groupby("stage_name")["stage_status"]
            .apply(lambda x: (x == "completed").sum() / len(x) * 100)
            .reset_index()
        )
        success_rates.columns = ["stage_name", "success_rate"]

        bars = ax6.bar(
            range(len(success_rates)),
            success_rates["success_rate"],
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        ax6.set_xlabel("Stage Name")
        ax6.set_ylabel("Success Rate (%)")
        ax6.set_title("Success Rate by Stage")
        ax6.set_xticks(range(len(success_rates)))
        ax6.set_xticklabels(
            success_rates["stage_name"], rotation=45, ha="right", fontsize=9
        )

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates["success_rate"]):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax6.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(pad=3.0)  # Increased padding for better spacing

        if save_plots:
            self._save_fig(fig, output_folder / "stage_results_statistics")

        plt.show()

        # Additional detailed stage analysis
        logger.info("\n📊 STAGE RESULTS ANALYSIS:")
        logger.info("=" * 80)

        # Overall statistics
        logger.info(f"\n📈 OVERALL STATISTICS:")
        logger.info(f"  Total stage results: {len(stage_df)}")
        logger.info(f"  Unique programs: {stage_df['program_id'].nunique()}")
        logger.info(f"  Unique stages: {stage_df['stage_name'].nunique()}")
        logger.info(f"  Unique statuses: {stage_df['stage_status'].nunique()}")

        # Stage-by-stage analysis
        logger.info(f"\n🔍 STAGE-BY-STAGE ANALYSIS:")
        for stage_name in stage_df["stage_name"].unique():
            stage_data = stage_df[stage_df["stage_name"] == stage_name]
            total_executions = len(stage_data)

            logger.info(f"\n{stage_name.upper()}:")
            logger.info(f"  Total executions: {total_executions}")

            # Status breakdown
            status_counts = stage_data["stage_status"].value_counts()
            for status, count in status_counts.items():
                percentage = count / total_executions * 100
                logger.info(f"  {status}: {count} ({percentage:.1f}%)")

            # Duration analysis if available
            if (
                "duration_seconds" in stage_data.columns
                and stage_data["duration_seconds"].notna().any()
            ):
                valid_duration = stage_data[
                    stage_data["duration_seconds"].notna()
                ]["duration_seconds"]
                if not valid_duration.empty:
                    logger.info(
                        f"  Duration: {valid_duration.mean():.1f}s mean, {valid_duration.median():.1f}s median"
                    )
                    logger.info(
                        f"  Duration range: {valid_duration.min():.1f}s - {valid_duration.max():.1f}s"
                    )

            # Error analysis if available
            if "has_error" in stage_data.columns:
                error_count = stage_data["has_error"].sum()
                error_rate = error_count / total_executions * 100
                logger.info(f"  Errors: {error_count} ({error_rate:.1f}%)")

        # Save detailed stage statistics to file
        stage_stats_file_path = output_folder / "stage_results_statistics.txt"
        with open(stage_stats_file_path, "w") as f:
            f.write("DAG Stage Results Statistics\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total stage results: {len(stage_df)}\n")
            f.write(f"  Unique programs: {stage_df['program_id'].nunique()}\n")
            f.write(f"  Unique stages: {stage_df['stage_name'].nunique()}\n")
            f.write(
                f"  Unique statuses: {stage_df['stage_status'].nunique()}\n\n"
            )

            f.write(f"STAGE-BY-STAGE ANALYSIS:\n")
            for stage_name in stage_df["stage_name"].unique():
                stage_data = stage_df[stage_df["stage_name"] == stage_name]
                total_executions = len(stage_data)

                f.write(f"\n{stage_name.upper()}:\n")
                f.write(f"  Total executions: {total_executions}\n")

                # Status breakdown
                status_counts = stage_data["stage_status"].value_counts()
                for status, count in status_counts.items():
                    percentage = count / total_executions * 100
                    f.write(f"  {status}: {count} ({percentage:.1f}%)\n")

                # Duration analysis if available
                if (
                    "duration_seconds" in stage_data.columns
                    and stage_data["duration_seconds"].notna().any()
                ):
                    valid_duration = stage_data[
                        stage_data["duration_seconds"].notna()
                    ]["duration_seconds"]
                    if not valid_duration.empty:
                        f.write(
                            f"  Duration: {valid_duration.mean():.1f}s mean, {valid_duration.median():.1f}s median\n"
                        )
                        f.write(
                            f"  Duration range: {valid_duration.min():.1f}s - {valid_duration.max():.1f}s\n"
                        )

                # Error analysis if available
                if "has_error" in stage_data.columns:
                    error_count = stage_data["has_error"].sum()
                    error_rate = error_count / total_executions * 100
                    f.write(f"  Errors: {error_count} ({error_rate:.1f}%)\n")

        logger.info(
            f"✅ Saved detailed stage statistics to {stage_stats_file_path}"
        )

    def analyze_top_programs(
        self,
        df: pd.DataFrame,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        top_n: int = 10,
    ):
        """Analyze the top performing programs and save to file."""

        if not fitness_analysis:
            return

        fitness_col = fitness_analysis["fitness_col"]
        timeline_df = fitness_analysis["timeline_df"]

        # Get top programs
        top_programs = timeline_df.nlargest(top_n, fitness_col)

        # Create detailed analysis file
        analysis_path = output_folder / "top_programs_analysis.txt"

        with open(analysis_path, "w") as f:
            f.write(f"🏆 TOP {top_n} PROGRAMS ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            for i, (_, program) in enumerate(top_programs.iterrows(), 1):
                f.write(f"{i}. Program ID: {program['program_id'][:12]}...\n")
                f.write(f"   Fitness: {program[fitness_col]:.4f}\n")
                f.write(f"   Created: {program['created_at']}\n")
                f.write(f"   Generation: {program['generation']}\n")
                f.write(f"   State: {program['state']}\n")

                # Show lineage info if available
                if program["lineage_parents"] > 0:
                    f.write(f"   Parents: {program['lineage_parents']}\n")
                    if program["lineage_mutation"]:
                        f.write(f"   Mutation: {program['lineage_mutation']}\n")

                # Show other metrics if available
                metric_cols = [
                    col
                    for col in program.index
                    if col.startswith("metric_") and col != fitness_col
                ]
                if metric_cols:
                    f.write(
                        f"   Other metrics: {', '.join([f'{col[7:]}: {program[col]:.3f}' for col in metric_cols[:3]])}\n"
                    )

                f.write("\n")

            # Show fitness improvement timeline
            f.write(f"\n📈 FITNESS IMPROVEMENT TIMELINE:\n")
            f.write("=" * 80 + "\n")

            improvements = timeline_df[
                timeline_df[fitness_col] == timeline_df["running_best_fitness"]
            ]

            for i, (_, improvement) in enumerate(improvements.iterrows(), 1):
                time_since_start = improvement["time_since_start"]
                fitness = improvement[fitness_col]
                program_id = improvement["program_id"][:12]

                f.write(
                    f"{i:2d}. Time: {time_since_start:8.1f}s | Fitness: {fitness:8.4f} | Program: {program_id}...\n"
                )

        logger.info(f"✅ Saved top programs analysis to {analysis_path}")

        # Also print to console for immediate feedback
        logger.info(f"🏆 TOP {top_n} PROGRAMS:")
        logger.info("=" * 80)

        for i, (_, program) in enumerate(top_programs.iterrows(), 1):
            logger.info(f"\n{i}. Program ID: {program['program_id'][:12]}...")
            logger.info(f"   Fitness: {program[fitness_col]:.4f}")
            logger.info(f"   Created: {program['created_at']}")
            logger.info(f"   Generation: {program['generation']}")
            logger.info(f"   State: {program['state']}")

            # Show lineage info if available
            if program["lineage_parents"] > 0:
                logger.info(f"   Parents: {program['lineage_parents']}")
                if program["lineage_mutation"]:
                    logger.info(f"   Mutation: {program['lineage_mutation']}")

            # Show other metrics if available
            metric_cols = [
                col
                for col in program.index
                if col.startswith("metric_") and col != fitness_col
            ]
            if metric_cols:
                logger.info(
                    f"   Other metrics: {', '.join([f'{col[7:]}: {program[col]:.3f}' for col in metric_cols[:3]])}"
                )

        # Show fitness improvement timeline
        logger.info(f"\n📈 FITNESS IMPROVEMENT TIMELINE:")
        logger.info("=" * 80)

        improvements = timeline_df[
            timeline_df[fitness_col] == timeline_df["running_best_fitness"]
        ]

        for i, (_, improvement) in enumerate(improvements.iterrows(), 1):
            time_since_start = improvement["time_since_start"]
            fitness = improvement[fitness_col]
            program_id = improvement["program_id"][:12]

            logger.info(
                f"{i:2d}. Time: {time_since_start:8.1f}s | Fitness: {fitness:8.4f} | Program: {program_id}..."
            )

    def export_evolution_data(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        base_filename: str = "evolution_data",
    ):
        """Export the evolution data to CSV and JSON for further analysis."""

        if not fitness_analysis:
            logger.warning("No data to export")
            return

        timeline_df = fitness_analysis["timeline_df"]

        # Export timeline data
        timeline_path = output_folder / f"{base_filename}.csv"
        timeline_df.to_csv(timeline_path, index=False)
        logger.info(f"✅ Exported timeline data to {timeline_path}")

        # Export generation stats if available
        if fitness_analysis["generation_stats"] is not None:
            gen_path = output_folder / f"{base_filename}_generations.csv"
            fitness_analysis["generation_stats"].to_csv(gen_path, index=False)
            logger.info(f"✅ Exported generation stats to {gen_path}")

        # Create summary report
        summary = {
            "total_programs_with_fitness": fitness_analysis["total_programs"],
            "total_all_programs": fitness_analysis["total_all_programs"],
            "best_fitness": fitness_analysis["best_fitness"],
            "worst_fitness": fitness_analysis["worst_fitness"],
            "mean_fitness": fitness_analysis["mean_fitness"],
            "start_time": fitness_analysis["start_time"],
            "analysis_timestamp": datetime.now(timezone.utc),
        }

        summary_path = output_folder / f"{base_filename}_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Evolution Analysis Summary\n")
            f.write("=" * 50 + "\n")
            f.write(
                f"Total Programs with Fitness: {summary['total_programs_with_fitness']}\n"
            )
            f.write(f"Total All Programs: {summary['total_all_programs']}\n")
            f.write(
                f"Success Rate: {summary['total_programs_with_fitness']/summary['total_all_programs']*100:.1f}%\n"
            )
            f.write(f"Best Fitness: {summary['best_fitness']:.4f}\n")
            f.write(f"Worst Fitness: {summary['worst_fitness']:.4f}\n")
            f.write(f"Mean Fitness: {summary['mean_fitness']:.4f}\n")
            f.write(f"Start Time: {summary['start_time']}\n")
            f.write(f"Analysis Time: {summary['analysis_timestamp']}\n")

        logger.info(f"✅ Exported summary to {summary_path}")

        # Export JSON data for easy comparison between runs
        self.export_json_data(fitness_analysis, output_folder, base_filename)

    def export_json_data(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        base_filename: str = "evolution_data",
    ):
        """Export comprehensive evolution data to JSON for easy comparison between runs."""

        if not fitness_analysis:
            logger.warning("No data to export to JSON")
            return

        import json
        from datetime import datetime, timezone

        # Prepare comprehensive JSON data structure
        json_data = {
            "metadata": {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "redis_host": self.redis_host,
                "redis_port": self.redis_port,
                "redis_db": self.redis_db,
                "redis_prefix": self.redis_prefix,
                "extreme_threshold": self.extreme_threshold,
                "outlier_multiplier": self.outlier_multiplier,
                "outlier_removal_enabled": self.remove_outliers,
                "fitness_column": fitness_analysis.get("fitness_col", "metric_fitness"),
                "start_time": fitness_analysis["start_time"].isoformat() if fitness_analysis.get("start_time") else None,
            },
            "summary_statistics": {
                "total_programs_with_fitness": fitness_analysis["total_programs"],
                "total_all_programs": fitness_analysis["total_all_programs"],
                "best_fitness": fitness_analysis["best_fitness"],
                "worst_fitness": fitness_analysis["worst_fitness"],
                "mean_fitness": fitness_analysis["mean_fitness"],
                "success_rate_percent": (
                    fitness_analysis["total_programs"] / fitness_analysis["total_all_programs"] * 100
                    if fitness_analysis["total_all_programs"] > 0 else 0
                ),
            },
            "running_statistics": None,
            "generation_statistics": None,
            "timeline_summary": None,
        }

        # Add running statistics if available
        if "running_statistics" in fitness_analysis:
            running_stats = fitness_analysis["running_statistics"]
            timeline_data = running_stats["timeline_data"]
            
            # Calculate summary statistics from running data
            final_idx = len(timeline_data) - 1
            if final_idx >= 0:
                final_running_mean = timeline_data[final_idx].get("running_mean_fitness")
                final_running_std = timeline_data[final_idx].get("running_std_fitness")
                final_running_best = timeline_data[final_idx].get("running_best_fitness")
                
                # Calculate improvement metrics
                first_valid_mean_idx = 0
                for i, data_point in enumerate(timeline_data):
                    if data_point.get("running_mean_fitness") is not None:
                        first_valid_mean_idx = i
                        break
                
                initial_running_mean = timeline_data[first_valid_mean_idx].get("running_mean_fitness", 0)
                mean_improvement = final_running_mean - initial_running_mean if final_running_mean and initial_running_mean else 0
                
                # Calculate time span
                total_time = timeline_data[final_idx].get("time_since_start", 0) - timeline_data[0].get("time_since_start", 0)
                
                json_data["running_statistics"] = {
                    "rolling_window": running_stats["rolling_window"],
                    "summary": {
                        "final_running_mean": final_running_mean,
                        "final_running_std": final_running_std,
                        "final_running_best": final_running_best,
                        "initial_running_mean": initial_running_mean,
                        "mean_improvement": mean_improvement,
                        "total_time_seconds": total_time,
                        "mean_improvement_rate_per_second": mean_improvement / total_time if total_time > 0 else 0,
                        "programs_analyzed": len(timeline_data),
                    },
                    "timeline_data": timeline_data,  # Full timeline for detailed comparison
                }
            else:
                json_data["running_statistics"] = {
                    "rolling_window": running_stats["rolling_window"],
                    "summary": {},
                    "timeline_data": timeline_data,
                }

        # Add generation statistics if available
        if fitness_analysis.get("generation_stats") is not None:
            gen_stats = fitness_analysis["generation_stats"]
            json_data["generation_statistics"] = {
                "summary": {
                    "total_generations": len(gen_stats),
                    "max_generation": gen_stats["generation"].max() if not gen_stats.empty else 0,
                    "programs_per_generation_mean": gen_stats["program_count"].mean() if not gen_stats.empty else 0,
                    "best_fitness_by_generation": gen_stats["max_fitness"].max() if not gen_stats.empty else None,
                    "mean_fitness_trend": {
                        "first_generation": gen_stats["mean_fitness"].iloc[0] if not gen_stats.empty else None,
                        "last_generation": gen_stats["mean_fitness"].iloc[-1] if not gen_stats.empty else None,
                        "improvement": (
                            gen_stats["mean_fitness"].iloc[-1] - gen_stats["mean_fitness"].iloc[0]
                            if len(gen_stats) > 1 else 0
                        ),
                    },
                },
                "generation_data": gen_stats.to_dict('records') if not gen_stats.empty else [],
            }

        # Add timeline summary (key milestones)
        timeline_df = fitness_analysis["timeline_df"]
        if not timeline_df.empty:
            # Find fitness improvements (new best fitness achieved)
            improvements = timeline_df[
                timeline_df[fitness_analysis["fitness_col"]] == timeline_df.get("running_best_fitness", timeline_df[fitness_analysis["fitness_col"]].expanding().max())
            ].copy()
            
            milestones = []
            for _, improvement in improvements.iterrows():
                milestones.append({
                    "time_since_start": improvement["time_since_start"],
                    "fitness": improvement[fitness_analysis["fitness_col"]],
                    "program_id": improvement.get("program_id", "unknown"),
                    "generation": improvement.get("generation", None),
                    "created_at": improvement.get("created_at").isoformat() if improvement.get("created_at") else None,
                })

            json_data["timeline_summary"] = {
                "total_time_span_seconds": timeline_df["time_since_start"].max() - timeline_df["time_since_start"].min(),
                "fitness_improvements_count": len(improvements),
                "milestones": milestones,
            }

        # Export main JSON file
        json_path = output_folder / f"{base_filename}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"✅ Exported comprehensive JSON data to {json_path}")

        # Also export a compact version for quick comparison
        compact_data = {
            "metadata": json_data["metadata"],
            "summary": json_data["summary_statistics"],
        }
        
        # Add running statistics summary only
        if json_data["running_statistics"]:
            compact_data["running_summary"] = json_data["running_statistics"]["summary"]
        
        # Add generation summary only
        if json_data["generation_statistics"]:
            compact_data["generation_summary"] = json_data["generation_statistics"]["summary"]

        # Add key milestones only (first 10)
        if json_data["timeline_summary"]:
            compact_data["timeline_summary"] = json_data["timeline_summary"].copy()
            if len(compact_data["timeline_summary"]["milestones"]) > 10:
                compact_data["timeline_summary"]["milestones"] = compact_data["timeline_summary"]["milestones"][:10]
                compact_data["timeline_summary"]["milestones_truncated"] = True

        compact_path = output_folder / f"{base_filename}_compact.json"
        with open(compact_path, "w") as f:
            json.dump(compact_data, f, indent=2, default=str)
        logger.info(f"✅ Exported compact JSON summary to {compact_path}")

        # Log JSON export summary
        logger.info(f"\n📊 JSON EXPORT SUMMARY:")
        logger.info("=" * 50)
        logger.info(f"  Main JSON file: {json_path.name}")
        logger.info(f"  Compact JSON file: {compact_path.name}")
        
        if json_data["running_statistics"]:
            rs = json_data["running_statistics"]["summary"]
            logger.info(f"  Running statistics window: {json_data['running_statistics']['rolling_window']} programs")
            logger.info(f"  Final running mean: {rs.get('final_running_mean', 'N/A')}")
            logger.info(f"  Mean improvement: {rs.get('mean_improvement', 'N/A')}")
        
        if json_data["generation_statistics"]:
            gs = json_data["generation_statistics"]["summary"]
            logger.info(f"  Generations analyzed: {gs.get('total_generations', 'N/A')}")
        
        if json_data["timeline_summary"]:
            ts = json_data["timeline_summary"]
            logger.info(f"  Fitness improvements: {ts.get('fitness_improvements_count', 'N/A')}")
            logger.info(f"  Total time span: {ts.get('total_time_span_seconds', 'N/A'):.1f}s")

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

    # ------------------------------------------------------------------
    # 🔧  INTERNAL UTILITIES  🔧
    # ------------------------------------------------------------------

    def _configure_plotting_style(self):
        """Set global Matplotlib / Seaborn parameters for a polished look."""

        # Use seaborn's modern theme as a base
        sns.set_theme(style="whitegrid", context="talk", palette="deep")

        # Global rcParams tweaks – fonts & figure size / DPI
        plt.rcParams.update(
            {
                # Typography
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                "font.size": 14,
                "axes.titlesize": 18,
                "axes.titleweight": "bold",
                "axes.labelsize": 14,
                "axes.labelweight": "semibold",
                "legend.fontsize": 12,
                # Grid & ticks
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                # High-resolution outputs by default
                "savefig.dpi": 300,
                "figure.dpi": 150,
            }
        )

    # Small helper to save figures both as PNG & PDF for crisp presentations
    def _save_fig(self, fig: plt.Figure, path_no_ext: Path):
        png_path = path_no_ext.with_suffix(".png")
        pdf_path = path_no_ext.with_suffix(".pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        logger.info(f"✅ Saved figure to {png_path.name} & {pdf_path.name}")

    def plot_validity_distribution(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Create comprehensive validity distribution analysis."""

        if not fitness_analysis:
            logger.warning("No data to plot validity distribution")
            return

        full_df = fitness_analysis["full_df"]
        fitness_col = fitness_analysis.get("fitness_col", "metric_fitness")

        if fitness_col not in full_df.columns:
            logger.warning(
                f"No fitness column ({fitness_col}) found for validity analysis"
            )
            return

        # Define validity criteria for each program
        validity_analysis = full_df.copy()

        # 1. Basic validity: has fitness value and not -1000.0 (failure indicator)
        validity_analysis["has_fitness"] = validity_analysis[
            fitness_col
        ].notna()
        validity_analysis["not_failure"] = (
            validity_analysis[fitness_col] != -1000.0
        )
        validity_analysis["basic_valid"] = (
            validity_analysis["has_fitness"] & validity_analysis["not_failure"]
        )

        # 2. Outlier status (if outlier removal is enabled)
        if self.remove_outliers:
            fitness_values = validity_analysis[fitness_col]

            # Apply same outlier detection as in main analysis
            extreme_outliers = fitness_values < self.extreme_threshold

            # IQR method for statistical outliers
            non_extreme = fitness_values[
                fitness_values >= self.extreme_threshold
            ]
            if len(non_extreme) > 0:
                Q1 = non_extreme.quantile(0.25)
                Q3 = non_extreme.quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:
                    lower_bound = Q1 - self.outlier_multiplier * IQR
                    upper_bound = Q3 + self.outlier_multiplier * IQR
                    statistical_outliers = (fitness_values < lower_bound) | (
                        fitness_values > upper_bound
                    )
                else:
                    statistical_outliers = pd.Series(
                        [False] * len(fitness_values),
                        index=fitness_values.index,
                    )
            else:
                statistical_outliers = pd.Series(
                    [False] * len(fitness_values), index=fitness_values.index
                )

            all_outliers = extreme_outliers | statistical_outliers
            validity_analysis["is_outlier"] = all_outliers
            validity_analysis["analysis_valid"] = (
                validity_analysis["basic_valid"]
                & ~validity_analysis["is_outlier"]
            )
        else:
            validity_analysis["is_outlier"] = False
            validity_analysis["analysis_valid"] = validity_analysis[
                "basic_valid"
            ]

        # 3. State-based validity
        validity_analysis["is_completed"] = (
            validity_analysis["state"] == "completed"
        )
        validity_analysis["is_running"] = validity_analysis["state"].isin(
            ["running", "evolving"]
        )
        validity_analysis["is_failed"] = validity_analysis["state"].isin(
            ["failed", "error"]
        )

        # Create comprehensive validity categories
        def categorize_validity(row):
            if not row["has_fitness"]:
                return "No Fitness Data"
            elif row["not_failure"] == False:  # fitness == -1000.0
                return "Marked as Failure"
            elif row.get("is_outlier", False):
                return "Outlier (Removed)"
            elif row["analysis_valid"]:
                return "Valid for Analysis"
            else:
                return "Other Invalid"

        validity_analysis["validity_category"] = validity_analysis.apply(
            categorize_validity, axis=1
        )

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(
            "Program Validity Distribution Analysis",
            fontsize=18,
            fontweight="bold",
        )

        # 1. Overall Validity Distribution (Pie Chart)
        ax1 = axes[0, 0]
        validity_counts = validity_analysis["validity_category"].value_counts()
        colors = plt.cm.Set3.colors[: len(validity_counts)]

        ax1.pie(
            validity_counts.values,
            labels=validity_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=[0.05] * len(validity_counts),
        )
        ax1.set_title("Program Validity Distribution")

        # 2. Validity by State (Stacked Bar)
        ax2 = axes[0, 1]
        state_validity_pivot = (
            pd.crosstab(
                validity_analysis["state"],
                validity_analysis["validity_category"],
                normalize="index",
            )
            * 100
        )

        state_validity_pivot.plot(
            kind="bar", stacked=True, ax=ax2, colormap="Set3"
        )
        ax2.set_xlabel("Program State")
        ax2.set_ylabel("Percentage")
        ax2.set_title("Validity Distribution by Program State")
        ax2.tick_params(axis="x", rotation=45, labelsize=10)
        ax2.legend(title="Validity", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Validity Timeline
        ax3 = axes[0, 2]

        # Create consistent y-axis ordering for validity categories
        categories_sorted = sorted(
            validity_analysis["validity_category"].unique()
        )
        category_to_y = {
            name: idx for idx, name in enumerate(categories_sorted)
        }

        for category in categories_sorted:
            category_programs = validity_analysis[
                validity_analysis["validity_category"] == category
            ]
            y = np.full(len(category_programs), category_to_y[category])
            ax3.scatter(
                category_programs["time_since_start"],
                y,
                alpha=0.6,
                s=20,
                label=f"{category} ({len(category_programs)})",
            )

        ax3.set_xlabel("Time Since Start (seconds)")
        ax3.set_ylabel("Validity Category")
        ax3.set_title("Program Validity Timeline")
        ax3.set_yticks(list(category_to_y.values()))
        ax3.set_yticklabels(categories_sorted, fontsize=9)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.grid(True, alpha=0.3, axis="x")

        # 4. Validity Counts (Bar Chart)
        ax4 = axes[1, 0]
        bars = ax4.bar(
            range(len(validity_counts)),
            validity_counts.values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax4.set_xlabel("Validity Category")
        ax4.set_ylabel("Number of Programs")
        ax4.set_title("Program Validity Counts")
        ax4.set_xticks(range(len(validity_counts)))
        ax4.set_xticklabels(
            validity_counts.index, rotation=45, ha="right", fontsize=10
        )

        # Add value labels on bars
        for bar, count in zip(bars, validity_counts.values):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(validity_counts.values) * 0.01,
                f"{count}\n({count/len(validity_analysis)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax4.grid(True, alpha=0.3, axis="y")

        # 5. Success Rate Analysis
        ax5 = axes[1, 1]

        # Calculate success rates by different criteria
        success_metrics = {
            "Has Any Fitness": (
                validity_analysis["has_fitness"].sum()
                / len(validity_analysis)
                * 100
            ),
            "Not Failure (-1000)": (
                validity_analysis["not_failure"].sum()
                / len(validity_analysis)
                * 100
            ),
            "Basic Valid": (
                validity_analysis["basic_valid"].sum()
                / len(validity_analysis)
                * 100
            ),
            "Analysis Valid": (
                validity_analysis["analysis_valid"].sum()
                / len(validity_analysis)
                * 100
            ),
            "Completed State": (
                validity_analysis["is_completed"].sum()
                / len(validity_analysis)
                * 100
            ),
        }

        success_names = list(success_metrics.keys())
        success_rates = list(success_metrics.values())

        bars = ax5.barh(
            range(len(success_names)),
            success_rates,
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        ax5.set_xlabel("Success Rate (%)")
        ax5.set_ylabel("Success Metric")
        ax5.set_title("Program Success Rates")
        ax5.set_yticks(range(len(success_names)))
        ax5.set_yticklabels(success_names)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            width = bar.get_width()
            ax5.text(
                width + 1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{rate:.1f}%",
                ha="left",
                va="center",
                fontweight="bold",
            )

        ax5.grid(True, alpha=0.3, axis="x")
        ax5.set_xlim(0, 105)

        # 6. Detailed Statistics Table
        ax6 = axes[1, 2]

        # Create detailed statistics
        total_programs = len(validity_analysis)
        stats_data = []

        for category in validity_counts.index:
            count = validity_counts[category]
            percentage = count / total_programs * 100
            stats_data.append([category, count, f"{percentage:.1f}%"])

        # Add summary row
        valid_for_analysis = validity_analysis["analysis_valid"].sum()
        stats_data.append(["─" * 20, "─" * 10, "─" * 10])
        stats_data.append(["TOTAL PROGRAMS", total_programs, "100.0%"])
        stats_data.append(
            [
                "VALID FOR ANALYSIS",
                valid_for_analysis,
                f"{valid_for_analysis/total_programs*100:.1f}%",
            ]
        )

        # Create table
        table_data = [["Validity Category", "Count", "Percentage"]] + stats_data
        table = ax6.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc="center",
            loc="center",
            colWidths=[0.5, 0.25, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        ax6.set_title("Validity Statistics Summary")
        ax6.axis("off")

        plt.tight_layout(pad=3.0)

        if save_plots:
            self._save_fig(fig, output_folder / "validity_distribution")

        plt.show()

        # Log detailed validity analysis
        logger.info("\n✅ PROGRAM VALIDITY ANALYSIS:")
        logger.info("=" * 80)

        total = len(validity_analysis)
        logger.info(f"📊 OVERALL STATISTICS:")
        logger.info(f"  Total programs: {total}")

        for category, count in validity_counts.items():
            percentage = count / total * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")

        logger.info(f"\n📈 SUCCESS RATES:")
        for metric, rate in success_metrics.items():
            logger.info(f"  {metric}: {rate:.1f}%")

        # Show validity by state
        logger.info(f"\n🔍 VALIDITY BY STATE:")
        for state in validity_analysis["state"].unique():
            state_data = validity_analysis[validity_analysis["state"] == state]
            valid_count = state_data["analysis_valid"].sum()
            total_count = len(state_data)
            logger.info(
                f"  {state}: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)"
            )

        # Save detailed validity statistics to file
        validity_stats_path = output_folder / "validity_statistics.txt"
        with open(validity_stats_path, "w") as f:
            f.write("Program Validity Distribution Analysis\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total programs: {total}\n\n")

            for category, count in validity_counts.items():
                percentage = count / total * 100
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")

            f.write(f"\nSUCCESS RATES:\n")
            for metric, rate in success_metrics.items():
                f.write(f"  {metric}: {rate:.1f}%\n")

            f.write(f"\nVALIDITY BY STATE:\n")
            for state in validity_analysis["state"].unique():
                state_data = validity_analysis[
                    validity_analysis["state"] == state
                ]
                valid_count = state_data["analysis_valid"].sum()
                total_count = len(state_data)
                f.write(
                    f"  {state}: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)\n"
                )

        logger.info(
            f"✅ Saved detailed validity statistics to {validity_stats_path}"
        )

    # ------------------------------------------------------------------
    # 📊 Metric Correlation Heatmap
    # ------------------------------------------------------------------
    def plot_metric_correlations(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Plot correlation heat-map between all numeric metric_ columns."""

        if not fitness_analysis:
            return

        df = fitness_analysis["full_df"]
        metric_cols = [c for c in df.columns if c.startswith("metric_")]
        # Keep fitness in the analysis – we will use a centred diverging palette so ±1 correlations remain readable

        numeric_cols = (
            metric_cols + ["lineage_generation", "generation"]
            if "generation" in df.columns
            else metric_cols
        )
        if len(numeric_cols) < 2:
            logger.info("Not enough metrics for correlation heatmap")
            return

        # Filter outliers from metric data to avoid correlation distortion (if enabled)
        df_filtered = df.copy()
        outliers_removed = 0

        if self.remove_outliers:
            for col in metric_cols:
                if (
                    col in df_filtered.columns
                    and df_filtered[col].notna().sum() > 0
                ):
                    values = df_filtered[col].dropna()

                    # Remove extreme outliers using IQR method
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:  # Only apply if there's variation
                        lower_bound = Q1 - self.outlier_multiplier * IQR
                        upper_bound = Q3 + self.outlier_multiplier * IQR

                        # For fitness column, also apply extreme threshold
                        if col == "metric_fitness":
                            lower_bound = max(
                                lower_bound, self.extreme_threshold
                            )

                        outliers = (df_filtered[col] < lower_bound) | (
                            df_filtered[col] > upper_bound
                        )
                        outliers_in_col = outliers.sum()

                        if outliers_in_col > 0:
                            df_filtered.loc[outliers, col] = np.nan
                            outliers_removed += outliers_in_col
                            logger.info(
                                f"   Removed {outliers_in_col} outliers from {col} (range: {lower_bound:.2f} to {upper_bound:.2f})"
                            )

            if outliers_removed > 0:
                logger.info(
                    f"🔍 Removed {outliers_removed} total outliers from correlation analysis"
                )
        else:
            logger.info("⚠️ Outlier removal disabled for correlation analysis")

        corr = df_filtered[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,  # make 0 white so positive/negative stand out equally
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Metric Correlation Heat-map")

        plt.tight_layout()
        if save_plots:
            self._save_fig(fig, output_folder / "metric_correlations")
        plt.show()

    def plot_metric_distributions(
        self,
        fitness_analysis: Dict[str, Any],
        output_folder: Path,
        save_plots: bool = True,
    ):
        """Create comprehensive distribution plots for all metrics across all programs."""

        if not fitness_analysis:
            logger.warning("No data to plot metric distributions")
            return

        full_df = fitness_analysis["full_df"]

        # Find all metric columns
        metric_cols = [
            col for col in full_df.columns if col.startswith("metric_")
        ]

        if not metric_cols:
            logger.warning("No metric columns found for distribution analysis")
            return

        logger.info(
            f"📊 Found {len(metric_cols)} metrics for distribution analysis: {metric_cols}"
        )

        # Create metrics folder
        metrics_folder = output_folder / "metrics"
        metrics_folder.mkdir(exist_ok=True)
        logger.info(f"📁 Created metrics folder: {metrics_folder}")

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")

        # Process each metric
        for metric_col in metric_cols:
            metric_name = metric_col.replace("metric_", "")
            logger.info(f"📈 Processing metric: {metric_name}")

            # Get valid values for this metric
            metric_data = full_df[metric_col].dropna()

            if metric_data.empty:
                logger.warning(f"⚠️ No valid data for metric {metric_name}")
                continue

            # Apply outlier removal if enabled
            if self.remove_outliers:
                # Remove extreme outliers that disrupt plotting
                values = metric_data.copy()

                # For fitness metric, apply extreme threshold
                if metric_col == "metric_fitness":
                    extreme_outliers = values < self.extreme_threshold
                    non_extreme = values[values >= self.extreme_threshold]
                else:
                    # For other metrics, use a more general approach
                    extreme_outliers = pd.Series(
                        [False] * len(values), index=values.index
                    )
                    non_extreme = values

                # Use IQR method for additional outlier detection
                if len(non_extreme) > 0:
                    Q1 = non_extreme.quantile(0.25)
                    Q3 = non_extreme.quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:
                        lower_bound = Q1 - self.outlier_multiplier * IQR
                        upper_bound = Q3 + self.outlier_multiplier * IQR
                        statistical_outliers = (values < lower_bound) | (
                            values > upper_bound
                        )
                    else:
                        statistical_outliers = pd.Series(
                            [False] * len(values), index=values.index
                        )
                else:
                    statistical_outliers = pd.Series(
                        [False] * len(values), index=values.index
                    )

                # Combine outlier detection methods
                all_outliers = extreme_outliers | statistical_outliers

                # Log outlier detection results
                num_extreme = extreme_outliers.sum()
                num_statistical = statistical_outliers.sum()
                num_total_outliers = all_outliers.sum()

                logger.info(f"   🔍 {metric_name} outlier detection:")
                logger.info(f"      Extreme outliers: {num_extreme}")
                logger.info(f"      Statistical outliers: {num_statistical}")
                logger.info(
                    f"      Total outliers removed: {num_total_outliers}"
                )

                # Filter out outliers
                metric_data = values[~all_outliers]

                if metric_data.empty:
                    logger.warning(
                        f"⚠️ No valid data for metric {metric_name} after outlier removal"
                    )
                    continue

            # Create comprehensive distribution plot for this metric
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(
                f"{metric_name.upper()} Distribution Analysis",
                fontsize=16,
                fontweight="bold",
            )

            # 1. Histogram
            ax1 = axes[0, 0]
            ax1.hist(
                metric_data,
                bins=30,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax1.axvline(
                metric_data.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {metric_data.mean():.3f}",
            )
            ax1.axvline(
                metric_data.median(),
                color="green",
                linestyle="--",
                label=f"Median: {metric_data.median():.3f}",
            )
            ax1.set_xlabel(metric_name)
            ax1.set_ylabel("Frequency")
            ax1.set_title(f"{metric_name} Distribution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Box Plot
            ax2 = axes[0, 1]
            ax2.boxplot(
                metric_data,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.7),
            )
            ax2.set_ylabel(metric_name)
            ax2.set_title(f"{metric_name} Box Plot")
            ax2.grid(True, alpha=0.3)

            # 3. Violin Plot
            ax3 = axes[1, 0]
            ax3.violinplot(metric_data, showmeans=True)
            ax3.set_ylabel(metric_name)
            ax3.set_title(f"{metric_name} Violin Plot")
            ax3.grid(True, alpha=0.3)

            # 4. Statistics Table
            ax4 = axes[1, 1]

            # Calculate comprehensive statistics
            stats_data = [
                ["Count", f"{len(metric_data)}"],
                ["Mean", f"{metric_data.mean():.4f}"],
                ["Median", f"{metric_data.median():.4f}"],
                ["Std Dev", f"{metric_data.std():.4f}"],
                ["Min", f"{metric_data.min():.4f}"],
                ["Max", f"{metric_data.max():.4f}"],
                ["Q1 (25%)", f"{metric_data.quantile(0.25):.4f}"],
                ["Q3 (75%)", f"{metric_data.quantile(0.75):.4f}"],
                [
                    "IQR",
                    f"{metric_data.quantile(0.75) - metric_data.quantile(0.25):.4f}",
                ],
                ["Skewness", f"{metric_data.skew():.4f}"],
                ["Kurtosis", f"{metric_data.kurtosis():.4f}"],
            ]

            # Create table
            table_data = [["Statistic", "Value"]] + stats_data
            table = ax4.table(
                cellText=table_data[1:],
                colLabels=table_data[0],
                cellLoc="center",
                loc="center",
                colWidths=[0.5, 0.5],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            ax4.set_title(f"{metric_name} Statistics")
            ax4.axis("off")

            plt.tight_layout(pad=2.0)

            if save_plots:
                # Save individual metric plot
                metric_filename = f"{metric_name}_distribution"
                self._save_fig(fig, metrics_folder / metric_filename)
                logger.info(f"   ✅ Saved {metric_name} distribution plot")

            plt.show()

            # Log metric statistics
            logger.info(f"   📊 {metric_name} statistics:")
            logger.info(f"      Count: {len(metric_data)}")
            logger.info(f"      Mean: {metric_data.mean():.4f}")
            logger.info(f"      Median: {metric_data.median():.4f}")
            logger.info(f"      Std Dev: {metric_data.std():.4f}")
            logger.info(
                f"      Range: {metric_data.min():.4f} - {metric_data.max():.4f}"
            )

        # Create a comprehensive metrics summary plot
        self._create_metrics_summary_plot(
            full_df, metric_cols, metrics_folder, save_plots
        )

        # Save detailed metrics statistics to file
        metrics_stats_path = metrics_folder / "metrics_statistics.txt"
        with open(metrics_stats_path, "w") as f:
            f.write("Metrics Distribution Analysis\n")
            f.write("=" * 50 + "\n\n")

            for metric_col in metric_cols:
                metric_name = metric_col.replace("metric_", "")
                metric_data = full_df[metric_col].dropna()

                if metric_data.empty:
                    f.write(f"\n{metric_name.upper()}:\n")
                    f.write(f"  No valid data available\n")
                    continue

                # Apply outlier removal if enabled
                if self.remove_outliers:
                    values = metric_data.copy()

                    if metric_col == "metric_fitness":
                        extreme_outliers = values < self.extreme_threshold
                        non_extreme = values[values >= self.extreme_threshold]
                    else:
                        extreme_outliers = pd.Series(
                            [False] * len(values), index=values.index
                        )
                        non_extreme = values

                    if len(non_extreme) > 0:
                        Q1 = non_extreme.quantile(0.25)
                        Q3 = non_extreme.quantile(0.75)
                        IQR = Q3 - Q1

                        if IQR > 0:
                            lower_bound = Q1 - self.outlier_multiplier * IQR
                            upper_bound = Q3 + self.outlier_multiplier * IQR
                            statistical_outliers = (values < lower_bound) | (
                                values > upper_bound
                            )
                        else:
                            statistical_outliers = pd.Series(
                                [False] * len(values), index=values.index
                            )
                    else:
                        statistical_outliers = pd.Series(
                            [False] * len(values), index=values.index
                        )

                    all_outliers = extreme_outliers | statistical_outliers
                    metric_data = values[~all_outliers]

                    if metric_data.empty:
                        f.write(f"\n{metric_name.upper()}:\n")
                        f.write(f"  No valid data after outlier removal\n")
                        continue

                f.write(f"\n{metric_name.upper()}:\n")
                f.write(f"  Count: {len(metric_data)}\n")
                f.write(f"  Mean: {metric_data.mean():.4f}\n")
                f.write(f"  Median: {metric_data.median():.4f}\n")
                f.write(f"  Std Dev: {metric_data.std():.4f}\n")
                f.write(f"  Min: {metric_data.min():.4f}\n")
                f.write(f"  Max: {metric_data.max():.4f}\n")
                f.write(f"  Q1 (25%): {metric_data.quantile(0.25):.4f}\n")
                f.write(f"  Q3 (75%): {metric_data.quantile(0.75):.4f}\n")
                f.write(
                    f"  IQR: {metric_data.quantile(0.75) - metric_data.quantile(0.25):.4f}\n"
                )
                f.write(f"  Skewness: {metric_data.skew():.4f}\n")
                f.write(f"  Kurtosis: {metric_data.kurtosis():.4f}\n")

        logger.info(
            f"✅ Saved detailed metrics statistics to {metrics_stats_path}"
        )
        logger.info(
            f"✅ Metrics distribution analysis completed. Files saved in: {metrics_folder}"
        )

    def _create_metrics_summary_plot(
        self,
        full_df: pd.DataFrame,
        metric_cols: list,
        metrics_folder: Path,
        save_plots: bool = True,
    ):
        """Create a summary plot showing all metrics distributions side by side."""

        if len(metric_cols) < 2:
            logger.info("Not enough metrics for summary plot")
            return

        # Prepare data for summary plot
        summary_data = []
        for metric_col in metric_cols:
            metric_name = metric_col.replace("metric_", "")
            metric_data = full_df[metric_col].dropna()

            if metric_data.empty:
                continue

            # Apply outlier removal if enabled
            if self.remove_outliers:
                values = metric_data.copy()

                if metric_col == "metric_fitness":
                    extreme_outliers = values < self.extreme_threshold
                    non_extreme = values[values >= self.extreme_threshold]
                else:
                    extreme_outliers = pd.Series(
                        [False] * len(values), index=values.index
                    )
                    non_extreme = values

                if len(non_extreme) > 0:
                    Q1 = non_extreme.quantile(0.25)
                    Q3 = non_extreme.quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:
                        lower_bound = Q1 - self.outlier_multiplier * IQR
                        upper_bound = Q3 + self.outlier_multiplier * IQR
                        statistical_outliers = (values < lower_bound) | (
                            values > upper_bound
                        )
                    else:
                        statistical_outliers = pd.Series(
                            [False] * len(values), index=values.index
                        )
                else:
                    statistical_outliers = pd.Series(
                        [False] * len(values), index=values.index
                    )

                all_outliers = extreme_outliers | statistical_outliers
                metric_data = values[~all_outliers]

                if metric_data.empty:
                    continue

            summary_data.append(
                {
                    "metric": metric_name,
                    "data": metric_data,
                    "mean": metric_data.mean(),
                    "median": metric_data.median(),
                    "std": metric_data.std(),
                }
            )

        if not summary_data:
            logger.warning("No valid metric data for summary plot")
            return

        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(
            "All Metrics Summary Analysis", fontsize=18, fontweight="bold"
        )

        # 1. Box plots for all metrics
        ax1 = axes[0, 0]
        metric_names = [d["metric"] for d in summary_data]
        metric_data_list = [d["data"].values for d in summary_data]

        bp = ax1.boxplot(
            metric_data_list, labels=metric_names, patch_artist=True
        )

        # Color the boxes
        colors = plt.cm.Set3.colors[: len(metric_data_list)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_ylabel("Value")
        ax1.set_title("All Metrics Box Plots")
        ax1.tick_params(axis="x", rotation=45, labelsize=10)
        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Violin plots for all metrics
        ax2 = axes[0, 1]
        vp = ax2.violinplot(
            metric_data_list,
            positions=range(len(metric_data_list)),
            showmeans=True,
        )

        # Color the violins
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)

        ax2.set_ylabel("Value")
        ax2.set_title("All Metrics Violin Plots")
        ax2.set_xticks(range(len(metric_names)))
        ax2.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Statistics comparison
        ax3 = axes[1, 0]

        means = [d["mean"] for d in summary_data]
        medians = [d["median"] for d in summary_data]
        stds = [d["std"] for d in summary_data]

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax3.bar(
            x - width / 2,
            means,
            width,
            label="Mean",
            alpha=0.7,
            color="skyblue",
        )
        bars2 = ax3.bar(
            x + width / 2,
            medians,
            width,
            label="Median",
            alpha=0.7,
            color="lightcoral",
        )

        ax3.set_xlabel("Metric")
        ax3.set_ylabel("Value")
        ax3.set_title("Mean vs Median Comparison")
        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Standard deviation comparison
        ax4 = axes[1, 1]

        bars = ax4.bar(
            range(len(metric_names)),
            stds,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        ax4.set_xlabel("Metric")
        ax4.set_ylabel("Standard Deviation")
        ax4.set_title("Standard Deviation Comparison")
        ax4.set_xticks(range(len(metric_names)))
        ax4.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=10)

        # Add value labels on bars
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(stds) * 0.01,
                f"{std:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(pad=3.0)

        if save_plots:
            self._save_fig(fig, metrics_folder / "all_metrics_summary")
            logger.info("✅ Saved all metrics summary plot")

        plt.show()


async def main():
    """Main function to run the evolution fitness analysis."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze evolution fitness data from MetaEvolve"
    )
    parser.add_argument(
        "--redis-host",
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
        "--redis-db", type=int, default=0, help="Redis database (default: 0)"
    )
    parser.add_argument(
        "--redis-prefix", required=True, help="Redis key prefix (required)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top programs to analyze (default: 10)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting (just export data)",
    )
    parser.add_argument(
        "--no-fitness-plots",
        action="store_true",
        help="Skip fitness evolution plots",
    )
    parser.add_argument(
        "--no-stage-plots",
        action="store_true",
        help="Skip program stage statistics plots",
    )
    parser.add_argument(
        "--no-persistence-plots",
        action="store_true",
        help="Skip time since last update analysis plots",
    )
    parser.add_argument(
        "--no-dag-stage-plots",
        action="store_true",
        help="Skip DAG stage results statistics plots",
    )
    parser.add_argument(
        "--no-island-plots",
        action="store_true",
        help="Skip island statistics plots",
    )
    parser.add_argument(
        "--no-metric-plots",
        action="store_true",
        help="Skip metric correlation plots",
    )
    parser.add_argument(
        "--no-validity-plots",
        action="store_true",
        help="Skip validity distribution plots",
    )
    parser.add_argument(
        "--no-metric-distribution-plots",
        action="store_true",
        help="Skip metric distribution plots",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder for all results (required)",
    )
    parser.add_argument(
        "--base-filename",
        default="evolution_data",
        help="Base filename for CSV files (default: evolution_data)",
    )
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=-10000.0,
        help="Threshold for extreme outliers (default: -10000.0)",
    )
    parser.add_argument(
        "--outlier-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier for outlier detection (default: 3.0)",
    )
    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Skip outlier removal (keep all fitness values)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Rolling window size for running mean/std calculations (default: 50)",
    )

    args = parser.parse_args()

    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output folder: {output_folder.absolute()}")

    # Create analyzer
    analyzer = EvolutionFitnessAnalyzer(
        redis_prefix=args.redis_prefix,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        extreme_threshold=args.extreme_threshold,
        outlier_multiplier=args.outlier_multiplier,
        remove_outliers=not args.no_outlier_removal,
    )

    try:
        # Extract data
        evolution_df = await analyzer.extract_evolution_data()

        if evolution_df.empty:
            logger.error("No data found. Exiting.")
            return

        # Analyze fitness data
        fitness_analysis = analyzer.analyze_fitness_data(evolution_df)

        if not fitness_analysis:
            logger.error("No valid fitness data found. Exiting.")
            return

        # Print summary
        logger.info(
            f"\n🏆 Best Fitness: {fitness_analysis['best_fitness']:.4f}"
        )
        logger.info(f"📊 Mean Fitness: {fitness_analysis['mean_fitness']:.4f}")
        logger.info(
            f"📈 Total Programs with Fitness: {fitness_analysis['total_programs']}"
        )
        logger.info(
            f"📈 Total All Programs: {fitness_analysis['total_all_programs']}"
        )

        # Create plots (unless disabled)
        if not args.no_plots:
            # Create fitness evolution plots
            if not args.no_fitness_plots:
                try:
                    analyzer.plot_fitness_evolution(
                        fitness_analysis, output_folder, rolling_window=args.rolling_window
                    )
                    logger.info("✅ Fitness evolution plots completed")
                except Exception as e:
                    logger.error(
                        f"❌ Error creating fitness evolution plots: {e}"
                    )

            # Create program stage statistics plots
            if not args.no_stage_plots:
                try:
                    analyzer.plot_program_stage_statistics(
                        fitness_analysis, output_folder
                    )
                    logger.info("✅ Program stage statistics plots completed")
                except Exception as e:
                    logger.error(
                        f"❌ Error creating program stage statistics plots: {e}"
                    )

            # Create time since last update analysis plots
            if not args.no_persistence_plots:
                try:
                    analyzer.plot_state_persistence_analysis(
                        fitness_analysis, output_folder
                    )
                    logger.info(
                        "✅ Time since last update analysis plots completed"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error creating time since last update analysis plots: {e}"
                    )

            # Create DAG stage results statistics plots
            if not args.no_dag_stage_plots:
                try:
                    analyzer.plot_stage_results_statistics(
                        fitness_analysis, output_folder
                    )
                    logger.info(
                        "✅ DAG stage results statistics plots completed"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error creating DAG stage results statistics plots: {e}"
                    )

            # Create island statistics plots
            if not args.no_island_plots:
                try:
                    analyzer.plot_island_statistics(
                        fitness_analysis, output_folder
                    )
                    logger.info("✅ Island statistics plots completed")
                except Exception as e:
                    logger.error(
                        f"❌ Error creating island statistics plots: {e}"
                    )

            # Create metric correlation heatmap
            if not args.no_metric_plots:
                try:
                    analyzer.plot_metric_correlations(
                        fitness_analysis, output_folder
                    )
                    logger.info("✅ Metric correlation plots completed")
                except Exception as e:
                    logger.error(
                        f"❌ Error creating metric correlation plots: {e}"
                    )

            # Create validity distribution plots
            if not args.no_validity_plots:
                try:
                    analyzer.plot_validity_distribution(
                        fitness_analysis, output_folder
                    )
                    logger.info("✅ Validity distribution plots completed")
                except Exception as e:
                    logger.error(
                        f"❌ Error creating validity distribution plots: {e}"
                    )

            # Create metric distribution plots
            if not args.no_metric_distribution_plots:
                try:
                    analyzer.plot_metric_distributions(
                        fitness_analysis, output_folder
                    )
                    logger.info("✅ Metric distribution plots completed")
                except Exception as e:
                    logger.error(
                        f"❌ Error creating metric distribution plots: {e}"
                    )

        # Analyze top programs
        analyzer.analyze_top_programs(
            evolution_df, fitness_analysis, output_folder, top_n=args.top_n
        )

        # Export data
        analyzer.export_evolution_data(
            fitness_analysis, output_folder, args.base_filename
        )

        # Create a summary of all generated files
        logger.info(f"\n📋 Generated files in {output_folder}:")
        for file_path in output_folder.glob("*"):
            if file_path.is_file():
                logger.info(f"  📄 {file_path.name}")

        logger.info("\n🎉 Analysis complete!")

    except KeyboardInterrupt:
        logger.info("🛑 Analysis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise
    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    # Run the analysis
    asyncio.run(main())
