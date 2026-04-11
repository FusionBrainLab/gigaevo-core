"""SoloPlugin -- standard MAP-Elites experiment monitoring.

Handles any single-population MAP-Elites experiment. Default fallback
for unknown experiment types. Generates comparison.py fitness curves
and renders standard markdown status tables.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from loguru import logger

from gigaevo.monitoring.notifications import (
    PlotAttachment,
    format_status_table_markdown,
)
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, register

_log = logger.bind(component="plugin.solo")

# Project root for tools/comparison.py
_PROJ = Path(__file__).resolve().parent.parent.parent.parent


@register("solo")
class SoloPlugin(WatchdogPlugin):
    """Standard MAP-Elites watchdog plugin.

    - generate_plots: calls tools/comparison.py subprocess
    - format_status_body: markdown header + status table
    """

    def generate_plots(
        self,
        snapshots: list[RunSnapshot],
        output_dir: Path,
        cycle: int,
    ) -> list[PlotAttachment]:
        if not snapshots:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        run_args: list[str] = []
        for snap in snapshots:
            spec = snap.run_spec
            run_args.extend(["--run", f"{spec.prefix}@{spec.db}:{spec.label}"])

        cmd = [
            sys.executable,
            str(_PROJ / "tools" / "comparison.py"),
            *run_args,
            "--annotate-frontier",
            "--output-folder",
            str(output_dir),
        ]

        try:
            subprocess.run(
                cmd,
                cwd=str(_PROJ),
                env={"PYTHONPATH": str(_PROJ)},
                capture_output=True,
                timeout=120,
                check=True,
            )
        except Exception as exc:
            _log.error(f"comparison.py failed: {exc}")
            return []

        png = output_dir / "evolution_runs_comparison.png"
        if png.exists():
            stamped = output_dir / f"comparison_cycle_{cycle:04d}.png"
            shutil.copy2(png, stamped)
            return [
                PlotAttachment(path=stamped, caption=f"Fitness curves (cycle {cycle})")
            ]

        return []

    def format_status_body(
        self,
        snapshots: list[RunSnapshot],
        experiment_name: str,
        cycle: int,
        max_generations: int | None,
    ) -> str:
        from datetime import UTC, datetime

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        progress = f" / {max_generations}" if max_generations else ""

        header = f"### Watchdog #{cycle} -- {experiment_name} -- {now}\n\n"
        table = format_status_table_markdown(snapshots)
        footer = f"\n\n*Cycle {cycle}{progress} -- posted by WatchdogEngine*"

        return header + table + footer
