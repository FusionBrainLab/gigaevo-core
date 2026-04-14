"""SoloPlugin -- standard MAP-Elites experiment monitoring.

Delegates plot generation to `gigaevo plot comparison` CLI command
via subprocess. Provides run-by-run Telegram formatting.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from loguru import logger

from gigaevo.monitoring.notifications import (
    PlotAttachment,
    format_status_table_markdown,
)
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, register

_log = logger.bind(component="plugin.solo")

_SUBPROCESS_TIMEOUT = 120


@register("solo")
class SoloPlugin(WatchdogPlugin):
    """Standard MAP-Elites watchdog plugin.

    - generate_plots: delegates to gigaevo plot comparison CLI
    - format_status_body: markdown header + status table
    - format_telegram_body: run-by-run format with stall flags
    """

    def __init__(self, sentinel_value: float | None = None, **kwargs):
        self._sentinel_value = sentinel_value

    def _sentinel_args(self) -> list[str]:
        if self._sentinel_value is not None:
            return ["--sentinel", str(self._sentinel_value)]
        return []

    @staticmethod
    def _build_run_args(snapshots: list[RunSnapshot]) -> list[str]:
        """Build -r args for CLI from snapshots."""
        args: list[str] = []
        for snap in snapshots:
            spec = snap.run_spec
            args.extend(["-r", f"{spec.prefix}@{spec.db}:{spec.label}"])
        return args

    def generate_plots(
        self,
        snapshots: list[RunSnapshot],
        output_dir: Path,
        cycle: int,
    ) -> list[PlotAttachment]:
        if not snapshots:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        run_args = self._build_run_args(snapshots)

        cmd = (
            ["gigaevo"]
            + run_args
            + [
                "plot",
                "comparison",
                "--metric",
                "fitness",
                "--smoothing",
                "ema",
                "--window",
                "5",
                "-o",
                str(output_dir),
            ]
            + self._sentinel_args()
        )

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=_SUBPROCESS_TIMEOUT
            )
            if result.returncode != 0:
                _log.warning(
                    f"Comparison plot failed: {result.stderr.decode()[:500]}"
                )
                return []
        except Exception as exc:
            _log.error(f"Solo plot subprocess error: {exc}")
            return []

        comp_png = output_dir / "evolution_runs_comparison.png"
        if comp_png.exists():
            stamped = output_dir / f"comparison_cycle_{cycle:04d}.png"
            shutil.copy2(comp_png, stamped)
            return [
                PlotAttachment(
                    path=stamped,
                    caption=f"Fitness curves (cycle {cycle})",
                )
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

    def format_telegram_body(
        self,
        snapshots: list[RunSnapshot],
        experiment_name: str,
        cycle: int,
        max_generations: int | None,
        baseline: float | None = None,
    ) -> str | None:
        lines = [f"Experiment: {experiment_name} #{cycle}"]
        lines.append("")
        for s in snapshots:
            fit = s.metrics.get("fitness")
            fit_str = f"{fit:.5f}" if fit is not None else "N/A"
            gen = s.generation or 0
            max_g = f"/{max_generations}" if max_generations else ""
            stalled = (
                s.running_programs is not None
                and s.running_programs == 0
                and gen > 0
            )
            flag = "!" if stalled else "ok"
            lines.append(
                f"  {flag} {s.run_spec.label}: gen {gen}{max_g} fit={fit_str}"
            )
        if baseline is not None:
            lines.append(f"\n  SOTA baseline: {baseline:.5f}")
        return "\n".join(lines)
