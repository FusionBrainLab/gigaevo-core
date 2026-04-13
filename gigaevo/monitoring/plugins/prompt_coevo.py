"""PromptCoevoPlugin -- prompt co-evolution experiment monitoring.

Co-evolution experiments have two populations:
- Code runs: evolve Python programs (prefix like chains/hover/static_soft)
- Prompt runs: evolve mutation prompts (prefix like prompt_evolution_hover)

Delegates plot generation to `gigaevo plot comparison` CLI command
per population group via subprocess. Provides grouped Telegram formatting.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from loguru import logger

from gigaevo.monitoring.notifications import (
    PlotAttachment,
    format_status_table_markdown,
)
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, register

_log = logger.bind(component="plugin.prompt_coevo")

_SUBPROCESS_TIMEOUT = 120


@register("prompt_coevo")
class PromptCoevoPlugin(WatchdogPlugin):
    """Prompt co-evolution watchdog plugin.

    Groups runs by prefix (code vs prompt populations).
    Each group gets its own comparison plot via CLI subprocess.
    """

    def _group_runs(self, snapshots: list[RunSnapshot]) -> dict[str, list[RunSnapshot]]:
        groups: dict[str, list[RunSnapshot]] = defaultdict(list)
        for snap in snapshots:
            groups[snap.run_spec.prefix].append(snap)
        return dict(groups)

    def _classify_group(self, group_name: str) -> str:
        """Classify a group as 'code' or 'prompt' based on prefix."""
        if "prompt" in group_name.lower():
            return "Prompt Population"
        return "Code Population"

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
        groups = self._group_runs(snapshots)
        plots: list[PlotAttachment] = []

        for group_name, group_snaps in groups.items():
            safe_name = group_name.replace("/", "_")
            run_args = self._build_run_args(group_snaps)
            pop_type = self._classify_group(group_name)

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
            )

            try:
                result = subprocess.run(
                    cmd, capture_output=True, timeout=_SUBPROCESS_TIMEOUT
                )
                if result.returncode != 0:
                    _log.warning(
                        f"Plot for group {group_name} failed: "
                        f"{result.stderr.decode()[:500]}"
                    )
                    continue
            except Exception as exc:
                _log.error(f"Plot subprocess error for group {group_name}: {exc}")
                continue

            comp_png = output_dir / "evolution_runs_comparison.png"
            if comp_png.exists():
                stamped = output_dir / f"{safe_name}_cycle_{cycle:04d}.png"
                shutil.copy2(comp_png, stamped)
                plots.append(
                    PlotAttachment(
                        path=stamped,
                        caption=f"{pop_type} ({group_name}) fitness curves (cycle {cycle})",
                    )
                )

        return plots

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

        if not snapshots:
            return header + "_No runs to display._\n"

        groups = self._group_runs(snapshots)
        sections: list[str] = []
        for group_name, group_snaps in sorted(groups.items()):
            pop_type = self._classify_group(group_name)
            sections.append(
                f"**{pop_type}** (`{group_name}`)\n\n"
                f"{format_status_table_markdown(group_snaps)}"
            )

        body = "\n\n".join(sections)
        footer = (
            f"\n\n*Cycle {cycle}{progress} -- posted by WatchdogEngine (prompt_coevo)*"
        )

        return header + body + footer

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

        groups = self._group_runs(snapshots)
        for group_name, group_snaps in sorted(groups.items()):
            pop_type = self._classify_group(group_name)
            lines.append(f"{pop_type} ({group_name}):")
            for s in group_snaps:
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
            lines.append("")

        if baseline is not None:
            lines.append(f"SOTA baseline: {baseline:.5f}")

        return "\n".join(lines)
