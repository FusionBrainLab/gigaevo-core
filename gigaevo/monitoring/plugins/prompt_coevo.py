"""PromptCoevoPlugin -- prompt co-evolution experiment monitoring.

Co-evolution experiments have two populations:
- Code runs: evolve Python programs (prefix like chains/hover/static_soft)
- Prompt runs: evolve mutation prompts (prefix like prompt_evolution_hover)

This plugin groups runs by prefix and generates separate comparison plots
for each population, with population-specific status formatting.
"""

from __future__ import annotations

from collections import defaultdict
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

_log = logger.bind(component="plugin.prompt_coevo")
_PROJ = Path(__file__).resolve().parent.parent.parent.parent


@register("prompt_coevo")
class PromptCoevoPlugin(WatchdogPlugin):
    """Prompt co-evolution watchdog plugin.

    Groups runs by prefix (code vs prompt populations).
    Each group gets its own comparison.py plot and status section.
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
            group_dir = output_dir / safe_name
            group_dir.mkdir(parents=True, exist_ok=True)

            run_args: list[str] = []
            for snap in group_snaps:
                spec = snap.run_spec
                run_args.extend(["--run", f"{spec.prefix}@{spec.db}:{spec.label}"])

            cmd = [
                sys.executable,
                str(_PROJ / "tools" / "comparison.py"),
                *run_args,
                "--annotate-frontier",
                "--output-folder",
                str(group_dir),
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
                _log.error(f"comparison.py failed for group {group_name}: {exc}")
                continue

            png = group_dir / "evolution_runs_comparison.png"
            if png.exists():
                pop_type = self._classify_group(group_name)
                stamped = output_dir / f"{safe_name}_cycle_{cycle:04d}.png"
                shutil.copy2(png, stamped)
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
