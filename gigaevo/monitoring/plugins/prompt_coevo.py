"""PromptCoevoPlugin -- prompt co-evolution experiment monitoring.

Co-evolution experiments have two populations:
- Code runs: evolve Python programs (prefix like chains/hover/static_soft)
- Prompt runs: evolve mutation prompts (prefix like prompt_evolution_hover)

This plugin groups runs by prefix and generates inline matplotlib bar
charts per population, with population-specific status formatting.

All plotting is done inline via matplotlib.
"""

from __future__ import annotations

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


@register("prompt_coevo")
class PromptCoevoPlugin(WatchdogPlugin):
    """Prompt co-evolution watchdog plugin.

    Groups runs by prefix (code vs prompt populations).
    Each group gets its own inline matplotlib bar chart and status section.
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
            fig = None

            try:
                import matplotlib
                import matplotlib.pyplot as plt

                matplotlib.use("Agg")

                labels = [s.run_spec.label for s in group_snaps]
                values = [s.metrics.get("fitness") for s in group_snaps]

                valid_pairs = [
                    (lbl, val)
                    for lbl, val in zip(labels, values)
                    if val is not None
                ]

                if not valid_pairs:
                    _log.warning(f"No valid fitness values for group {group_name}")
                    continue

                bar_labels, bar_vals = zip(*valid_pairs)
                pop_type = self._classify_group(group_name)

                fig, ax = plt.subplots(
                    figsize=(max(6, len(bar_labels) * 1.5), 4)
                )
                ax.bar(bar_labels, bar_vals, color="steelblue", alpha=0.8)
                ax.set_ylabel("Fitness")
                ax.set_title(f"{pop_type} ({group_name}) -- Cycle {cycle}")
                fig.tight_layout()

                plot_path = output_dir / f"{safe_name}_cycle_{cycle:04d}.png"
                fig.savefig(plot_path, dpi=100, bbox_inches="tight")

                plots.append(
                    PlotAttachment(
                        path=plot_path,
                        caption=f"{pop_type} ({group_name}) fitness curves (cycle {cycle})",
                    )
                )

            except Exception as exc:
                _log.error(f"Plot generation failed for group {group_name}: {exc}")
                continue
            finally:
                if fig is not None:
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass

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
