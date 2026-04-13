"""SoloPlugin -- standard MAP-Elites experiment monitoring.

Handles any single-population MAP-Elites experiment. Default fallback
for unknown experiment types. Generates inline matplotlib fitness bar
charts and renders standard markdown status tables.

All plotting is done inline via matplotlib.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from gigaevo.monitoring.notifications import (
    PlotAttachment,
    format_status_table_markdown,
)
from gigaevo.monitoring.snapshot import RunSnapshot
from gigaevo.monitoring.watchdog_plugin import WatchdogPlugin, register

_log = logger.bind(component="plugin.solo")


@register("solo")
class SoloPlugin(WatchdogPlugin):
    """Standard MAP-Elites watchdog plugin.

    - generate_plots: inline matplotlib bar chart of latest fitness per run
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
        fig = None

        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("Agg")

            labels = [s.run_spec.label for s in snapshots]
            values = [s.metrics.get("fitness") for s in snapshots]

            valid_pairs = [
                (lbl, val)
                for lbl, val in zip(labels, values)
                if val is not None
            ]

            if not valid_pairs:
                _log.warning("No valid fitness values for solo plot")
                return []

            bar_labels, bar_vals = zip(*valid_pairs)

            fig, ax = plt.subplots(figsize=(max(6, len(bar_labels) * 1.5), 4))
            ax.bar(bar_labels, bar_vals, color="steelblue", alpha=0.8)
            ax.set_ylabel("Fitness")
            ax.set_title(f"Fitness -- Cycle {cycle}")
            fig.tight_layout()

            plot_path = output_dir / f"comparison_cycle_{cycle:04d}.png"
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")

            return [
                PlotAttachment(
                    path=plot_path,
                    caption=f"Fitness curves (cycle {cycle})",
                )
            ]

        except Exception as exc:
            _log.error(f"Solo plot generation failed: {exc}")
            return []
        finally:
            if fig is not None:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:
                    pass

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
