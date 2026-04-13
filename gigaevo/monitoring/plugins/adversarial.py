"""AdversarialPlugin -- paired arms-race experiment monitoring.

Groups runs by prefix and generates multi-metric panel plots with
configurable metrics from watchdog_plugin_options.plot_metrics.
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

_log = logger.bind(component="plugin.adversarial")


@register("adversarial")
class AdversarialPlugin(WatchdogPlugin):
    """Adversarial arms-race watchdog plugin with multi-metric panels.

    - generate_plots: multi-panel matplotlib (metrics x groups grid)
    - format_status_body: grouped tables with group headers
    - extra_telegram_content: best metric highlight
    """

    def __init__(
        self,
        plot_metrics: list[str] | None = None,
        problem_name: str | None = None,
    ):
        self._plot_metrics = plot_metrics or ["fitness"]
        if plot_metrics and problem_name:
            from gigaevo.monitoring.manifest_schema import WatchdogPluginOptions

            WatchdogPluginOptions(
                plot_metrics=plot_metrics
            ).validate_plot_metrics(problem_name)

    def _group_runs(
        self, snapshots: list[RunSnapshot]
    ) -> dict[str, list[RunSnapshot]]:
        groups: dict[str, list[RunSnapshot]] = defaultdict(list)
        for snap in snapshots:
            groups[snap.run_spec.prefix].append(snap)
        return dict(groups)

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

            groups = self._group_runs(snapshots)
            n_groups = max(len(groups), 1)
            n_metrics = len(self._plot_metrics)

            fig, axes = plt.subplots(
                n_metrics,
                n_groups,
                figsize=(6 * n_groups, 4 * n_metrics),
                squeeze=False,
            )

            for col_idx, (group_name, group_snaps) in enumerate(
                sorted(groups.items())
            ):
                for row_idx, metric in enumerate(self._plot_metrics):
                    ax = axes[row_idx][col_idx]
                    labels = [s.run_spec.label for s in group_snaps]
                    values = [s.metrics.get(metric) for s in group_snaps]

                    valid_pairs = [
                        (lbl, val)
                        for lbl, val in zip(labels, values)
                        if val is not None
                    ]
                    if valid_pairs:
                        bar_labels, bar_vals = zip(*valid_pairs)
                        ax.bar(bar_labels, bar_vals, color="steelblue", alpha=0.8)

                    ax.set_title(f"{group_name}\n{metric}")
                    ax.set_ylabel(metric)

            fig.suptitle(f"Adversarial Metrics -- Cycle {cycle}", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            plot_path = output_dir / f"adversarial_panel_cycle_{cycle:04d}.png"
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")

            return [
                PlotAttachment(
                    path=plot_path,
                    caption=f"Adversarial metrics panel (cycle {cycle})",
                )
            ]

        except Exception as exc:
            _log.error(f"Adversarial plot generation failed: {exc}")
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

        if not snapshots:
            return header + "_No runs to display._\n"

        groups = self._group_runs(snapshots)
        sections: list[str] = []
        for group_name, group_snaps in sorted(groups.items()):
            sections.append(
                f"**{group_name}**\n\n{format_status_table_markdown(group_snaps)}"
            )

        body = "\n\n".join(sections)
        footer = (
            f"\n\n*Cycle {cycle}{progress} -- posted by WatchdogEngine (adversarial)*"
        )

        return header + body + footer

    def extra_telegram_content(self, snapshots: list[RunSnapshot]) -> str | None:
        if not snapshots:
            return None

        first_metric = self._plot_metrics[0]
        best_val = None
        best_label = None
        for snap in snapshots:
            val = snap.metrics.get(first_metric)
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_label = snap.run_spec.label

        if best_val is not None:
            return f"Best {first_metric}: {best_val:.5f} ({best_label})"
        return None
