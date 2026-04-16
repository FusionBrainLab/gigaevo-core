"""AdversarialPlugin -- paired arms-race experiment monitoring.

Delegates plot generation to `gigaevo plot` CLI commands (arms-race,
comparison) via subprocess. Provides G/D-separated Telegram formatting
with SOTA comparison.

Population role detection uses ``run_spec.role`` (``"constructor"`` = G,
``"improver"`` = D). The role field is populated from the manifest via
``ManifestRunSpec`` — runs with no role indicator are skipped.
"""

from __future__ import annotations

from collections import defaultdict
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


def _constructors(snapshots: list[RunSnapshot]) -> list[RunSnapshot]:
    return [s for s in snapshots if s.run_spec.role == "constructor"]


def _improvers(snapshots: list[RunSnapshot]) -> list[RunSnapshot]:
    return [s for s in snapshots if s.run_spec.role == "improver"]


_log = logger.bind(component="plugin.adversarial")

_SUBPROCESS_TIMEOUT = 120


@register("adversarial")
class AdversarialPlugin(WatchdogPlugin):
    """Adversarial arms-race watchdog plugin.

    - generate_plots: delegates to gigaevo plot arms-race + comparison CLI
    - format_status_body: grouped tables with group headers
    - format_telegram_body: G/D-separated with SOTA comparison
    """

    def __init__(
        self,
        plot_metrics: list[str] | None = None,
        plot_commands: list | None = None,
        sentinel_value: float | None = None,
    ):
        self._plot_metrics = plot_metrics or ["fitness"]
        self._plot_commands = plot_commands or []
        self._sentinel_value = sentinel_value

    def _group_runs(self, snapshots: list[RunSnapshot]) -> dict[str, list[RunSnapshot]]:
        groups: dict[str, list[RunSnapshot]] = defaultdict(list)
        for snap in snapshots:
            groups[snap.run_spec.prefix].append(snap)
        return dict(groups)

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
        plots: list[PlotAttachment] = []
        run_args = self._build_run_args(snapshots)
        metric = self._plot_metrics[0] if self._plot_metrics else "fitness"

        if self._plot_commands:
            for pc in self._plot_commands:
                plot = self._run_plot_command(
                    pc, run_args, snapshots, output_dir, cycle
                )
                if plot:
                    plots.append(plot)
        else:
            arms_race = self._generate_arms_race(
                run_args, snapshots, output_dir, cycle, metric
            )
            if arms_race:
                plots.append(arms_race)

            comparison = self._generate_comparison(
                run_args, snapshots, output_dir, cycle, metric
            )
            if comparison:
                plots.append(comparison)

        return plots

    def _generate_arms_race(
        self,
        run_args: list[str],
        snapshots: list[RunSnapshot],
        output_dir: Path,
        cycle: int,
        metric: str,
    ) -> PlotAttachment | None:
        g_labels = [s.run_spec.label for s in _constructors(snapshots)]
        d_labels = [s.run_spec.label for s in _improvers(snapshots)]
        if not g_labels or not d_labels:
            _log.warning("Cannot generate arms-race: missing G or D runs")
            return None

        paired_arg = ",".join(f"{g}:{d}" for g, d in zip(g_labels, d_labels))
        cmd = (
            ["gigaevo"]
            + run_args
            + [
                "plot",
                "arms-race",
                "--metric",
                metric,
                "--paired",
                paired_arg,
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
                _log.warning(f"Arms-race plot failed: {result.stderr.decode()[:500]}")
                return None
        except Exception as exc:
            _log.error(f"Arms-race subprocess error: {exc}")
            return None

        png = output_dir / "arms_race.png"
        if png.exists():
            stamped = output_dir / f"arms_race_cycle_{cycle:04d}.png"
            shutil.copy2(png, stamped)
            return PlotAttachment(
                path=stamped, caption=f"Arms-race dynamics (cycle {cycle})"
            )
        return None

    def _generate_comparison(
        self,
        run_args: list[str],
        snapshots: list[RunSnapshot],
        output_dir: Path,
        cycle: int,
        metric: str,
    ) -> PlotAttachment | None:
        d_labels = [s.run_spec.label for s in _improvers(snapshots)]

        cmd = (
            ["gigaevo"]
            + run_args
            + [
                "plot",
                "comparison",
                "--metric",
                metric,
                "--smoothing",
                "ema",
                "--window",
                "10",
                "--annotate-frontier",
                "--max-annotations",
                "3",
                "-o",
                str(output_dir),
            ]
            + self._sentinel_args()
        )
        if d_labels:
            cmd.extend(["--no-frontier-for", ",".join(d_labels)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=_SUBPROCESS_TIMEOUT
            )
            if result.returncode != 0:
                _log.warning(f"Comparison plot failed: {result.stderr.decode()[:500]}")
                return None
        except Exception as exc:
            _log.error(f"Comparison subprocess error: {exc}")
            return None

        comp_png = output_dir / "evolution_runs_comparison.png"
        if comp_png.exists():
            stamped = output_dir / f"comparison_cycle_{cycle:04d}.png"
            shutil.copy2(comp_png, stamped)
            return PlotAttachment(
                path=stamped, caption=f"Fitness comparison (cycle {cycle})"
            )
        return None

    def _run_plot_command(
        self,
        plot_command,
        run_args: list[str],
        snapshots: list[RunSnapshot],
        output_dir: Path,
        cycle: int,
    ) -> PlotAttachment | None:
        # Auto-inject --paired for arms-race when not provided: pair constructor
        # (G) runs with improver (D) runs in declaration order.
        args = dict(plot_command.args)
        if plot_command.command == "arms-race" and "paired" not in args:
            g_labels = [s.run_spec.label for s in _constructors(snapshots)]
            d_labels = [s.run_spec.label for s in _improvers(snapshots)]
            if g_labels and d_labels:
                args["paired"] = ",".join(
                    f"{g}:{d}" for g, d in zip(g_labels, d_labels)
                )

        # Auto-inject --no-frontier-for on improver labels for comparison so we
        # don't draw monotonic frontiers over a non-monotonic D population.
        if plot_command.command == "comparison" and "no-frontier-for" not in args:
            d_labels = [s.run_spec.label for s in _improvers(snapshots)]
            if d_labels:
                args["no-frontier-for"] = ",".join(d_labels)

        cmd = ["gigaevo"] + run_args + ["plot", plot_command.command]
        for key, val in args.items():
            # Booleans map to Click-style flags: True -> --key, False -> --no-key.
            # Everything else is rendered as "--key value".
            if isinstance(val, bool):
                cmd.append(f"--{key}" if val else f"--no-{key}")
            else:
                cmd.extend([f"--{key}", str(val)])
        cmd.extend(["-o", str(output_dir)])
        cmd.extend(self._sentinel_args())

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=_SUBPROCESS_TIMEOUT
            )
            if result.returncode != 0:
                _log.warning(
                    f"Plot '{plot_command.command}' failed: "
                    f"{result.stderr.decode()[:500]}"
                )
                return None
        except Exception as exc:
            _log.error(f"Plot '{plot_command.command}' error: {exc}")
            return None

        output_name = (
            plot_command.output_name or f"{plot_command.command.replace('-', '_')}.png"
        )
        out_file = output_dir / output_name
        if out_file.exists():
            stamped = output_dir / f"{out_file.stem}_cycle_{cycle:04d}.png"
            shutil.copy2(out_file, stamped)
            caption = plot_command.caption or f"{plot_command.command} (cycle {cycle})"
            return PlotAttachment(path=stamped, caption=caption)
        return None

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

        g_snaps = _constructors(snapshots)
        d_snaps = _improvers(snapshots)

        metric = self._plot_metrics[0] if self._plot_metrics else "fitness"

        lines.append(f"Constructor (G) -- {metric}:")
        for s in g_snaps:
            fit = s.metrics.get(metric)
            fit_str = f"{fit:.5f}" if fit is not None else "N/A"
            vs_sota = ""
            if fit is not None and baseline is not None and baseline > 0:
                ratio = fit / baseline
                vs_sota = f" ({ratio:.1%} of SOTA)"
            gen = s.generation or 0
            max_g = f"/{max_generations}" if max_generations else ""
            stalled = (
                s.running_programs is not None and s.running_programs == 0 and gen > 0
            )
            flag = "!" if stalled else "ok"
            lines.append(
                f"  {flag} {s.run_spec.label}: gen {gen}{max_g} fit={fit_str}{vs_sota}"
            )

        if baseline is not None:
            lines.append(f"  SOTA baseline: {baseline:.5f}")

        lines.append("")

        lines.append(f"Improver (D) -- {metric}:")
        for s in d_snaps:
            fit = s.metrics.get(metric)
            fit_str = f"{fit:.5f}" if fit is not None else "N/A"
            gen = s.generation or 0
            max_g = f"/{max_generations}" if max_generations else ""
            stalled = (
                s.running_programs is not None and s.running_programs == 0 and gen > 0
            )
            flag = "!" if stalled else "ok"
            lines.append(f"  {flag} {s.run_spec.label}: gen {gen}{max_g} fit={fit_str}")

        if baseline is not None and g_snaps and d_snaps:
            lines.append("")
            lines.append(f"max(G,D) {metric} per pair vs SOTA:")
            for g_s, d_s in zip(g_snaps, d_snaps):
                g_fit = g_s.metrics.get(metric)
                d_fit = d_s.metrics.get(metric)
                vals = [v for v in (g_fit, d_fit) if v is not None]
                pair_max = max(vals) if vals else None
                pair_str = f"{pair_max:.5f}" if pair_max is not None else "N/A"
                vs_sota = ""
                if pair_max is not None and baseline > 0:
                    vs_sota = f" ({pair_max / baseline:.1%} of SOTA)"
                pair_name = g_s.run_spec.label.replace("_G", "")
                lines.append(f"  {pair_name}: {pair_str}{vs_sota}")

        if max_generations and all(
            s.generation is not None and s.generation >= max_generations
            for s in snapshots
        ):
            lines.append("")
            lines.append("ALL RUNS COMPLETE -- run closeout")

        return "\n".join(lines)

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
