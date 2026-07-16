"""Trajectory subcommand -- iteration-by-iteration fitness trajectory."""

from __future__ import annotations

import click

from gigaevo.cli.metric_history import MetricHistorySource, build_metric_history_source
from gigaevo.cli.output_formatter import OutputFormatter
from gigaevo.cli.run_resolver import RunResolver


def _fetch_trajectory(
    source: MetricHistorySource,
    metric: str,
) -> list[dict]:
    """Fetch per-iteration trajectory data from a metric history source."""
    frontier_entries = source.get_history(f"program_metrics/valid_frontier_{metric}")
    mean_entries = source.get_history(f"program_metrics/valid_iter_{metric}_mean")

    frontier_by_iter: dict[int, float] = {}
    for entry in frontier_entries:
        try:
            frontier_by_iter[int(entry["s"])] = float(entry["v"])
        except (KeyError, ValueError, TypeError):
            pass

    mean_by_iter: dict[int, float] = {}
    for entry in mean_entries:
        try:
            mean_by_iter[int(entry["s"])] = float(entry["v"])
        except (KeyError, ValueError, TypeError):
            pass

    all_iters = sorted(set(frontier_by_iter.keys()) | set(mean_by_iter.keys()))
    current_frontier: float | None = None
    rows: list[dict] = []
    for it in all_iters:
        best = frontier_by_iter.get(it)
        if best is not None:
            # This history is already the canonical frontier. Carry its latest
            # value forward without assuming whether lower or higher is better.
            current_frontier = best
        mean = mean_by_iter.get(it)
        rows.append(
            {
                "Iter": it,
                "Best": current_frontier,
                "Mean": mean,
            }
        )
    return rows


@click.command()
@click.option(
    "--tail",
    type=click.IntRange(min=1),
    default=None,
    help="Show the last N iterations per run and metric.",
)
@click.option(
    "--metric",
    multiple=True,
    default=None,
    help="Metric(s) to display. Repeatable. Auto-discovers from metrics.yaml in --experiment mode.",
)
@click.option(
    "-f",
    "--format",
    "format_name",
    type=click.Choice(["table", "json", "csv", "markdown"], case_sensitive=False),
    default=None,
    help=(
        "Output format override (table|json|csv|markdown). Passed AFTER "
        "the subcommand — overrides the global `-f/--format` flag when "
        "given."
    ),
)
@click.pass_context
def trajectory(
    ctx: click.Context,
    tail: int | None,
    metric: tuple[str, ...],
    format_name: str | None,
) -> None:
    """Show iteration-by-iteration metric trajectory (running best + mean).

    Reads `valid_frontier_<metric>` (the canonical running frontier)
    and `valid_iter_<metric>_mean` histories from Redis or disk JSONL.
    Values are plain (unsmoothed); use `gigaevo plot trajectory` for plots.
    Disk paths use the standard `<run-dir>/metrics` directory next to storage.

    Metric selection: if `--metric` is omitted, metrics are auto-discovered
    from the resolved runs' `metric_names` (populated from
    `experiment.yaml`'s manifest in `-e/--experiment` mode, falling back
    to `["fitness"]`). Pass `--metric` one or more times to override.
    """
    formatter = ctx.obj["formatter"]
    if format_name is not None:
        formatter = OutputFormatter(format_name=format_name)
        ctx.obj["formatter"] = formatter
    experiment = ctx.obj["experiment"]
    runs = ctx.obj["runs"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    run_configs = RunResolver.resolve(
        experiment=experiment,
        runs=runs,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    # Auto-discover metrics from run_configs when none explicitly specified
    if not metric:
        seen: set[str] = set()
        discovered: list[str] = []
        for rc in run_configs:
            for m in rc.metric_names:
                if m not in seen:
                    seen.add(m)
                    discovered.append(m)
        metrics_to_show = discovered if discovered else ["fitness"]
    else:
        metrics_to_show = list(metric)

    redis_factory = ctx.obj.get("redis_factory")
    all_rows: list[dict] = []
    rows_per_metric: dict[str, int] = dict.fromkeys(metrics_to_show, 0)

    for rc in run_configs:
        spec = rc.run_spec
        source: MetricHistorySource | None = None
        try:
            source = build_metric_history_source(
                spec,
                redis_host,
                redis_port,
                redis_factory=redis_factory,
            )
            for m in metrics_to_show:
                rows = _fetch_trajectory(source, m)
                rows_per_metric[m] += len(rows)
                if tail is not None:
                    rows = rows[-tail:]
                for row in rows:
                    if len(run_configs) > 1:
                        row["Label"] = spec.label
                    if len(metrics_to_show) > 1:
                        row["Metric"] = m
                all_rows.extend(rows)
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(
                f"Failed to read trajectory for {spec.label}: {exc}"
            ) from exc
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass

    # If user explicitly asked for metric(s) that returned zero rows across
    # all runs, warn on stderr — otherwise the empty [] output is confusing.
    if metric:
        empty = [m for m in metrics_to_show if rows_per_metric[m] == 0]
        if empty:
            click.echo(
                f"Warning: no trajectory data for metric(s): {', '.join(empty)}. "
                f"Check the metric name or whether the run has emitted data yet.",
                err=True,
            )

    columns = ["Iter", "Best", "Mean"]
    if len(run_configs) > 1:
        columns = ["Label"] + columns
    if len(metrics_to_show) > 1:
        columns = ["Metric"] + columns

    formatter.echo(all_rows, columns=columns, title="Trajectory")
