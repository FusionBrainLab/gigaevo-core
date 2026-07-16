"""`gigaevo metrics` command — dump persisted metrics for inspection.

Reads Redis or disk JSONL metric history and prints it one record per line.
Default output is suitable for `grep` and `awk`; TSV and JSON are available.

Examples:

    gigaevo -r heilbron@0 metrics | grep tokens
    gigaevo -r heilbron@0 metrics --tag "valid/frontier/*" --tail 20
    gigaevo -r heilbron@0 metrics --tag "*tokens*" --format tsv
    gigaevo -r heilbron@0 metrics --since 100 --until 200
    gigaevo -r outputs/run metrics --tag "valid/frontier/*"
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import fnmatch
from io import StringIO
import json
from typing import Any

import click

from gigaevo.cli.metric_history import MetricHistorySource, build_metric_history_source
from gigaevo.cli.run_resolver import RunResolver

KIND_CHOICES = ("scalar", "hist", "text", "all")
FORMAT_CHOICES = ("plain", "tsv", "json")


def _iso_wall(wall: Any) -> str:
    """Render a `wall_time` epoch float as ISO-8601 UTC. Best-effort."""
    try:
        ts = float(wall)
    except (TypeError, ValueError):
        return str(wall)
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _record(tag: str, entry: dict[str, Any], label: str | None) -> dict[str, Any]:
    """Normalize one history entry into a record dict."""
    rec: dict[str, Any] = {
        "tag": tag,
        "step": entry.get("s"),
        "wall": _iso_wall(entry.get("t")),
        "kind": entry.get("k", "scalar"),
        "value": entry.get("v"),
    }
    if label:
        rec["label"] = label
    return rec


def _filter_step(
    records: list[dict[str, Any]], since: int | None, until: int | None
) -> list[dict[str, Any]]:
    if since is None and until is None:
        return records
    out: list[dict[str, Any]] = []
    for rec in records:
        step = rec.get("step")
        try:
            step_i = int(step)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if since is not None and step_i < since:
            continue
        if until is not None and step_i > until:
            continue
        out.append(rec)
    return out


def _filter_kind(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "all":
        return records
    return [r for r in records if r.get("kind") == kind]


def _format_value(v: Any) -> str:
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"), default=str)
    return str(v)


def _emit_plain(records: list[dict[str, Any]], include_label: bool) -> str:
    lines: list[str] = []
    for rec in records:
        parts: list[str] = []
        if include_label and rec.get("label"):
            parts.append(f"label={rec['label']}")
        parts.append(str(rec["tag"]))
        parts.append(f"step={rec['step']}")
        parts.append(f"wall={rec['wall']}")
        parts.append(f"value={_format_value(rec['value'])}")
        lines.append("\t".join(parts))
    return "\n".join(lines)


def _emit_tsv(records: list[dict[str, Any]], include_label: bool) -> str:
    cols = ["tag", "step", "wall", "kind", "value"]
    if include_label:
        cols = ["label"] + cols
    output = StringIO()
    writer = csv.writer(output, dialect="excel-tab", lineterminator="\n")
    writer.writerow(cols)
    for rec in records:
        row = [_format_value(rec.get(c, "")) for c in cols]
        writer.writerow(row)
    return output.getvalue().rstrip("\n")


def _emit_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, default=str)


@click.command("metrics")
@click.option(
    "--tag",
    "tag_pattern",
    default=None,
    help="Glob pattern to filter tag names (e.g. 'valid/iter/*', '*tokens*').",
)
@click.option(
    "--since",
    type=click.IntRange(min=0),
    default=None,
    help="Earliest step/iteration to include (inclusive).",
)
@click.option(
    "--until",
    type=click.IntRange(min=0),
    default=None,
    help="Latest step/iteration to include (inclusive).",
)
@click.option(
    "--kind",
    type=click.Choice(KIND_CHOICES, case_sensitive=False),
    default="scalar",
    show_default=True,
    help="Filter by metric kind.",
)
@click.option(
    "--format",
    "format_name",
    type=click.Choice(FORMAT_CHOICES, case_sensitive=False),
    default="plain",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--tail",
    type=click.IntRange(min=1),
    default=None,
    help="Show only the last N records per tag.",
)
@click.pass_context
def metrics(
    ctx: click.Context,
    tag_pattern: str | None,
    since: int | None,
    until: int | None,
    kind: str,
    format_name: str,
    tail: int | None,
) -> None:
    """Dump Redis or disk metric history, one record per line.

    \b
    Plain output (default), one record per line:
        <tag>\\tstep=<n>\\twall=<iso>\\tvalue=<v>

    Read-only. Disk paths use the default `<run-dir>/metrics` directory next
    to `<run-dir>/storage`.

    \b
    Examples:
        gigaevo -r heilbron@0 metrics | grep tokens
        gigaevo -r heilbron@0 metrics --tag "valid/frontier/*"
        gigaevo -r heilbron@0 metrics --tag "*tokens*" --tail 10
        gigaevo -r heilbron@0 metrics --since 50 --until 100 --format tsv
        gigaevo -r outputs/run metrics --tag "valid/frontier/*"
    """
    experiment = ctx.obj["experiment"]
    runs = ctx.obj["runs"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    if since is not None and until is not None and since > until:
        raise click.UsageError("--since must be less than or equal to --until")

    run_configs = RunResolver.resolve(
        experiment=experiment,
        runs=runs,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    redis_factory = ctx.obj.get("redis_factory")
    include_label = len(run_configs) > 1
    all_records: list[dict[str, Any]] = []

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
            tags = source.list_tags()
            if tag_pattern:
                if any(ch in tag_pattern for ch in "*?["):
                    tags = [t for t in tags if fnmatch.fnmatchcase(t, tag_pattern)]
                else:
                    tags = [tag_pattern]

            for tag in tags:
                entries = source.get_history(tag)
                records = [_record(tag, e, spec.label) for e in entries]
                records = _filter_kind(records, kind)
                records = _filter_step(records, since, until)
                if tail is not None:
                    records = records[-tail:]
                all_records.extend(records)
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(
                f"Failed to read metrics for {spec.label}: {exc}"
            ) from exc
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass

    fmt = format_name.lower()
    if fmt == "json":
        click.echo(_emit_json(all_records))
    elif fmt == "tsv":
        click.echo(_emit_tsv(all_records, include_label))
    else:
        click.echo(_emit_plain(all_records, include_label))
