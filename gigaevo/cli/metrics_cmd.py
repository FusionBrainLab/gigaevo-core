"""`gigaevo metrics` command — dump metrics from Redis for grep-friendly inspection.

Reads the metrics that experiments write into Redis via
:class:`gigaevo.utils.trackers.backends.redis.RedisMetricsBackend` and prints
them one record per line. Default output is plain text suitable for `grep`,
`awk`, and friends; TSV and JSON are also available.

Examples:

    gigaevo -r heilbron@0 metrics | grep tokens
    gigaevo -r heilbron@0 metrics --tag "valid/frontier/*" --tail 20
    gigaevo -r heilbron@0 metrics --tag "*tokens*" --format tsv
    gigaevo -r heilbron@0 metrics --since 100 --until 200
"""

from __future__ import annotations

from datetime import UTC, datetime
import fnmatch
import json
from typing import Any

import click
import redis as redis_lib

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


def _list_tags(r: redis_lib.Redis, key_prefix: str) -> list[str]:
    """Enumerate metric tag names from the `latest` hash for this prefix.

    Histograms aren't written to `latest`, so they aren't enumerable here.
    That matches the issue's default `--kind scalar` and the documented
    non-goal of histogram inspection.
    """
    raw = r.hkeys(f"{key_prefix}:latest")
    return sorted(str(k) for k in raw)


def _safe_tag(tag: str) -> str:
    """Mirror RedisMetricsBackend._k_history sanitization."""
    return tag.replace("/", ":").replace(" ", "_")


def _fetch_history(
    r: redis_lib.Redis, key_prefix: str, tag: str
) -> list[dict[str, Any]]:
    """Return the parsed history list for `tag` (oldest first)."""
    history_key = f"{key_prefix}:history:{_safe_tag(tag)}"
    raw_entries = r.lrange(history_key, 0, -1)
    out: list[dict[str, Any]] = []
    for raw in raw_entries:
        try:
            out.append(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return out


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
    lines = ["\t".join(cols)]
    for rec in records:
        row = [_format_value(rec.get(c, "")) for c in cols]
        lines.append("\t".join(row))
    return "\n".join(lines)


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
    type=int,
    default=None,
    help="Earliest step/iteration to include (inclusive).",
)
@click.option(
    "--until",
    type=int,
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
    type=int,
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
    """Dump metrics from Redis as plain text, one record per line.

    \b
    Plain output (default), one record per line:
        <tag>\\tstep=<n>\\twall=<iso>\\tvalue=<v>

    Read-only — never writes to Redis. Histograms are not enumerable from
    the `latest` hash; use `--kind hist` together with `--tag <exact-name>`
    if you need to inspect a known histogram tag.

    \b
    Examples:
        gigaevo -r heilbron@0 metrics | grep tokens
        gigaevo -r heilbron@0 metrics --tag "valid/frontier/*"
        gigaevo -r heilbron@0 metrics --tag "*tokens*" --tail 10
        gigaevo -r heilbron@0 metrics --since 50 --until 100 --format tsv
    """
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

    redis_factory = ctx.obj.get("redis_factory")
    include_label = len(run_configs) > 1
    all_records: list[dict[str, Any]] = []

    for rc in run_configs:
        spec = rc.run_spec
        if redis_factory:
            r = redis_factory(spec.db)
        else:
            r = redis_lib.Redis(
                host=redis_host, port=redis_port, db=spec.db, decode_responses=True
            )
        try:
            key_prefix = f"{spec.prefix}:metrics"
            tags = _list_tags(r, key_prefix)
            if tag_pattern:
                # If the pattern is exact (no glob meta), allow it through
                # even when the tag is absent from `latest` (e.g. histograms).
                if any(ch in tag_pattern for ch in "*?["):
                    tags = [t for t in tags if fnmatch.fnmatchcase(t, tag_pattern)]
                elif tag_pattern not in tags:
                    tags = [tag_pattern]
                else:
                    tags = [tag_pattern]

            for tag in tags:
                entries = _fetch_history(r, key_prefix, tag)
                records = [_record(tag, e, spec.label) for e in entries]
                records = _filter_kind(records, kind)
                records = _filter_step(records, since, until)
                if tail is not None and tail > 0:
                    records = records[-tail:]
                all_records.extend(records)
        finally:
            r.close()

    fmt = format_name.lower()
    if fmt == "json":
        click.echo(_emit_json(all_records))
    elif fmt == "tsv":
        click.echo(_emit_tsv(all_records, include_label))
    else:
        click.echo(_emit_plain(all_records, include_label))
