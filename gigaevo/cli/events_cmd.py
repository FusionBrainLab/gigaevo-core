"""`gigaevo events` command group.

General-purpose event analytics. Event discovery is driven by
`CANONICAL_EVENTS`, so new event classes are auto-visible to the plot tool
with zero CLI changes. Role semantics are not hardcoded — `--group-by` takes
a regex whose named capture group supplies the grouping key.

Example:

    gigaevo -e heilbron/k5-budget-v3 events plot \\
        --log experiments/heilbron/k5-budget-v3/smoke.log \\
        --group-by '.*_(?P<role>[GD])$' \\
        --out experiments/heilbron/k5-budget-v3/event_distribution
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import click

# Trigger registration.
import gigaevo  # noqa: F401
from gigaevo.monitoring.events import CANONICAL_EVENTS
from tools.experiment.log_audit import _EVENT_LINE_RE


@click.group("events")
def events() -> None:
    """Canonical-event analytics (general, registry-driven)."""


def _parse_log(log_path: Path) -> list[dict]:
    """Return every well-formed canonical-event payload from the log.

    Each entry carries the event name, run_label (None if absent), and the
    raw payload dict. Malformed lines are silently skipped — use
    `gigaevo events audit` / `tools.experiment.log_audit` for strict
    validation; plotting is tolerant by design.
    """
    out: list[dict] = []
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EVENT_LINE_RE.search(raw)
        if not match:
            continue
        name = match.group("name")
        if name not in CANONICAL_EVENTS:
            continue
        try:
            payload = json.loads(match.group("json"))
        except json.JSONDecodeError:
            continue
        payload.pop("event", None)
        out.append(
            {
                "name": name,
                "run_label": payload.get("run_label"),
                "payload": payload,
            }
        )
    return out


def _apply_filters(
    rows: list[dict],
    events_filter: set[str] | None,
    runs_filter: set[str] | None,
) -> list[dict]:
    out = rows
    if events_filter:
        out = [r for r in out if r["name"] in events_filter]
    if runs_filter:
        out = [r for r in out if (r["run_label"] or "") in runs_filter]
    return out


def _group_by_regex(
    rows: list[dict], pattern: re.Pattern[str]
) -> tuple[dict[str, list[dict]], list[str]]:
    """Group rows by the first named-capture value from `pattern`.

    Returns (groups, ungrouped_labels). Rows with no run_label or a
    non-matching label are collected under `ungrouped_labels`.
    """
    groups: dict[str, list[dict]] = {}
    ungrouped_labels: set[str] = set()
    for row in rows:
        label = row.get("run_label")
        if not label:
            continue
        match = pattern.search(label)
        if not match or not match.groupdict():
            ungrouped_labels.add(label)
            continue
        # Pick the first non-None named group as the bucket key.
        key = next((v for v in match.groupdict().values() if v is not None), None)
        if key is None:
            ungrouped_labels.add(label)
            continue
        groups.setdefault(key, []).append(row)
    return groups, sorted(ungrouped_labels)


def _counts_by_run(rows: list[dict]) -> dict[str, Counter]:
    """{run_label: Counter(event_name -> count)}."""
    out: dict[str, Counter] = {}
    for row in rows:
        label = row.get("run_label") or "<no-label>"
        out.setdefault(label, Counter())[row["name"]] += 1
    return out


def _draw_counts_per_run(
    counts: dict[str, Counter], out_path: Path, event_names: list[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    runs = sorted(counts.keys())
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 4))
    bottom = [0.0] * len(runs)
    for name in event_names:
        vals = [counts[r].get(name, 0) for r in runs]
        ax.bar(runs, vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("event count")
    ax.set_title("Canonical events per run")
    ax.legend(loc="upper right", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _draw_events_over_time(rows: list[dict], out_path: Path) -> None:
    """Cumulative-count plot indexed by each event's `gen` field when present.

    Events without a `gen` field are counted under a synthetic bucket at gen=0.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_event: dict[str, list[int]] = {}
    for row in rows:
        gen = row["payload"].get("gen")
        gen_i = int(gen) if isinstance(gen, (int, float)) else 0
        per_event.setdefault(row["name"], []).append(gen_i)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, gens in sorted(per_event.items()):
        gens_sorted = sorted(gens)
        cum = list(range(1, len(gens_sorted) + 1))
        ax.plot(gens_sorted, cum, label=name)
    ax.set_xlabel("gen")
    ax.set_ylabel("cumulative events")
    ax.set_title("Canonical events over time")
    ax.legend(loc="upper left", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _draw_group_totals(
    groups: dict[str, list[dict]], out_path: Path, event_names: list[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_keys = sorted(groups.keys())
    fig, ax = plt.subplots(figsize=(max(5, len(group_keys) * 1.5), 4))
    bottom = [0.0] * len(group_keys)
    for name in event_names:
        vals = [sum(1 for r in groups[k] if r["name"] == name) for k in group_keys]
        ax.bar(group_keys, vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("event count")
    ax.set_title("Canonical events by group")
    ax.legend(loc="upper right", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _max_observed_gen(rows: list[dict]) -> int:
    best = -1
    for row in rows:
        if row["name"] != "GENERATION_BOUNDARY":
            continue
        gen = row["payload"].get("gen")
        if isinstance(gen, int) and gen > best:
            best = gen
    return best


def _render_summary(
    rows: list[dict],
    counts_per_run: dict[str, Counter],
    groups: dict[str, list[dict]] | None,
    ungrouped: list[str],
    event_names: list[str],
    group_by: str | None,
) -> str:
    lines = ["# Canonical Events Summary", ""]
    lines.append(f"Total events parsed: {len(rows)}")
    lines.append("")
    lines.append("## Per-run counts")
    lines.append("")
    for label in sorted(counts_per_run):
        items = ", ".join(f"{n}={counts_per_run[label].get(n, 0)}" for n in event_names)
        lines.append(f"- `{label}`: {items}")
    lines.append("")

    if group_by and groups is not None:
        lines.append("## Grouping")
        lines.append("")
        lines.append(f"`--group-by {group_by}`")
        lines.append("")
        for key in sorted(groups):
            n = len(groups[key])
            lines.append(f"- `{key}`: {n} events")
        if ungrouped:
            lines.append("")
            lines.append("### Ungrouped")
            lines.append("")
            for label in ungrouped:
                lines.append(f"- `{label}` (no regex match)")
        lines.append("")

    # Registry health questions and missing-by-gen check.
    max_gen = _max_observed_gen(rows)
    lines.append("## Registry health check")
    lines.append("")
    for name in sorted(CANONICAL_EVENTS):
        if event_names and name not in event_names:
            continue
        cls = CANONICAL_EVENTS[name]
        q = getattr(cls, "health_question", "") or ""
        expected = int(getattr(cls, "expected_after_gen", 0) or 0)
        total = sum(c.get(name, 0) for c in counts_per_run.values())
        flag = ""
        if expected > 0 and max_gen >= expected and total == 0:
            flag = (
                f" — MISSING (expected_after_gen={expected}, "
                f"observed max gen={max_gen})"
            )
        suffix = f" — {q}" if q else ""
        lines.append(f"- **{name}** ({total}){suffix}{flag}")
    lines.append("")

    return "\n".join(lines)


@events.command("plot")
@click.option(
    "--log",
    "log_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="Log file to parse. Repeatable.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory. Created if missing.",
)
@click.option(
    "--events",
    "events_csv",
    default="all",
    help="Comma-separated event names to include, or 'all'.",
)
@click.option(
    "--runs",
    "runs_csv",
    default="all",
    help="Comma-separated run labels to include, or 'all'.",
)
@click.option(
    "--group-by",
    "group_by",
    default=None,
    help="Regex with a named capture group used for grouping runs.",
)
@click.option("--no-over-time", is_flag=True, help="Skip events_over_time.png")
@click.option("--no-totals", is_flag=True, help="Skip counts_per_run.png")
@click.option("--no-group", is_flag=True, help="Skip the grouped plot")
def plot(
    log_paths: tuple[Path, ...],
    out_dir: Path,
    events_csv: str,
    runs_csv: str,
    group_by: str | None,
    no_over_time: bool,
    no_totals: bool,
    no_group: bool,
) -> None:
    """Emit per-run and optionally grouped event plots + summary.md."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for log_path in log_paths:
        rows.extend(_parse_log(log_path))

    events_filter: set[str] | None
    if events_csv.strip().lower() == "all":
        events_filter = None
    else:
        events_filter = {n.strip() for n in events_csv.split(",") if n.strip()}

    runs_filter: set[str] | None
    if runs_csv.strip().lower() == "all":
        runs_filter = None
    else:
        runs_filter = {n.strip() for n in runs_csv.split(",") if n.strip()}

    rows = _apply_filters(rows, events_filter, runs_filter)
    counts_per_run = _counts_by_run(rows)

    # Stable event ordering — prefer registry order, fall back to sorted.
    if events_filter:
        event_names = sorted(events_filter)
    else:
        seen = {r["name"] for r in rows}
        event_names = [n for n in CANONICAL_EVENTS if n in seen]

    if not no_totals:
        _draw_counts_per_run(
            counts_per_run, out_dir / "counts_per_run.png", event_names
        )
    if not no_over_time:
        _draw_events_over_time(rows, out_dir / "events_over_time.png")

    groups: dict[str, list[dict]] | None = None
    ungrouped: list[str] = []
    if group_by:
        pattern = re.compile(group_by)
        groups, ungrouped = _group_by_regex(rows, pattern)
        if not no_group and groups:
            _draw_group_totals(groups, out_dir / "role_totals.png", event_names)

    summary = _render_summary(
        rows, counts_per_run, groups, ungrouped, event_names, group_by
    )
    (out_dir / "summary.md").write_text(summary)
    click.echo(f"Wrote {out_dir}")
