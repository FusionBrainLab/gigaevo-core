"""Flow profiler: parse evolution log, emit text summary + HTML diagram.

Parses per-event timeline from a runner log:

- DAG run starts/completions
- Per-stage execution timings (canonical ``STAGE_EXEC`` JSON events)
- Cached-stage skips
- ``ParentRefresher`` flips (parent re-evaluation triggers)
- Ingest decisions (accepted / rejected)

and renders both a plain-text summary (pipeline metrics) and an interactive
Plotly HTML diagram (per-program lifecycle bars, stage sub-bars, refresh +
re-eval bands, ingest decision lines).

Used by ``gigaevo profiler``. The renderer is fully self-contained — Plotly
is imported lazily inside :func:`render_full_html` so the parser remains
dependency-free for callers that only need the data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

# ----- log parsing -----

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
DAG_START_RE = re.compile(r"\[DAG\]\[([0-9a-f]{8})\] Run started")
DAG_DONE_RE = re.compile(r"\[DagScheduler\] DAG completed for ([0-9a-f]{8})")
MUT_RE = re.compile(
    r"\[mutation\] Task (\d+): \[(.*?)\] [-→]+>?\s*([0-9a-f]{8})"
    r"(?:\s*\(model=([^,)]+?),\s*archetype=([^,)]+?)(?:,\s*prompt_id=[^)]+)?\))?"
)
ADD_RE = re.compile(r"MultiIsland: adding program ([0-9a-f]{8})")
ADD_OK_RE = re.compile(r"successfully added to island")
REJECT_RE = re.compile(r"\[ingestor\] ([0-9a-f]{8}) REJECTED")
ENGINE_REJECT_RE = re.compile(r"Program ([0-9a-f]{8}) REJECTED")
REFRESH_RE = re.compile(r"ParentRefresher: flipped (\d+) parents DONE->QUEUED")
STAGE_EXEC_RE = re.compile(r"\[STAGE_EXEC\] (\{.*\})$")
LLM_CALL_RE = re.compile(r"\[LLM_CALL\] (\{.*\})$")
BACKPRESSURE_RE = re.compile(r"\[BACKPRESSURE_SAMPLE\] (\{.*\})$")
DAG_CACHED_RE = re.compile(
    r"\[DAG\]\[([0-9a-f]{8})\] Stages CACHED \(skipped execution\): \[(.*?)\]"
)


@dataclass
class StageRun:
    stage: str
    start: datetime
    end: datetime
    decision: str
    cache_key_hash: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start).total_seconds() * 1000.0


@dataclass
class Program:
    short_id: str
    parents: tuple[str, ...] = ()
    birth: datetime | None = None
    dag_starts: list[datetime] = field(default_factory=list)
    dag_dones: list[datetime] = field(default_factory=list)
    accepted: datetime | None = None
    rejected: datetime | None = None
    refreshed_at: list[datetime] = field(default_factory=list)
    stage_runs: list[StageRun] = field(default_factory=list)
    mutation_archetype: str | None = None
    mutation_model: str | None = None
    row: int = -1

    @property
    def first_dag_done(self) -> datetime | None:
        return self.dag_dones[0] if self.dag_dones else None


@dataclass
class LLMCallEvent:
    """One ``[LLM_CALL]`` canonical event.

    Reconstructed start = ``end - duration_ms``; downstream overlap math
    treats every event as an interval ``[start, end]``.
    """

    stage: str
    program_id: str | None
    end: datetime
    duration_ms: float
    ok: bool
    model: str
    error_type: str | None

    @property
    def start(self) -> datetime:
        return self.end - timedelta(milliseconds=self.duration_ms)

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class BackpressureSampleEvent:
    """One ``[BACKPRESSURE_SAMPLE]`` canonical event.

    The flow profiler renders ``producer_held`` and ``buffer_held`` against
    ``max_in_flight`` so an operator can read saturation directly off the
    dashboard. ``llm_active`` breaks down producer occupancy into LLM vs DAG
    phases (llm_active tasks in LLM inference, producer_held-llm_active in DAG).
    Time-series ordering is preserved by ``parse_log``.
    """

    timestamp: datetime
    producer_held: int
    buffer_held: int
    in_flight: int
    max_in_flight: int
    llm_active: int


def parse_ts(line: str) -> datetime | None:
    m = TS_RE.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


def parse_log(
    path: Path,
) -> tuple[
    dict[str, Program],
    list[datetime],
    list[LLMCallEvent],
    list[BackpressureSampleEvent],
]:
    """Parse the runner log into program timelines + refresh + LLM events +
    backpressure samples.

    Returns ``(programs_by_short_id, refresh_timestamps, llm_events,
    backpressure_samples)``. Backpressure samples are ordered by log
    appearance (monotonic timestamps).
    """
    programs: dict[str, Program] = {}
    refreshes: list[datetime] = []
    llm_events: list[LLMCallEvent] = []
    backpressure_samples: list[BackpressureSampleEvent] = []
    pending_refresh: datetime | None = None
    last_add: str | None = None

    with path.open() as f:
        for raw in f:
            ts = parse_ts(raw)
            if ts is None:
                continue
            if REFRESH_RE.search(raw):
                pending_refresh = ts
                refreshes.append(ts)
                continue
            if m := MUT_RE.search(raw):
                _, plist, child = m.group(1), m.group(2), m.group(3)
                model = m.group(4)
                archetype = m.group(5)
                parents = tuple(p.strip("'\" ") for p in plist.split(",") if p.strip())
                prog = programs.setdefault(child, Program(short_id=child))
                prog.parents = parents
                prog.birth = ts
                if archetype:
                    prog.mutation_archetype = archetype.strip()
                if model:
                    prog.mutation_model = model.strip()
                for p in parents:
                    par = programs.setdefault(p, Program(short_id=p))
                    if pending_refresh is not None:
                        par.refreshed_at.append(pending_refresh)
                pending_refresh = None
                continue
            if m := BACKPRESSURE_RE.search(raw):
                try:
                    ev = json.loads(m.group(1))
                except Exception:
                    continue
                try:
                    backpressure_samples.append(
                        BackpressureSampleEvent(
                            timestamp=ts,
                            producer_held=int(ev.get("producer_held", 0)),
                            buffer_held=int(ev.get("buffer_held", 0)),
                            in_flight=int(ev.get("in_flight", 0)),
                            max_in_flight=int(ev.get("max_in_flight", 0)),
                            llm_active=int(ev.get("llm_active", 0)),
                        )
                    )
                except (TypeError, ValueError):
                    # Malformed integer fields — silently skip rather than
                    # taking the parser down with rubbish.
                    pass
                continue
            if m := LLM_CALL_RE.search(raw):
                try:
                    ev = json.loads(m.group(1))
                except Exception:
                    continue
                pid_full = ev.get("program_id")
                pid_short = pid_full[:8] if isinstance(pid_full, str) else None
                llm_events.append(
                    LLMCallEvent(
                        stage=str(ev.get("stage", "?")),
                        program_id=pid_short,
                        end=ts,
                        duration_ms=float(ev.get("latency_ms", 0.0)),
                        ok=bool(ev.get("ok", False)),
                        model=str(ev.get("model", "unknown")),
                        error_type=ev.get("error_type"),
                    )
                )
                continue
            if m := DAG_START_RE.search(raw):
                programs.setdefault(
                    m.group(1), Program(short_id=m.group(1))
                ).dag_starts.append(ts)
                continue
            if m := DAG_DONE_RE.search(raw):
                programs.setdefault(
                    m.group(1), Program(short_id=m.group(1))
                ).dag_dones.append(ts)
                continue
            if m := ADD_RE.search(raw):
                last_add = m.group(1)
                programs.setdefault(last_add, Program(short_id=last_add))
                continue
            if ADD_OK_RE.search(raw) and last_add is not None:
                programs[last_add].accepted = ts
                last_add = None
                continue
            if m := (REJECT_RE.search(raw) or ENGINE_REJECT_RE.search(raw)):
                programs.setdefault(
                    m.group(1), Program(short_id=m.group(1))
                ).rejected = ts
                continue
            if m := DAG_CACHED_RE.search(raw):
                sid = m.group(1)
                stages = [s.strip("' \"") for s in m.group(2).split(",") if s.strip()]
                prog = programs.setdefault(sid, Program(short_id=sid))
                for s in stages:
                    prog.stage_runs.append(
                        StageRun(stage=s, start=ts, end=ts, decision="cached_skip")
                    )
                continue
            if m := STAGE_EXEC_RE.search(raw):
                try:
                    ev = json.loads(m.group(1))
                except Exception:
                    continue
                pid_full = ev.get("program_id")
                if not pid_full:
                    continue
                sid = pid_full[:8]
                stage = ev.get("stage", "?")
                dur = float(ev.get("duration_ms", 0.0))
                decision = ev.get("decision", "miss")
                end = ts
                start = end - timedelta(milliseconds=dur)
                prog = programs.setdefault(sid, Program(short_id=sid))
                prog.stage_runs.append(
                    StageRun(
                        stage=stage,
                        start=start,
                        end=end,
                        decision=decision,
                        cache_key_hash=ev.get("cache_key_hash"),
                    )
                )

    return programs, refreshes, llm_events, backpressure_samples


def assign_rows(programs: dict[str, Program]) -> list[Program]:
    """Iteration-order row assignment: seeds first, then strictly by birth.

    Birth time is the only stable proxy for evolution iteration available
    here (we don't track Task N on Program). Keeping seeds at the top is
    natural — they have ``birth=None`` and precede every mutant. The "last
    N rows" window on the dashboard therefore always shows the most
    recently produced programs.
    """
    seeds = [p for p in programs.values() if p.birth is None]
    born = [p for p in programs.values() if p.birth is not None]
    seeds.sort(key=lambda p: p.short_id)
    born.sort(key=lambda p: p.birth)  # type: ignore[arg-type,return-value]
    ordered = seeds + born
    for i, p in enumerate(ordered):
        p.row = i
    return ordered


# ----- refresh-event pairing (shared by text summary and HTML render) -----


@dataclass
class RefreshPair:
    """One ``ParentRefresher`` flip paired with its next DAG re-eval (if any).

    ``queue_s`` is the wall delay between the flip and the next DAG start on
    the parent; ``exec_s`` is the wall duration of that re-eval DAG run.
    ``exec_s`` is ``None`` when the run was cancelled before completion,
    ``queue_s`` is ``None`` when the run ended before a DAG re-start fired.
    """

    program: Program
    flip_at: datetime
    queue_s: float | None
    exec_s: float | None


def pair_refreshes(ordered: list[Program]) -> tuple[list[RefreshPair], int, int]:
    """Pair each refresh flip with the next-unused DAG start on that program.

    Returns ``(pairs, no_start_count, no_done_count)``.
    """
    pairs: list[RefreshPair] = []
    no_start = 0
    no_done = 0
    for p in ordered:
        used_s: set[int] = set()
        used_d: set[int] = set()
        for rt in p.refreshed_at:
            paired_start: datetime | None = None
            paired_done: datetime | None = None
            for i, ds in enumerate(p.dag_starts):
                if i in used_s or ds <= rt:
                    continue
                paired_start = ds
                used_s.add(i)
                break
            if paired_start is not None:
                for j, dd in enumerate(p.dag_dones):
                    if j in used_d or dd <= paired_start:
                        continue
                    paired_done = dd
                    used_d.add(j)
                    break
            q_s = (paired_start - rt).total_seconds() if paired_start else None
            e_s = (
                (paired_done - paired_start).total_seconds()
                if paired_done and paired_start
                else None
            )
            if q_s is None:
                no_start += 1
            elif e_s is None:
                no_done += 1
            pairs.append(RefreshPair(program=p, flip_at=rt, queue_s=q_s, exec_s=e_s))
    return pairs, no_start, no_done


# ----- stage classification + utilization aggregation -----


_LLM_STAGE_NAMES = frozenset(
    {
        "LineageStage",
        "InsightsStage",
        "LineageAgent",
        "InsightsAgent",
        "MutationAgent",
    }
)
_EXEC_STAGE_NAMES = frozenset(
    {
        "CallProgramFunction",
        "CallValidatorFunction",
    }
)


def classify_stage(name: str) -> str:
    """Bucket a stage / agent name into ``llm`` / ``exec`` / ``orchestration``.

    The two buckets that count toward utilization:

    - ``llm`` — LLM-bound stages and the LangGraph agents (LineageStage,
      InsightsStage, plus the canonical ``*Agent`` names emitted on
      ``LLM_CALL`` events).
    - ``exec`` — program/validator execution (``CallProgramFunction``,
      ``CallValidatorFunction``).

    Anything else (orchestration glue, sub-second housekeeping) maps to
    ``orchestration`` and is ignored by :func:`compute_utilization`.
    """
    if name in _LLM_STAGE_NAMES:
        return "llm"
    if name in _EXEC_STAGE_NAMES:
        return "exec"
    return "orchestration"


def _union_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge a list of ``(start, end)`` pairs into a disjoint union."""
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda x: x[0])
    out: list[tuple[datetime, datetime]] = [ivs[0]]
    for s, e in ivs[1:]:
        last_s, last_e = out[-1]
        if s <= last_e:
            out[-1] = (last_s, max(last_e, e))
        else:
            out.append((s, e))
    return out


def _total_seconds(intervals: list[tuple[datetime, datetime]]) -> float:
    return sum((e - s).total_seconds() for s, e in intervals)


def _intersect_total_s(
    a: list[tuple[datetime, datetime]],
    b: list[tuple[datetime, datetime]],
) -> float:
    """Total seconds of overlap between two union-ed interval lists."""
    if not a or not b:
        return 0.0
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += (e - s).total_seconds()
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _peak_concurrent(intervals: list[tuple[datetime, datetime]]) -> int:
    """Max simultaneous overlapping intervals (sweep-line)."""
    if not intervals:
        return 0
    points: list[tuple[datetime, int]] = []
    for s, e in intervals:
        points.append((s, 1))
        points.append((e, -1))
    points.sort(key=lambda x: (x[0], -x[1]))
    cur = peak = 0
    for _, delta in points:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


@dataclass
class SaturationReport:
    """Aggregated answer to "is ``max_in_flight`` actually being saturated?".

    ``producer_saturation_pct`` and ``buffer_saturation_pct`` count the
    fraction of BACKPRESSURE_SAMPLE snapshots where the respective held
    count equals the per-sample ``max_in_flight``. Near 100% means the cap
    is the load-bearing constraint (true saturation); well below 100%
    means something upstream of the cap (LLM rate, parent selection,
    Redis) is the actual limiter.

    ``peak_llm_active`` and ``peak_dag_active`` break down producer
    occupancy into LLM inference vs DAG evaluation phases.
    """

    sample_count: int
    max_in_flight: int
    peak_producer_held: int
    peak_buffer_held: int
    peak_llm_active: int
    peak_dag_active: int
    producer_saturation_pct: float
    buffer_saturation_pct: float


def compute_saturation(samples: list[BackpressureSampleEvent]) -> SaturationReport:
    """Aggregate a backpressure time-series into a saturation report.

    Saturation is per-sample (``held == max_in_flight`` for that sample's
    own cap), so a mid-run cap change is handled gracefully — the report
    surfaces the LAST observed cap as the headline number that matches
    the engine's current configuration.

    LLM vs DAG breakdown: ``llm_active`` is the count of tasks in LLM
    inference; ``dag_active = producer_held - llm_active`` is the count
    in DAG evaluation. Peaks are tracked separately.
    """
    if not samples:
        return SaturationReport(
            sample_count=0,
            max_in_flight=0,
            peak_producer_held=0,
            peak_buffer_held=0,
            peak_llm_active=0,
            peak_dag_active=0,
            producer_saturation_pct=0.0,
            buffer_saturation_pct=0.0,
        )
    n = len(samples)
    producer_sat = sum(1 for s in samples if s.producer_held >= s.max_in_flight)
    buffer_sat = sum(1 for s in samples if s.buffer_held >= s.max_in_flight)
    peak_llm = max(s.llm_active for s in samples)
    peak_dag = max(s.producer_held - s.llm_active for s in samples)
    return SaturationReport(
        sample_count=n,
        max_in_flight=samples[-1].max_in_flight,
        peak_producer_held=max(s.producer_held for s in samples),
        peak_buffer_held=max(s.buffer_held for s in samples),
        peak_llm_active=peak_llm,
        peak_dag_active=peak_dag,
        producer_saturation_pct=100.0 * producer_sat / n,
        buffer_saturation_pct=100.0 * buffer_sat / n,
    )


@dataclass
class UtilizationReport:
    """Aggregated efficiency view across an entire run.

    All seconds-valued fields are interval *union* totals — concurrent
    work on multiple programs is counted once. This is what makes
    ``overlap_efficiency`` a meaningful "are we using both the LLM and
    the local executor at the same time" signal rather than a sum of
    per-program timings.
    """

    overlap_s: float
    total_llm_s: float
    total_exec_s: float
    llm_only_s: float
    exec_only_s: float
    overlap_efficiency: float
    peak_concurrent_dags: int
    archetype_counts: dict[str, dict[str, int]]
    model_counts: dict[str, int]
    llm_event_count: int
    llm_failure_count: int


def compute_utilization(
    programs: dict[str, Program],
    refreshes: list[datetime],
    llm_events: list[LLMCallEvent],
) -> UtilizationReport:
    """Aggregate LLM-vs-exec overlap stats across all programs.

    Two buckets:

    - LLM intervals: union of ``[start, end]`` for every ``LLM_CALL``
      event plus every non-cached stage run classified as ``llm``.
    - Exec intervals: union of non-cached stage runs classified as
      ``exec``.

    ``overlap_efficiency`` = overlap / min(total_llm, total_exec). It
    answers: when both sides have work, are we running them in parallel
    or serializing? 1.0 = always parallel; 0.0 = always serialized.
    """
    llm_intervals_raw: list[tuple[datetime, datetime]] = []
    exec_intervals_raw: list[tuple[datetime, datetime]] = []
    dag_intervals_raw: list[tuple[datetime, datetime]] = []

    for ev in llm_events:
        if ev.duration_ms <= 0:
            continue
        llm_intervals_raw.append((ev.start, ev.end))

    for p in programs.values():
        for sr in p.stage_runs:
            if sr.decision == "cached_skip":
                continue
            bucket = classify_stage(sr.stage)
            if bucket == "llm":
                llm_intervals_raw.append((sr.start, sr.end))
            elif bucket == "exec":
                exec_intervals_raw.append((sr.start, sr.end))
        # Pair dag_starts with dag_dones positionally to derive
        # full-DAG intervals for peak_concurrent_dags. Imbalanced lists
        # (cancelled mid-eval) are skipped after the shorter list runs out.
        for s, e in zip(p.dag_starts, p.dag_dones):
            if e > s:
                dag_intervals_raw.append((s, e))

    llm_intervals = _union_intervals(llm_intervals_raw)
    exec_intervals = _union_intervals(exec_intervals_raw)
    total_llm_s = _total_seconds(llm_intervals)
    total_exec_s = _total_seconds(exec_intervals)
    overlap_s = _intersect_total_s(llm_intervals, exec_intervals)
    llm_only_s = max(0.0, total_llm_s - overlap_s)
    exec_only_s = max(0.0, total_exec_s - overlap_s)
    denom = min(total_llm_s, total_exec_s)
    overlap_efficiency = (overlap_s / denom) if denom > 0 else 0.0

    archetype_counts: dict[str, dict[str, int]] = {}
    model_counts: dict[str, int] = {}
    for p in programs.values():
        if p.mutation_archetype:
            row = archetype_counts.setdefault(
                p.mutation_archetype, {"accepted": 0, "rejected": 0, "other": 0}
            )
            if p.accepted:
                row["accepted"] += 1
            elif p.rejected:
                row["rejected"] += 1
            else:
                row["other"] += 1
        if p.mutation_model:
            model_counts[p.mutation_model] = model_counts.get(p.mutation_model, 0) + 1

    llm_failures = sum(1 for ev in llm_events if not ev.ok)

    return UtilizationReport(
        overlap_s=overlap_s,
        total_llm_s=total_llm_s,
        total_exec_s=total_exec_s,
        llm_only_s=llm_only_s,
        exec_only_s=exec_only_s,
        overlap_efficiency=overlap_efficiency,
        peak_concurrent_dags=_peak_concurrent(dag_intervals_raw),
        archetype_counts=archetype_counts,
        model_counts=model_counts,
        llm_event_count=len(llm_events),
        llm_failure_count=llm_failures,
    )


# ----- text summary -----


def _fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%H:%M:%S.%f")[:-3] if dt else "—"


def format_summary_text(
    programs: dict[str, Program],
    refreshes: list[datetime],
    log_path: Path | None = None,
    include_timeline: bool = True,
    utilization: UtilizationReport | None = None,
    saturation: SaturationReport | None = None,
) -> str:
    """Plain-text profile suitable for terminal output or a .txt file."""
    ordered = assign_rows(programs)
    n = len(ordered)
    n_seeds = sum(1 for p in ordered if p.birth is None)
    n_accepted = sum(1 for p in ordered if p.birth is not None and p.accepted)
    n_rej = sum(1 for p in ordered if p.birth is not None and p.rejected)
    n_other = n - n_seeds - n_accepted - n_rej

    pairs, no_start, no_done = pair_refreshes(ordered)
    queue_waits = [pp.queue_s for pp in pairs if pp.queue_s is not None]
    exec_times = [pp.exec_s for pp in pairs if pp.exec_s is not None]
    avg_q = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0
    avg_e = sum(exec_times) / len(exec_times) if exec_times else 0.0
    max_q = max(queue_waits) if queue_waits else 0.0
    max_e = max(exec_times) if exec_times else 0.0
    q_share = (avg_q / (avg_q + avg_e) * 100) if (avg_q + avg_e) else 0.0

    lines: list[str] = []
    sep = "=" * 60
    title = log_path.name if log_path else "evolution flow profile"
    lines.append(f"Flow Profile: {title}")
    if log_path is not None:
        lines.append(f"  source: {log_path}")
    lines.append(sep)
    lines.append("")
    lines.append("Programs:")
    lines.append(f"  total:      {n}")
    lines.append(f"  seeds:      {n_seeds}")
    lines.append(f"  accepted:   {n_accepted}")
    lines.append(f"  rejected:   {n_rej}")
    lines.append(f"  in-flight:  {n_other}")
    lines.append("")
    lines.append("Refresh:")
    lines.append(f"  flips:                 {len(refreshes)}")
    lines.append(f"  paired re-eval runs:   {len(exec_times)} / {len(refreshes)}")
    lines.append(f"  queue wait avg/max:    {avg_q:.2f}s / {max_q:.2f}s")
    lines.append(
        f"  exec avg/max:          {avg_e * 1000:.0f}ms / {max_e * 1000:.0f}ms"
    )
    lines.append(f"  queue share:           {q_share:.0f}%")
    if no_start or no_done:
        lines.append(
            f"  unpaired:              "
            f"{no_start} no-DAG-start (run ended pre-reeval), "
            f"{no_done} no-completion (cancelled mid-eval)"
        )
    else:
        lines.append("  unpaired:              0 (every flip paired cleanly)")

    if utilization is not None and (
        utilization.total_llm_s > 0 or utilization.total_exec_s > 0
    ):
        u = utilization
        lines.append("")
        lines.append("Utilization (interval union, concurrent work counted once):")
        lines.append(f"  LLM-bound wall:        {u.total_llm_s:.1f}s")
        lines.append(f"  exec-bound wall:       {u.total_exec_s:.1f}s")
        lines.append(
            f"  LLM ∩ exec (overlap):  {u.overlap_s:.1f}s  "
            f"({u.overlap_efficiency * 100:.0f}% of min(LLM, exec))"
        )
        lines.append(
            f"  LLM-only / exec-only:  {u.llm_only_s:.1f}s / {u.exec_only_s:.1f}s"
        )
        lines.append(
            f"  peak concurrent DAGs:  {u.peak_concurrent_dags}  ·  "
            f"LLM events: {u.llm_event_count} ({u.llm_failure_count} failed)"
        )
        if u.archetype_counts:
            lines.append("")
            lines.append("Mutation archetypes (accepted / rejected / other):")
            ranked = sorted(
                u.archetype_counts.items(),
                key=lambda kv: kv[1]["accepted"] + kv[1]["rejected"] + kv[1]["other"],
                reverse=True,
            )
            for name, row in ranked:
                lines.append(
                    f"  {name:<32}  "
                    f"{row['accepted']:>3}a / {row['rejected']:>3}r / {row['other']:>3}o"
                )

    if saturation is not None and saturation.sample_count > 0:
        s = saturation
        lines.append("")
        lines.append("Backpressure saturation (per BACKPRESSURE_SAMPLE):")
        lines.append(
            f"  samples / cap:         {s.sample_count} / max_in_flight={s.max_in_flight}"
        )
        lines.append(
            f"  producer @ cap:        {s.producer_saturation_pct:5.1f}%  "
            f"(peak held={s.peak_producer_held}/{s.max_in_flight})"
        )
        lines.append(
            f"  buffer   @ cap:        {s.buffer_saturation_pct:5.1f}%  "
            f"(peak held={s.peak_buffer_held}/{s.max_in_flight})"
        )
        # Phase breakdown of producer occupancy: peak count of tasks in LLM
        # inference vs DAG evaluation. A high peak_llm with low peak_dag means
        # producers are LLM-bound; the inverse points to DAG-eval as the
        # latency-dominant phase. Both phases share the producer-sema cap.
        lines.append(
            f"  peak LLM / DAG split:  {s.peak_llm_active} llm / "
            f"{s.peak_dag_active} dag (of {s.max_in_flight} producer slots)"
        )
        # Interpretation hint: near-100% on producer = LLM/refresh side is the
        # load-bearing constraint; near-100% on buffer = ingestion is the
        # downstream bottleneck. <50% on either = pipeline is not actually full.
        if s.producer_saturation_pct < 50.0 and s.buffer_saturation_pct < 50.0:
            lines.append(
                "  hint:                  cap is NOT the bottleneck; "
                "look upstream (parent selection, refresh, LLM rate)"
            )

    if include_timeline and ordered:
        lines.append("")
        lines.append(
            "Timeline (id  birth         dag_done      wall    stages   ingest      refresh)"
        )
        for p in ordered:
            stage_count = sum(
                1
                for sr in p.stage_runs
                if p.first_dag_done
                and sr.start <= p.first_dag_done
                and sr.decision != "cached_skip"
            )
            cached_count = sum(1 for sr in p.stage_runs if sr.decision == "cached_skip")
            if p.birth and p.first_dag_done and p.first_dag_done > p.birth:
                wall = f"{(p.first_dag_done - p.birth).total_seconds():.2f}s"
            else:
                wall = "—"
            ingest = "accepted" if p.accepted else "rejected" if p.rejected else "—"
            lines.append(
                f"  {p.short_id}  "
                f"{_fmt_dt(p.birth):<13}  "
                f"{_fmt_dt(p.first_dag_done):<13}  "
                f"{wall:>6}  "
                f"{stage_count}e/{cached_count}c    "
                f"{ingest:<10}  "
                f"refresh×{len(p.refreshed_at)}"
            )

    lines.append("")
    return "\n".join(lines)


# ----- HTML render (Plotly imported lazily) -----


# Curated qualitative palette (20 hues) — colorblind-aware mix of Tableau 10
# + Tableau 10 Light + a few extra distinct hues. Designed so adjacent slots
# stay visually distinguishable and no two slots collide under SHA-1 hashing
# for realistic stage-name sets.
STAGE_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f3f97",  # violet
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # tan
    "#f7b6d2",  # light pink
    "#9edae5",  # light cyan
    "#dbdb8d",  # light olive
    "#393b79",  # dark navy
]

# Lightness/saturation variants applied as a deterministic second axis when
# the hue slot alone would collide. Three levels keep the visual distance
# between variants well above the just-noticeable-difference for most stages.
_LIGHTNESS_LEVELS = 3
# Effective palette capacity = len(STAGE_PALETTE) * _LIGHTNESS_LEVELS.


def _hash_slot(name: str, n: int, salt: bytes) -> int:
    """Deterministic, well-distributed slot in ``[0, n)`` keyed by ``name``."""
    digest = hashlib.sha1(salt + name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % n


def _shade(hex_color: str, level: int) -> str:
    """Return ``hex_color`` shifted in lightness for variant ``level``.

    ``level == 0`` returns the base color unchanged. Higher levels nudge the
    color toward white/black in alternating directions so neighbouring
    variants stay visually distinct without wandering off the palette.
    """
    if level == 0:
        return hex_color
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # Alternate darker/lighter shifts; magnitude grows with level so 0/1/2
    # remain pairwise distinguishable on standard displays.
    if level % 2 == 1:
        # Darken
        factor = 1.0 - 0.22 * ((level + 1) // 2)
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))
    else:
        # Lighten (blend toward white)
        weight = 0.28 * (level // 2)
        r = min(255, int(r + (255 - r) * weight))
        g = min(255, int(g + (255 - g) * weight))
        b = min(255, int(b + (255 - b) * weight))
    return f"#{r:02x}{g:02x}{b:02x}"


def stage_color(name: str) -> str:
    """Stable, distinguishable color for a stage name.

    The mapping is deterministic and name-keyed: a given stage always maps
    to the same color, across runs, re-renders, and additions/removals of
    other stages. Two independent SHA-1-derived slots pick a hue and a
    lightness variant, expanding the effective palette to
    ``len(STAGE_PALETTE) * _LIGHTNESS_LEVELS`` distinct colors without
    wrapping back onto the primary palette.
    """
    hue_slot = _hash_slot(name, len(STAGE_PALETTE), b"hue:")
    lit_slot = _hash_slot(name, _LIGHTNESS_LEVELS, b"lit:")
    return _shade(STAGE_PALETTE[hue_slot], lit_slot)


MIN_BAR_VISUAL_MS = 50.0


def _build_label(p: Program) -> str:
    return f"<b>{p.short_id}</b>"


DEFAULT_LAST_N_ROWS = 50


def _visible_rows(n: int, last_n: int | None) -> int:
    """Number of rows to size the canvas for (capped to ``last_n``)."""
    if last_n is None or n <= last_n:
        return n
    return last_n


def _initial_y_range(n: int, last_n: int | None) -> list[float] | None:
    """Initial reversed y-axis range showing the last ``last_n`` rows.

    Returns ``None`` when the full chart fits (``n <= last_n`` or
    ``last_n`` is ``None``); otherwise returns ``[bottom_value, top_value]``
    for a reversed axis so the highest-indexed (newest) rows sit at the
    bottom of the visible window and older rows are scrollable above.
    """
    if last_n is None or n <= last_n:
        return None
    return [n - 0.5, n - last_n - 0.5]


def _yaxis_layout(n: int, y_labels: list[str], last_n: int | None) -> dict:
    """Y-axis layout dict, honoring the ``last_n`` initial window."""
    layout: dict = dict(
        title=None,
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=y_labels,
        tickfont=dict(size=13, color="#24292f"),
        showgrid=False,
        zeroline=False,
        linecolor="#d0d7de",
        showline=True,
        mirror=True,
        automargin=True,
        # Pannable so the user can scroll back to older programs even
        # when the initial window clips to the last N rows.
        fixedrange=False,
    )
    rng = _initial_y_range(n, last_n)
    if rng is None:
        layout["autorange"] = "reversed"
    else:
        layout["autorange"] = False
        layout["range"] = rng
    return layout


def make_figure(
    programs: dict[str, Program],
    refreshes: list[datetime],
    *,
    last_n: int | None = DEFAULT_LAST_N_ROWS,
):
    """Build the Plotly figure. Plotly is imported lazily.

    ``last_n`` controls the initial y-axis window: when set and the number
    of programs exceeds it, only the last ``last_n`` rows (highest indices
    after :func:`assign_rows`) are visible by default. Y-axis is left
    pannable (``fixedrange=False``) so users can scroll back to older
    programs; the toolbar exposes quick row-window buttons too. Set
    ``last_n=None`` to render the full chart with autoranged y.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ordered = assign_rows(programs)
    n = len(ordered)
    y_labels = [_build_label(p) for p in ordered]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])

    BAR_HEIGHT = 0.7
    STAGE_HEIGHT = 0.6
    REEVAL_HEIGHT = 0.36

    def vis(ms: float) -> float:
        return max(ms, MIN_BAR_VISUAL_MS)

    # DAG span backdrop
    # NOTE: caption text was previously rendered inside this bar but Plotly's
    # ``textposition=inside`` clips/hides labels unpredictably under zoom (see
    # issue #230). The DAG identity is conveyed by the birth tick hover and
    # the colored per-stage bars sitting on top of it, so this trace is now
    # purely a visual backdrop with no on-bar text.
    bd_w, bd_b, bd_y, bd_hov = [], [], [], []
    for p in ordered:
        if p.birth and p.first_dag_done and p.first_dag_done > p.birth:
            dur = (p.first_dag_done - p.birth).total_seconds()
            bd_w.append(vis(dur * 1000))
            bd_b.append(p.birth)
            bd_y.append(p.row)
            bd_hov.append(f"<b>DAG {p.short_id}</b><br>duration: {dur:.2f}s")
    if bd_w:
        fig.add_trace(
            go.Bar(
                name="DAG span",
                orientation="h",
                x=bd_w,
                base=bd_b,
                y=bd_y,
                textposition="none",
                cliponaxis=True,
                hovertext=bd_hov,
                hoverinfo="text",
                marker=dict(color="#eef1f4", line=dict(color="#b6bcc4", width=0.5)),
                width=BAR_HEIGHT,
            )
        )

    # Per-stage exec bars
    # On-bar text labels were removed (issue #230): under zoom Plotly's inside
    # text either clipped (small boxes), overflowed neighbours, or disappeared
    # on large boxes via ``uniformtext``. The stage title is reliably surfaced
    # through the hover tooltip instead (issue #231). The hovertemplate keeps
    # the stage name as the bold first line for EVERY box regardless of size.
    st_w, st_b, st_y, st_hov, st_col = [], [], [], [], []
    for p in ordered:
        for sr in p.stage_runs:
            if sr.decision == "cached_skip":
                continue
            dur_ms = sr.duration_ms
            col = stage_color(sr.stage)
            st_w.append(vis(dur_ms))
            st_b.append(sr.start)
            st_y.append(p.row)
            st_col.append(col)
            st_hov.append(
                f"<b>{sr.stage}</b><br>"
                f"program:    {p.short_id}<br>"
                f"decision:   {sr.decision}<br>"
                f"start:      {_fmt_dt(sr.start)}<br>"
                f"end:        {_fmt_dt(sr.end)}<br>"
                f"duration:   {dur_ms:.1f} ms<br>"
                f"cache_hash: {sr.cache_key_hash or '—'}"
            )
    if st_w:
        fig.add_trace(
            go.Bar(
                name="stage exec",
                orientation="h",
                x=st_w,
                base=st_b,
                y=st_y,
                textposition="none",
                cliponaxis=True,
                hovertext=st_hov,
                hoverinfo="text",
                hoverlabel=dict(namelength=-1),
                marker=dict(color=st_col, line=dict(color=st_col, width=0.4)),
                width=STAGE_HEIGHT,
            )
        )

    # Cached-stage ticks
    cs_x, cs_y, cs_hov = [], [], []
    for p in ordered:
        for sr in p.stage_runs:
            if sr.decision != "cached_skip":
                continue
            cs_x.append(sr.start)
            cs_y.append(p.row)
            cs_hov.append(
                f"<b>{sr.stage}</b> — cached (skipped)<br>"
                f"program: {p.short_id}<br>"
                f"at:      {_fmt_dt(sr.start)}"
            )
    if cs_x:
        fig.add_trace(
            go.Scatter(
                name="cached skip",
                x=cs_x,
                y=cs_y,
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    color="#9ba4ad",
                    size=8,
                    line=dict(width=1.2, color="#9ba4ad"),
                ),
                opacity=0.6,
                hovertext=cs_hov,
                hoverinfo="text",
            )
        )

    # Re-eval queue/exec bands
    pairs, _, _ = pair_refreshes(ordered)
    qw_widths, qw_bases, qw_rows, qw_text = [], [], [], []
    refresh_x, refresh_y, refresh_text = [], [], []
    # Need actual start/done timestamps for x-positioning; re-pair on the fly
    # to keep this rendering pass independent of the data class above.
    for p in ordered:
        used_s: set[int] = set()
        used_d: set[int] = set()
        for k, rt in enumerate(p.refreshed_at, 1):
            refresh_x.append(rt)
            refresh_y.append(p.row)
            paired_start = None
            paired_done = None
            for i, ds in enumerate(p.dag_starts):
                if i in used_s or ds <= rt:
                    continue
                paired_start = ds
                used_s.add(i)
                break
            if paired_start is not None:
                for j, dd in enumerate(p.dag_dones):
                    if j in used_d or dd <= paired_start:
                        continue
                    paired_done = dd
                    used_d.add(j)
                    break
            q_s = (paired_start - rt).total_seconds() if paired_start else None
            e_s = (
                (paired_done - paired_start).total_seconds()
                if paired_done and paired_start
                else None
            )
            if q_s is not None and e_s is not None:
                txt = (
                    f"<b>{p.short_id}</b> — refresh #{k}<br>"
                    f"flip @ {_fmt_dt(rt)}<br>"
                    f"queue wait: {q_s:.2f}s<br>"
                    f"exec:       {e_s:.2f}s<br>"
                    f"total:      {(q_s + e_s):.2f}s"
                )
            elif q_s is not None:
                txt = (
                    f"<b>{p.short_id}</b> — refresh #{k}<br>"
                    f"flip @ {_fmt_dt(rt)}<br>"
                    f"queue wait: {q_s:.2f}s<br>"
                    "(DAG started but never completed — run cancelled mid-eval)"
                )
            else:
                txt = (
                    f"<b>{p.short_id}</b> — refresh #{k}<br>"
                    f"flip @ {_fmt_dt(rt)}<br>"
                    "(no later DAG start on this row — run ended before re-eval)"
                )
            refresh_text.append(txt)
            if paired_start and q_s is not None:
                qw_widths.append(vis(q_s * 1000))
                qw_bases.append(rt)
                qw_rows.append(p.row)
                qw_text.append(
                    f"<b>{p.short_id}</b> — refresh #{k} QUEUE WAIT<br>"
                    f"flip:  {_fmt_dt(rt)}<br>"
                    f"start: {_fmt_dt(paired_start)}<br>"
                    f"wait:  {q_s:.2f}s"
                )
            # Re-eval execution is conveyed via the per-stage subbars and
            # the refresh-flip tick; no separate exec-time bar is drawn.
    if qw_widths:
        fig.add_trace(
            go.Bar(
                name="re-eval queue wait",
                orientation="h",
                x=qw_widths,
                base=qw_bases,
                y=qw_rows,
                hovertext=qw_text,
                hoverinfo="text",
                textposition="none",
                marker=dict(color="#b5b5b5", line=dict(color="#7a7a7a", width=0.3)),
                width=REEVAL_HEIGHT,
            )
        )
    # NB: re-eval exec is intentionally NOT drawn as its own bar. Per-stage
    # exec subbars cover the same time range with the same row positioning,
    # so a dark-purple exec band would just overlap them and — when zoomed
    # close — clipped text labels inside the band rendered as vertical-stripe
    # artifacts. The DAG span backdrop + per-stage bars + queue-wait bar
    # already convey total duration, per-stage activity, and queue time.

    # Birth + refresh-flip ticks
    def _ticks(name, xs, ys, hovertext, color, size=14, line_width=2):
        fig.add_trace(
            go.Scatter(
                name=name,
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    color=color,
                    size=size,
                    line=dict(width=line_width, color=color),
                ),
                opacity=0.95,
                hovertext=hovertext,
                hoverinfo="text",
            )
        )

    birth_x, birth_y, birth_hov = [], [], []
    for p in ordered:
        if not (p.birth and p.first_dag_done and p.first_dag_done > p.birth):
            continue
        stage_count = sum(
            1
            for sr in p.stage_runs
            if sr.start <= p.first_dag_done and sr.decision != "cached_skip"
        )
        cached_count = sum(1 for sr in p.stage_runs if sr.decision == "cached_skip")
        dur = (p.first_dag_done - p.birth).total_seconds()
        birth_x.append(p.birth)
        birth_y.append(p.row)
        birth_hov.append(
            f"<b>{p.short_id}</b> — initial DAG run<br>"
            f"birth:    {_fmt_dt(p.birth)}<br>"
            f"DAG done: {_fmt_dt(p.first_dag_done)}<br>"
            f"wall:     {dur:.2f}s<br>"
            f"parents:  {', '.join(p.parents) or '—'}<br>"
            f"stages executed: {stage_count}<br>"
            f"stages cached:   {cached_count}"
        )
    _ticks("birth", birth_x, birth_y, birth_hov, color="#0969da")
    _ticks("refresh flip", refresh_x, refresh_y, refresh_text, color="#6d4f8f")

    # Decision events as narrow Bar traces — same Plotly y-positioning machinery
    # as the DAG span / stage exec bars, so they sit exactly on the program's row
    # at every zoom level (scatter markers float at fixed pixel height; bars don't).
    def _decision_bars(name, triples, color):
        if not triples:
            return
        widths, bases, rows, hov = [], [], [], []
        for x, y, h in triples:
            widths.append(MIN_BAR_VISUAL_MS)
            bases.append(x)
            rows.append(y)
            hov.append(h)
        fig.add_trace(
            go.Bar(
                name=name,
                orientation="h",
                x=widths,
                base=bases,
                y=rows,
                marker=dict(color=color, line=dict(color=color, width=0.5)),
                hovertext=hov,
                hoverinfo="text",
                width=BAR_HEIGHT,
                textposition="none",
            )
        )

    acc_pts = [
        (
            p.accepted,
            p.row,
            f"<b>{p.short_id}</b> — ACCEPTED<br>at {_fmt_dt(p.accepted)}<br>"
            f"refreshed {len(p.refreshed_at)}× before ingest",
        )
        for p in ordered
        if p.accepted
    ]
    _decision_bars("accepted", acc_pts, "#1a7f37")

    rej_pts = [
        (
            p.rejected,
            p.row,
            f"<b>{p.short_id}</b> — REJECTED<br>at {_fmt_dt(p.rejected)}<br>"
            f"refreshed {len(p.refreshed_at)}× before ingest",
        )
        for p in ordered
        if p.rejected
    ]
    _decision_bars("rejected", rej_pts, "#a40e26")

    fig.update_layout(
        title=None,
        barmode="overlay",
        bargap=0.0,
        bargroupgap=0.0,
        font=dict(
            family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            size=14,
            color="#24292f",
        ),
        xaxis=dict(
            title=dict(text="wall time (UTC)", font=dict(size=13, color="#57606a")),
            type="date",
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor="#fafbfc",
                bordercolor="#d0d7de",
            ),
            tickformat="%H:%M:%S",
            showgrid=True,
            gridcolor="#eaeef2",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=13, color="#57606a"),
            linecolor="#d0d7de",
            showline=True,
            mirror=True,
        ),
        yaxis=_yaxis_layout(n, y_labels, last_n),
        height=max(560, 30 * _visible_rows(n, last_n) + 220),
        width=1480,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hoverlabel=dict(
            bgcolor="#1f2328",
            bordercolor="#1f2328",
            font=dict(
                family="ui-monospace, SFMono-Regular, Menlo, monospace",
                size=13,
                color="#f6f8fa",
            ),
            namelength=-1,
            align="left",
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=1.06,
            xanchor="left",
            yanchor="bottom",
            bgcolor="#ffffff",
            bordercolor="#d0d7de",
            borderwidth=1,
            font=dict(size=13, color="#24292f"),
            itemwidth=30,
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=140, r=90, t=80, b=80),
        modebar=dict(
            orientation="h",
            bgcolor="#ffffff",
            color="#57606a",
            activecolor="#24292f",
        ),
        # No on-bar text labels (see issues #230/#231): stage identity is
        # carried by color + hover, so the uniformtext "hide below minsize"
        # safety net is unnecessary and previously caused large-box labels
        # to vanish.
        hovermode="closest",
    )

    return fig


def _utilization_html(u: UtilizationReport) -> str:
    """Efficiency stat-bar + archetype table HTML."""
    if u.total_llm_s == 0 and u.total_exec_s == 0:
        return ""
    eff_pct = u.overlap_efficiency * 100
    # Color the efficiency cell: red <30, amber <60, green otherwise.
    if eff_pct < 30:
        eff_color = "#a40e26"
    elif eff_pct < 60:
        eff_color = "#9a6700"
    else:
        eff_color = "#1a7f37"
    stats = [
        ("LLM wall", f"{u.total_llm_s:.0f}s"),
        ("exec wall", f"{u.total_exec_s:.0f}s"),
        ("overlap", f"{u.overlap_s:.0f}s"),
        ("LLM-only", f"{u.llm_only_s:.0f}s"),
        ("exec-only", f"{u.exec_only_s:.0f}s"),
        ("peak DAGs", f"{u.peak_concurrent_dags}"),
        ("LLM events", f"{u.llm_event_count}"),
        ("LLM failures", f"{u.llm_failure_count}"),
    ]
    eff_cell = (
        f'<div class="stat stat--accent" style="background:{eff_color};color:#fff">'
        f'<span class="k" style="color:#fff;opacity:0.85">overlap efficiency</span>'
        f'<span class="v" style="color:#fff">{eff_pct:.0f}%</span></div>'
    )
    cells = eff_cell + "".join(
        f'<div class="stat"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in stats
    )
    arche_block = ""
    if u.archetype_counts:
        ranked = sorted(
            u.archetype_counts.items(),
            key=lambda kv: kv[1]["accepted"] + kv[1]["rejected"] + kv[1]["other"],
            reverse=True,
        )
        rows = "".join(
            f"<tr><td>{name}</td>"
            f"<td class='num'>{row['accepted']}</td>"
            f"<td class='num'>{row['rejected']}</td>"
            f"<td class='num'>{row['other']}</td>"
            f"<td class='num'>{row['accepted'] + row['rejected'] + row['other']}</td>"
            "</tr>"
            for name, row in ranked
        )
        arche_block = f"""
<div class="atable">
  <div class="atable-title">mutation archetypes</div>
  <table>
    <thead><tr>
      <th>archetype</th><th class='num'>accepted</th>
      <th class='num'>rejected</th><th class='num'>other</th>
      <th class='num'>total</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""
    interpretation = (
        f"overlap efficiency {eff_pct:.0f}% — when both LLM and program-exec "
        "stages have work, this is the fraction of time they run concurrently. "
        "100% means perfectly pipelined; 0% means fully serialized."
    )
    return f"""
<div class="statbar statbar--util">
  {cells}
</div>
<div class="note">{interpretation}</div>
{arche_block}
"""


def _saturation_html(s: SaturationReport) -> str:
    """Saturation stat-bar HTML, sits next to the utilization bar."""
    if s.sample_count == 0:
        return ""

    def _color_for(pct: float) -> str:
        # Green = "cap actually saturated, working as designed".
        # Amber = "partly there, headroom exists".
        # Red   = "cap is NOT the bottleneck — look upstream".
        if pct >= 80.0:
            return "#1a7f37"
        if pct >= 40.0:
            return "#9a6700"
        return "#a40e26"

    prod_color = _color_for(s.producer_saturation_pct)
    buf_color = _color_for(s.buffer_saturation_pct)
    prod_cell = (
        f'<div class="stat stat--accent" '
        f'style="background:{prod_color};color:#fff">'
        f'<span class="k" style="color:#fff;opacity:0.85">'
        f"producer @ cap</span>"
        f'<span class="v" style="color:#fff">'
        f"{s.producer_saturation_pct:.0f}%</span></div>"
    )
    buf_cell = (
        f'<div class="stat stat--accent" '
        f'style="background:{buf_color};color:#fff">'
        f'<span class="k" style="color:#fff;opacity:0.85">'
        f"buffer @ cap</span>"
        f'<span class="v" style="color:#fff">'
        f"{s.buffer_saturation_pct:.0f}%</span></div>"
    )
    stats = [
        ("max_in_flight", f"{s.max_in_flight}"),
        ("samples", f"{s.sample_count}"),
        ("peak producer", f"{s.peak_producer_held}/{s.max_in_flight}"),
        ("peak buffer", f"{s.peak_buffer_held}/{s.max_in_flight}"),
        ("peak LLM/DAG", f"{s.peak_llm_active}/{s.peak_dag_active}"),
    ]
    cells = (
        prod_cell
        + buf_cell
        + "".join(
            f'<div class="stat"><span class="k">{k}</span>'
            f'<span class="v">{v}</span></div>'
            for k, v in stats
        )
    )
    if s.producer_saturation_pct < 50.0 and s.buffer_saturation_pct < 50.0:
        interp = (
            "max_in_flight is NOT the bottleneck — both sides spend most "
            "samples below the cap. Look upstream (parent selection, "
            "refresh latency, LLM rate)."
        )
    elif s.producer_saturation_pct >= 80.0 and s.buffer_saturation_pct < 50.0:
        interp = (
            "Producer slots saturate but buffer drains — ingestion keeps up; "
            "LLM/refresh is the load-bearing side."
        )
    elif s.buffer_saturation_pct >= 80.0 and s.producer_saturation_pct < 50.0:
        interp = (
            "Buffer saturates while producer has headroom — ingestion is "
            "the downstream bottleneck."
        )
    else:
        interp = (
            "Both sides hover near the cap — pipeline is roughly balanced "
            "around max_in_flight."
        )
    return f"""
<div class="statbar statbar--util">
  {cells}
</div>
<div class="note">{interp}</div>
"""


def _backpressure_timeseries_html(
    samples: list[BackpressureSampleEvent], div_id: str
) -> str:
    """Render a small Plotly time-series of held counts vs cap.

    Four lines: producer_held, buffer_held, in_flight, and llm_active, with
    a horizontal max_in_flight reference. The llm_active line is a SUBSET of
    producer_held — the gap between them is DAG-eval occupancy. Sub-second
    loop_interval means this can render a few hundred points per minute —
    fine for short runs; long runs are downsampled to ~2000 points before
    plotting so the embedded HTML stays compact.
    """
    if not samples:
        return ""
    import plotly.graph_objects as go

    DOWNSAMPLE_TARGET = 2000
    step = max(1, len(samples) // DOWNSAMPLE_TARGET)
    ds = samples[::step]
    xs = [s.timestamp for s in ds]
    prod = [s.producer_held for s in ds]
    buf = [s.buffer_held for s in ds]
    inflight = [s.in_flight for s in ds]
    llm = [s.llm_active for s in ds]
    # Use the LAST observed cap as the headline line so a mid-run config
    # change doesn't draw a flat misleading reference.
    cap = samples[-1].max_in_flight

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=prod,
            mode="lines",
            name="producer_held",
            line=dict(color="#0969da", width=1.5),
            hovertemplate="producer_held=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=llm,
            mode="lines",
            name="llm_active",
            line=dict(color="#fb8500", width=1.2),
            hovertemplate="llm_active=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=buf,
            mode="lines",
            name="buffer_held",
            line=dict(color="#6f42c1", width=1.5),
            hovertemplate="buffer_held=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=inflight,
            mode="lines",
            name="in_flight",
            line=dict(color="#1a7f37", width=1.0, dash="dot"),
            hovertemplate="in_flight=%{y}<extra></extra>",
        )
    )
    fig.add_hline(
        y=cap,
        line=dict(color="#a40e26", width=1.0, dash="dash"),
        annotation_text=f"max_in_flight={cap}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#a40e26"),
    )
    fig.update_layout(
        height=200,
        width=1480,
        margin=dict(l=140, r=90, t=20, b=30),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(
            family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
            size=12,
            color="#24292f",
        ),
        xaxis=dict(
            type="date",
            tickformat="%H:%M:%S",
            showgrid=True,
            gridcolor="#eaeef2",
            tickfont=dict(size=11, color="#57606a"),
            linecolor="#d0d7de",
            showline=True,
            mirror=True,
        ),
        yaxis=dict(
            title=dict(
                text="held",
                font=dict(size=12, color="#57606a"),
            ),
            rangemode="tozero",
            tickfont=dict(size=11, color="#57606a"),
            linecolor="#d0d7de",
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor="#eaeef2",
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=1.15,
            xanchor="left",
            yanchor="bottom",
            bgcolor="#ffffff",
            bordercolor="#d0d7de",
            borderwidth=1,
            font=dict(size=11, color="#24292f"),
        ),
        showlegend=True,
        hovermode="x unified",
    )
    inner = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=f"{div_id}-backpressure",
        config=dict(displaylogo=False, displayModeBar=False),
    )
    return f"""
<div class="atable">
  <div class="atable-title">backpressure (producer / buffer / in_flight vs max_in_flight)</div>
  {inner}
</div>
"""


def _summary_html(programs: dict[str, Program], refreshes: list[datetime]) -> str:
    """Stat-bar HTML (sits above the plot in the rendered dashboard)."""
    ordered = assign_rows(programs)
    n = len(ordered)
    n_seeds = sum(1 for p in ordered if p.birth is None)
    n_accepted = sum(1 for p in ordered if p.birth is not None and p.accepted)
    n_rej = sum(1 for p in ordered if p.birth is not None and p.rejected)
    n_other = n - n_seeds - n_accepted - n_rej

    pairs, no_start, no_done = pair_refreshes(ordered)
    queue_waits = [pp.queue_s for pp in pairs if pp.queue_s is not None]
    exec_times = [pp.exec_s for pp in pairs if pp.exec_s is not None]
    avg_q = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0
    avg_e = sum(exec_times) / len(exec_times) if exec_times else 0.0
    max_q = max(queue_waits) if queue_waits else 0.0
    max_e = max(exec_times) if exec_times else 0.0

    stats = [
        ("programs", f"{n}"),
        ("seeds", f"{n_seeds}"),
        ("accepted", f"{n_accepted}"),
        ("rejected", f"{n_rej}"),
        ("in-flight", f"{n_other}"),
        ("refresh flips", f"{len(refreshes)}"),
        ("re-eval pairs", f"{len(exec_times)} / {len(refreshes)}"),
        ("queue wait avg/max", f"{avg_q:.2f}s / {max_q:.2f}s"),
        ("exec avg/max", f"{avg_e * 1000:.0f}ms / {max_e * 1000:.0f}ms"),
        (
            "queue share",
            f"{(avg_q / (avg_q + avg_e) * 100) if (avg_q + avg_e) else 0:.0f}%",
        ),
    ]
    cells = "".join(
        f'<div class="stat"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in stats
    )
    if no_start or no_done:
        tail = (
            f" {no_start} refresh flip(s) had no later DAG start "
            f"(run ended before re-eval was scheduled); {no_done} "
            f"had a start but no completion (cancelled mid-eval)."
        )
    else:
        tail = " every refresh flip paired cleanly to one DAG re-run."
    return f"""
<div class="statbar">
  {cells}
</div>
<div class="note">
  re-eval exec ≈ {avg_e * 1000:.0f}ms (cached); every multi-second refresh delay is
  queue wait for a runner slot, not compute.{tail}
</div>
"""


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --fg:       #24292f;
    --fg-muted: #57606a;
    --fg-faint: #8c959f;
    --bg:       #ffffff;
    --bg-alt:   #f6f8fa;
    --border:   #d0d7de;
    --border-l: #eaeef2;
    --mono:     "ui-monospace", "SFMono-Regular", "Menlo", "Consolas", monospace;
  }}
  body {{
    margin: 0; background: var(--bg-alt); color: var(--fg);
    font-family: var(--mono); font-size: 14px; line-height: 1.5;
  }}
  .hdr {{
    background: var(--bg); border-bottom: 1px solid var(--border);
    padding: 10px 18px; display: flex; align-items: baseline; gap: 22px;
  }}
  .hdr h1 {{ margin: 0; font-size: 16px; font-weight: 600; color: var(--fg); }}
  .wrap {{ max-width: 1500px; margin: 0 auto; padding: 10px 18px 18px; }}
  .statbar {{
    display: flex; flex-wrap: wrap; gap: 0;
    border: 1px solid var(--border); background: var(--bg);
    margin: 10px 0;
  }}
  .stat {{
    display: flex; flex-direction: column; padding: 8px 14px;
    border-right: 1px solid var(--border-l); min-width: 130px;
  }}
  .stat:last-child {{ border-right: 0; }}
  .stat .k {{
    font-size: 12px; color: var(--fg-muted); text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .stat .v {{
    font-size: 15px; color: var(--fg); font-weight: 600; margin-top: 3px;
  }}
  .statbar--util .stat {{ min-width: 110px; }}
  .stat--accent {{ min-width: 150px !important; }}
  .atable {{
    border: 1px solid var(--border); background: var(--bg);
    margin: 10px 0 0; padding: 0;
  }}
  .atable-title {{
    font-size: 12px; color: var(--fg-muted); text-transform: uppercase;
    letter-spacing: 0.5px; padding: 8px 14px;
    border-bottom: 1px solid var(--border-l);
  }}
  .atable table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .atable th, .atable td {{
    padding: 6px 14px; text-align: left;
    border-bottom: 1px solid var(--border-l);
  }}
  .atable th {{ color: var(--fg-muted); font-weight: 600; font-size: 12px; }}
  .atable td.num, .atable th.num {{
    text-align: right; font-variant-numeric: tabular-nums;
  }}
  .atable tr:last-child td {{ border-bottom: 0; }}
  .note {{
    font-size: 13px; color: var(--fg-muted); padding: 8px 0 12px;
    font-style: italic;
  }}
  .toolbar {{
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    border: 1px solid var(--border); background: var(--bg);
    padding: 6px 10px; margin-bottom: 0; border-bottom: 0;
    font-size: 13px;
  }}
  .toolbar button {{
    font-family: var(--mono); font-size: 13px;
    background: var(--bg-alt); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 4px 12px; cursor: pointer;
  }}
  .toolbar button:hover {{
    background: var(--bg); border-color: var(--fg-muted);
  }}
  .toolbar .sep {{
    width: 1px; align-self: stretch; background: var(--border-l);
    margin: 0 2px;
  }}
  .toolbar .group {{
    display: inline-flex; align-items: center; gap: 8px;
    color: var(--fg-muted); font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .toolbar .group .lbl {{ color: var(--fg-muted); }}
  .toolbar label.cat {{
    display: inline-flex; align-items: center; gap: 5px;
    color: var(--fg); font-size: 13px; text-transform: none;
    letter-spacing: 0; cursor: pointer; user-select: none;
    padding: 2px 6px; border-radius: 3px;
  }}
  .toolbar label.cat:hover {{ background: var(--bg-alt); }}
  .toolbar label.cat input {{ margin: 0; cursor: pointer; }}
  .toolbar label.cat .swatch {{
    display: inline-block; width: 9px; height: 9px;
    border: 1px solid var(--border); border-radius: 2px;
  }}
  .toolbar .hint {{ color: var(--fg-muted); flex-basis: 100%; }}
  .chart {{
    background: var(--bg); border: 1px solid var(--border);
    padding: 4px;
  }}
  .footer {{
    font-size: 12px; color: var(--fg-faint); padding: 10px 0 4px;
  }}
  .footer .src {{
    color: var(--fg-muted); font-variant-numeric: tabular-nums;
  }}
</style>
</head><body>
<div class="hdr">
  <h1>{title}</h1>
</div>
<div class="wrap">
{summary}
{utilization}
{saturation}
<div class="toolbar">
  <button id="reset-view-btn" type="button" title="Reset zoom to the initial last-N window">⟲ reset view</button>
  <span class="sep"></span>
  <span class="group">
    <span class="lbl">rows</span>
    <button data-rows="50"  type="button" title="Show the last 50 programs">last 50</button>
    <button data-rows="100" type="button" title="Show the last 100 programs">last 100</button>
    <button data-rows="200" type="button" title="Show the last 200 programs">last 200</button>
    <button data-rows="500" type="button" title="Show the last 500 programs">last 500</button>
    <button data-rows="all" type="button" title="Show every program">all ({total_n})</button>
  </span>
  <span class="sep"></span>
  <span class="group">
    <span class="lbl">events</span>
    <label class="cat" title="DAG span backdrop, per-stage execution bars, cached-stage ticks, and birth markers">
      <input type="checkbox" data-cat="lifecycle" checked>
      <span class="swatch" style="background:#eef1f4;border-color:#b6bcc4"></span>
      lifecycle
    </label>
    <label class="cat" title="Refresh-flip ticks, re-eval queue-wait band, re-eval exec band">
      <input type="checkbox" data-cat="reeval" checked>
      <span class="swatch" style="background:#6f42c1;border-color:#4b2a85"></span>
      re-evaluation
    </label>
    <label class="cat" title="Ingestor accepted / rejected decision lines">
      <input type="checkbox" data-cat="outcome" checked>
      <span class="swatch" style="background:#1a7f37;border-color:#1a7f37"></span>
      ingestor outcomes
    </label>
  </span>
  <span class="hint">
    drag plot = zoom · drag y-axis labels = pan rows · dblclick chart = reset ·
    range slider below = pan time · row buttons above clamp the y-window
  </span>
</div>
<div class="chart">
{plot_div}
</div>
{backpressure_plot}
<div class="footer">
  <span class="src">source: {subtitle}</span>
  <br>
  hover any segment for full timing detail · all bars clamped to ≥
  {min_bar_s:.2f}s visual width so sub-pixel events stay visible at full
  zoom-out (real durations in hover).
  <br>
  <b>About long re-eval queue waits:</b> the grey ribbon between a refresh
  flip and the purple exec bar is wall time the parent spent in QUEUED waiting
  for a runner slot, NOT compute. The producing mutant task is blocked on
  <code>ParentRefresher._await_done()</code> the whole time, pinning an
  in-flight slot. With N concurrent mutants × M-second refreshes, throughput
  collapses even though the per-DAG exec is cached and near-zero.
  LineageStage running uncached on re-eval amplifies M for every refresh.
</div>
</div>
<script>
  (function () {{
    var plot = document.getElementById("{div_id}");
    if (!plot || !window.Plotly) return;

    // Injected by render_full_html — total program count and the initial
    // last-N window. We use the highest row indices (newest programs by
    // birth) as the "last N" since assign_rows orders strictly by birth.
    var TOTAL_N = {total_n};
    var INITIAL_LAST_N = {initial_last_n};

    // Reversed y-axis: range is [bottom_value, top_value]. Bottom = newest
    // row (TOTAL_N - 0.5), top = TOTAL_N - k - 0.5 for "last k rows".
    function rangeForLastK(k) {{
      if (!TOTAL_N) return null;
      if (k === "all" || k >= TOTAL_N) {{
        return [TOTAL_N - 0.5, -0.5];
      }}
      return [TOTAL_N - 0.5, TOTAL_N - k - 0.5];
    }}

    var INITIAL_Y_RANGE = rangeForLastK(INITIAL_LAST_N);

    var btn = document.getElementById("reset-view-btn");
    if (btn) {{
      btn.addEventListener("click", function () {{
        var update = {{ "xaxis.autorange": true }};
        if (INITIAL_Y_RANGE) {{
          update["yaxis.autorange"] = false;
          update["yaxis.range"] = INITIAL_Y_RANGE;
        }} else {{
          update["yaxis.autorange"] = "reversed";
        }}
        Plotly.relayout(plot, update);
      }});
    }}

    document.querySelectorAll(".toolbar button[data-rows]")
      .forEach(function (b) {{
        b.addEventListener("click", function () {{
          var raw = b.dataset.rows;
          var k = raw === "all" ? "all" : parseInt(raw, 10);
          var rng = rangeForLastK(k);
          if (!rng) return;
          Plotly.relayout(plot, {{
            "yaxis.autorange": false,
            "yaxis.range": rng,
          }});
        }});
      }});

    var CAT_TRACES = {{
      "lifecycle": ["DAG span", "stage exec", "cached skip", "birth"],
      "reeval":    ["re-eval queue wait", "refresh flip"],
      "outcome":   ["accepted", "rejected"],
    }};

    function applyCategory(cat, on) {{
      var names = CAT_TRACES[cat] || [];
      var indices = [];
      (plot.data || []).forEach(function (t, i) {{
        if (names.indexOf(t.name) !== -1) indices.push(i);
      }});
      if (!indices.length) return;
      Plotly.restyle(plot, {{ "visible": on ? true : "legendonly" }}, indices);
    }}

    document.querySelectorAll(".toolbar input[type=checkbox][data-cat]")
      .forEach(function (cb) {{
        cb.addEventListener("change", function () {{
          applyCategory(cb.dataset.cat, cb.checked);
        }});
      }});
  }})();
</script>
</body></html>
"""


def render_full_html(
    programs: dict[str, Program],
    refreshes: list[datetime],
    title: str = "flow profile",
    subtitle: str = "",
    div_id: str = "gigaevo-flow",
    utilization: UtilizationReport | None = None,
    backpressure: list[BackpressureSampleEvent] | None = None,
    saturation: SaturationReport | None = None,
    *,
    last_n: int | None = DEFAULT_LAST_N_ROWS,
) -> str:
    """Return a standalone HTML document with Plotly.js inlined.

    Plotly is embedded directly (not loaded from a CDN) so the file
    renders inside sandboxed previews (VS Code HTML preview, archived
    artifacts, offline shares) without needing network access.

    ``last_n`` sets the initial y-axis window — only the most recent
    ``last_n`` programs are visible on open. The toolbar exposes quick
    buttons to widen the window or show all rows. Pass ``last_n=None``
    to disable clipping entirely.
    """
    fig = make_figure(programs, refreshes, last_n=last_n)
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="inline",
        div_id=div_id,
        config=dict(
            displaylogo=False,
            displayModeBar=True,
            modeBarButtonsToRemove=[
                "select2d",
                "lasso2d",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
            toImageButtonOptions=dict(
                format="png",
                filename=div_id,
                height=1200,
                width=1800,
                scale=2,
            ),
        ),
    )
    util_block = _utilization_html(utilization) if utilization is not None else ""
    sat_block = _saturation_html(saturation) if saturation is not None else ""
    # Inject the backpressure time-series ABOVE the main plot so an operator
    # scanning the dashboard top-to-bottom hits "is the pipeline full?"
    # before drilling into per-program timelines. Plotly.js is already
    # embedded inline in plot_div above; the timeseries fragment reuses it.
    bp_block = (
        _backpressure_timeseries_html(backpressure, div_id) if backpressure else ""
    )
    total_n = len(programs)
    initial_last_n = last_n if (last_n is not None and total_n > last_n) else total_n
    return _HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        summary=_summary_html(programs, refreshes),
        utilization=util_block,
        saturation=sat_block,
        backpressure_plot=bp_block,
        plot_div=plot_div,
        min_bar_s=MIN_BAR_VISUAL_MS / 1000,
        div_id=div_id,
        total_n=total_n,
        initial_last_n=initial_last_n,
    )
