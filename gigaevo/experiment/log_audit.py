"""Registry-backed canonical-event audit.

Parses a log file (or log text) for `[EVENT_NAME] {json-payload}` lines and
validates each payload against the corresponding Pydantic class in
`gigaevo.monitoring.events.CANONICAL_EVENTS`. The auditor holds no
event-specific invariants of its own — those live on the event classes as
Pydantic field/model validators.

For each registered event whose `expected_after_gen > 0`, the auditor also
reports if it never appeared in a log whose observed `gen` already crossed
that threshold (via `GENERATION_BOUNDARY` events).

Invoked via `gigaevo -e <experiment> events audit --log <log_file>`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from pydantic import ValidationError

# Trigger subclass registration.
import gigaevo  # noqa: F401
from gigaevo.monitoring.events import CANONICAL_EVENTS

EVENT_LINE_RE = re.compile(r"\[(?P<name>[A-Z_][A-Z0-9_]*)\]\s+(?P<json>\{.*\})")


@dataclass
class AuditReport:
    """Structured audit result.

    Attributes:
        failures: {event_name: [error messages]}. `_parse` bucket holds
            generic parse errors for lines that matched the bracket pattern
            but whose payload was not JSON.
        event_counts: {event_name: number of well-formed events seen}.
        missing_after_gen: {event_name: expected_after_gen} for events that
            should have appeared by the latest observed generation boundary
            but never did.
    """

    failures: dict[str, list[str]] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    missing_after_gen: dict[str, int] = field(default_factory=dict)


def _record_failure(report: AuditReport, name: str, msg: str) -> None:
    report.failures.setdefault(name, []).append(msg)


def _bump_count(report: AuditReport, name: str) -> None:
    report.event_counts[name] = report.event_counts.get(name, 0) + 1


def audit_log_text(log_text: str) -> AuditReport:
    """Audit canonical events in a log string.

    Walks each line, extracting the first `[EVENT_NAME] {json}` token.
    Lines that do not match are ignored (launcher banners, loguru noise).

    For matching lines:
    - Unknown event names are recorded under `failures[name]`.
    - Non-JSON payloads are recorded under `failures['_parse']`.
    - Pydantic validation errors are recorded under `failures[name]`.

    Missing-by-gen: after processing, every registered event with
    `expected_after_gen > 0` is checked against the maximum observed `gen`
    from `GENERATION_BOUNDARY` events. Events that never appeared despite
    the run having progressed past their expected generation are reported
    via `missing_after_gen`.
    """
    report = AuditReport()
    max_gen_observed = -1
    saw_generation_boundary = False

    for line_no, raw_line in enumerate(log_text.splitlines(), start=1):
        match = EVENT_LINE_RE.search(raw_line)
        if not match:
            # Also handle the case where the bracketed name matches but the
            # payload isn't JSON — the regex above requires `{...}` and won't
            # match `[EVENT] not-json`. So we check that separately.
            bracket_only = re.search(
                r"\[(?P<name>[A-Z_][A-Z0-9_]*)\](?P<rest>.*)$", raw_line
            )
            if bracket_only is None:
                continue
            name = bracket_only.group("name")
            rest = bracket_only.group("rest").strip()
            # Only flag if the bracketed token looks like a canonical event
            # and the payload starts with something other than valid JSON.
            if name in CANONICAL_EVENTS and rest and not rest.startswith("{"):
                _record_failure(
                    report,
                    "_parse",
                    f"Line {line_no}: [{name}] payload is not JSON: {rest[:80]!r}",
                )
            continue

        name = match.group("name")
        payload_text = match.group("json")

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            _record_failure(
                report,
                name if name in CANONICAL_EVENTS else "_parse",
                f"Line {line_no}: JSON parse error: {e}",
            )
            continue

        cls = CANONICAL_EVENTS.get(name)
        if cls is None:
            _record_failure(
                report,
                name,
                f"Line {line_no}: unknown canonical event name {name!r}",
            )
            continue

        # Pydantic does not need the `event` key; it's a ClassVar. Drop it if
        # present — the subclass defines it, so passing it in as a field
        # would raise `extra='forbid'`.
        payload.pop("event", None)

        try:
            instance = cls(**payload)
        except ValidationError as e:
            # Compact error listing: one short message per missing/invalid field.
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
                msg = err.get("msg", "validation error")
                _record_failure(
                    report,
                    name,
                    f"Line {line_no}: {loc}: {msg}",
                )
            continue
        except TypeError as e:
            _record_failure(report, name, f"Line {line_no}: constructor error: {e}")
            continue

        _bump_count(report, name)
        if name == "GENERATION_BOUNDARY":
            saw_generation_boundary = True
            gen = getattr(instance, "gen", None)
            if isinstance(gen, int) and gen > max_gen_observed:
                max_gen_observed = gen

    # Missing-by-gen check — only meaningful if we saw at least one
    # generation boundary.
    if saw_generation_boundary:
        for event_name, cls in CANONICAL_EVENTS.items():
            expected = int(getattr(cls, "expected_after_gen", 0) or 0)
            if expected <= 0:
                continue
            if max_gen_observed < expected:
                continue
            if report.event_counts.get(event_name, 0) == 0:
                report.missing_after_gen[event_name] = expected

    return report


def render_report(report: AuditReport, exp_name: str) -> str:
    """Render an audit report as a human-readable markdown string."""
    lines = [f"# Log Audit Report: {exp_name}", ""]

    if report.event_counts:
        lines.append("## Event Distribution")
        lines.append("")
        for name in sorted(report.event_counts):
            mark = "✗" if name in report.failures else "✓"
            lines.append(f"- {mark} {name}: {report.event_counts[name]}")
        lines.append("")

    if report.missing_after_gen:
        lines.append("## Missing-by-Generation")
        lines.append("")
        for name in sorted(report.missing_after_gen):
            expected = report.missing_after_gen[name]
            cls = CANONICAL_EVENTS.get(name)
            q = getattr(cls, "health_question", "") if cls else ""
            suffix = f" — {q}" if q else ""
            lines.append(
                f"- {name} expected after gen {expected}, never observed{suffix}"
            )
        lines.append("")

    if report.failures:
        lines.append("## Audit Failures")
        lines.append("")
        for name in sorted(report.failures):
            lines.append(f"### {name}")
            lines.append("")
            for msg in report.failures[name]:
                lines.append(f"- {msg}")
            lines.append("")

    if not report.failures and not report.missing_after_gen:
        lines.append("**Audit result: PASSED**")
    else:
        lines.append("**Audit result: FAILED**")

    return "\n".join(lines)
