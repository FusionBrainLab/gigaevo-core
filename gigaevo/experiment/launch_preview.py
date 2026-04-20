"""Render ``LAUNCH_PREVIEW.md`` from a ``DryRunResult`` — Phase 5 of the
Config-Override Integrity Pipeline. See .claude/plans/humble-weaving-shamir.md.

Public surface:
    - ``write_launch_preview(experiment, result)`` — write preview, return path

The preview is a markdown artifact committed to ``experiments/<exp>/``. It
shows, per run, every pinned path's resolved value and the declared-source
provenance (``extra_overrides`` > ``config.shared_overrides`` > ``task_group`` file >
Hydra default). Reviewers (human and AI) read the preview at launch Gate 2
and confirm every pinned row is ✓ and non-pinned overrides look right.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import yaml

from gigaevo.experiment import manifest as _manifest_mod
from gigaevo.experiment.dry_run import DryRunResult
from gigaevo.experiment.manifest import experiment_dir, load_manifest

_MISSING = object()


def write_launch_preview(experiment: str, result: DryRunResult) -> Path:
    """Write LAUNCH_PREVIEW.md for ``experiment`` and return its path."""
    manifest = load_manifest(experiment)
    out_path = experiment_dir(experiment) / "LAUNCH_PREVIEW.md"
    text = _render(experiment, manifest, result)
    out_path.write_text(text)
    return out_path


def _render(experiment: str, manifest, result: DryRunResult) -> str:
    contract_pins = dict(manifest.contract.config.pinned or {})
    task_group = manifest.contract.config.task_group
    task_group_values = _load_task_group_values(task_group)

    lines: list[str] = []
    lines.append(f"# Launch Preview — {experiment}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(UTC).isoformat()}")
    if task_group:
        lines.append(
            f"**Task group:** {task_group}  (config/experiment/{task_group}.yaml)"
        )
    else:
        lines.append("**Task group:** _none_")

    total_pins = 0
    total_failed = 0
    per_run_blocks: list[str] = []
    for run in manifest.contract.runs:
        resolved = result.resolved.get(run.label, {})
        cli_args = result.cli_args.get(run.label, [])
        declared_overrides = _parse_declared_overrides(cli_args, run.extra_overrides)
        run_pins = {**contract_pins, **(run.pinned or {})}

        rows: list[tuple[str, dict[str, Any]]] = []
        seen_paths: set[str] = set()
        for path in run_pins:
            rows.append(
                (
                    path,
                    _build_row(
                        path,
                        resolved,
                        manifest,
                        run,
                        task_group_values,
                        declared_overrides,
                        run_pins,
                    ),
                )
            )
            seen_paths.add(path)
        # Include non-pinned declared overrides for visibility
        for path in declared_overrides:
            if path in seen_paths:
                continue
            rows.append(
                (
                    path,
                    _build_row(
                        path,
                        resolved,
                        manifest,
                        run,
                        task_group_values,
                        declared_overrides,
                        run_pins,
                    ),
                )
            )
            seen_paths.add(path)

        matches = sum(
            1
            for _, r in rows
            if r.get("pinned") is not _MISSING and r.get("match") is True
        )
        fails = sum(
            1
            for _, r in rows
            if r.get("pinned") is not _MISSING and r.get("match") is False
        )
        total_pins += matches + fails
        total_failed += fails

        block: list[str] = []
        block.append("")
        block.append(f"### Run {run.label} (db={run.db})")
        block.append("")
        block.append(
            "| Path | Task group | shared_overrides | extra_overrides | Resolved | Pinned? | Match |"
        )
        block.append(
            "|------|------------|--------------|-----------------|----------|---------|-------|"
        )
        for path, row in rows:
            block.append(
                "| {path} | {tg} | {ex} | {ov} | {res} | {pinned} | {match} |".format(
                    path=path,
                    tg=_fmt(row["task_group"]),
                    ex=_fmt(row["shared_overrides"]),
                    ov=_fmt(row["extra_override"]),
                    res=_fmt(row["resolved"]),
                    pinned=_fmt(row["pinned"])
                    if row["pinned"] is not _MISSING
                    else "—",
                    match=_match_cell(row),
                )
            )
        per_run_blocks.append("\n".join(block))

    status = "PASS" if total_failed == 0 else "FAIL"
    lines.append(
        f"**Status:** {status} ({total_pins} pin assertion(s), {total_failed} failed)"
    )
    lines.append("")
    lines.append("## Per-Run Resolved Config")
    lines.extend(per_run_blocks)

    lines.append("")
    lines.append("## Hydra Default Fingerprint")
    lines.append("")
    lines.append("| File | sha256 |")
    lines.append("|------|--------|")
    for path, digest in sorted(result.fingerprint.items()):
        lines.append(f"| {path} | `{digest[:12]}…` |")
    lines.append("")
    lines.append(
        "On re-launch any drift in these digests causes preflight to CRITICAL-fail."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def _build_row(
    path: str,
    resolved: dict,
    manifest,
    run,
    task_group_values: dict,
    declared_overrides: dict,
    run_pins: dict,
) -> dict[str, Any]:
    resolved_val = _lookup_dotted(resolved, path)
    shared = manifest.contract.config.shared_overrides or {}
    shared_val = shared.get(path, _MISSING)
    extra_override_val = declared_overrides.get(path, _MISSING)
    task_group_val = task_group_values.get(path, _MISSING)
    pinned_val = run_pins.get(path, _MISSING)

    match: Any
    if pinned_val is _MISSING:
        match = None
    else:
        match = resolved_val is not _MISSING and _values_equal(resolved_val, pinned_val)

    return {
        "resolved": resolved_val,
        "shared_overrides": shared_val,
        "extra_override": extra_override_val,
        "task_group": task_group_val,
        "pinned": pinned_val,
        "match": match,
    }


def _parse_declared_overrides(cli_args: list[str], extra_overrides) -> dict[str, Any]:
    """Pull ``key=value`` pairs from extra_overrides so rows can show provenance.

    ``extra_overrides`` is the authoritative per-run source (it is what the
    manifest declared). We parse both it and the reconstructed cli_args
    fallback for robustness — cli_args reflects what actually hit run.py.
    """
    out: dict[str, Any] = {}
    for ov in extra_overrides or []:
        ov = ov.strip().strip("'\"")
        if "=" in ov and not ov.startswith("experiment="):
            k, _, v = ov.partition("=")
            out[k.strip()] = _coerce_scalar(v.strip())
    for a in cli_args:
        a = a.strip()
        if not a or a.startswith("--") or a.startswith("experiment="):
            continue
        if "=" in a:
            k, _, v = a.partition("=")
            k = k.strip()
            if k in out:
                continue  # extra_overrides wins for display
            out[k] = _coerce_scalar(v.strip())
    return out


def _coerce_scalar(raw: str) -> Any:
    """Best-effort string → scalar for display. Leaves unquoted strings as-is."""
    raw = raw.strip().strip('"').strip("'")
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _load_task_group_values(task_group: str | None) -> dict[str, Any]:
    """Return a flat dotted-path dict of values in the task-group YAML.

    Empty dict when no task_group set or file missing. Comments/defaults
    block are ignored — only scalar overrides are reported as provenance.
    """
    if not task_group:
        return {}
    proj = _manifest_mod.PROJ
    tg_path = proj / "config" / "experiment" / f"{task_group}.yaml"
    if not tg_path.exists():
        return {}
    try:
        cfg = OmegaConf.to_container(OmegaConf.load(tg_path), resolve=False)
    except Exception:
        try:
            cfg = yaml.safe_load(tg_path.read_text())
        except Exception:
            return {}
    if not isinstance(cfg, dict):
        return {}
    cfg.pop("defaults", None)
    return _flatten(cfg)


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        dotted = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{dotted}."))
        else:
            out[dotted] = v
    return out


def _lookup_dotted(doc, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _values_equal(a, b) -> bool:
    if type(a) is type(b):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _fmt(v: Any) -> str:
    if v is _MISSING:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def _match_cell(row: dict) -> str:
    if row["match"] is None:
        return "—"
    return "PASS ✓" if row["match"] else "FAIL ✗"
