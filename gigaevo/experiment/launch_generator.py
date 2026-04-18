#!/usr/bin/env python3
"""Generate launch.sh from experiment.yaml manifest.

Called internally by ``gigaevo.experiment.launch.run_launch()``.
All values (servers, runs, config, custom_env) come from experiment.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

from gigaevo.experiment.manifest import load_manifest

# Derive project root from this file's location (still needed for the generated
# launch.sh to reference the repo root via $PROJ).
PROJ_PATH = str(Path(__file__).resolve().parent.parent.parent)

PYTHON_PATH = os.environ.get(
    "GIGAEVO_PYTHON", "python3"
)


def generate(experiment: str) -> str:
    m = load_manifest(experiment)
    exp_dir_rel = f"experiments/{experiment}"

    lines: list[str] = []

    # Header
    lines.append("#!/usr/bin/env bash")
    lines.append("# GENERATED from experiment.yaml — do not edit manually.")
    lines.append(f"# Regenerate: gigaevo -e {experiment} launch --dry-run")
    lines.append("#")
    lines.append(f"# Experiment: {m.contract.identity.name}")
    lines.append(f"# Branch: {m.contract.identity.branch}")
    if m.contract.identity.pr_number:
        lines.append(f"# PR: #{m.contract.identity.pr_number}")
    if m.contract.identity.prereg_commit:
        lines.append(f"# Pre-reg commit: {m.contract.identity.prereg_commit}")
    lines.append("#")
    lines.append(f"# Runs: {', '.join(r.label for r in m.contract.runs)}")
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("")

    # Environment
    lines.append(f'PROJ="{PROJ_PATH}"')
    lines.append(f'PYTHON="{PYTHON_PATH}"')
    lines.append(f'LOG_DIR="$PROJ/{exp_dir_rel}"')
    lines.append("")
    lines.append('export GIGAEVO_PYTHON="$PYTHON"')
    lines.append("")

    # Custom env vars
    if m.contract.custom_env:
        lines.append(
            "# Task-specific environment variables (from experiment.yaml custom_env)"
        )
        for key, val in m.contract.custom_env.items():
            lines.append(f'export {key}="{val}"')
        lines.append("")

    # Banner
    lines.append(
        'echo "================================================================"'
    )
    lines.append(
        f"echo \"{m.contract.identity.name} experiment launch — $(date -u '+%Y-%m-%d %H:%M UTC')\""
    )
    if m.contract.identity.prereg_commit:
        lines.append(f'echo "Pre-reg commit: {m.contract.identity.prereg_commit}"')
    for run in m.contract.runs:
        lines.append(f'echo "{run.label}: {run.condition} — pipeline={run.pipeline}"')
    lines.append(
        'echo "================================================================"'
    )
    lines.append('echo ""')
    lines.append("")

    # Preflight is now run by `gigaevo launch` before exec — not in this script.

    # Config verification
    lines.append(
        "# ── Config verification (--cfg job) ───────────────────────────────────────"
    )
    for run in m.contract.runs:
        lines.append(f'echo "--- {run.label} config ---"')
        cfg_cmd = _build_run_cmd(run, m, cfg_only=True)
        lines.append('"$PYTHON" "$PROJ/run.py" \\')
        for i, param in enumerate(cfg_cmd):
            lines.append(f"    {param} \\")
        lines.append(f'    > "$LOG_DIR/cfg_run_{run.label}.txt" 2>&1')
        lines.append(f'cat "$LOG_DIR/cfg_run_{run.label}.txt" | head -40')
        lines.append('echo ""')
    lines.append("")

    lines.append(
        'echo "================================================================"'
    )
    lines.append('echo "Config verified."')
    lines.append('echo "Launching runs..."')
    lines.append('echo ""')
    lines.append("")

    # Launch from project root (Hydra resolves paths relative to CWD)
    lines.append(
        "# ── Launch from project root (Hydra resolves paths relative to CWD) ───────"
    )
    lines.append('cd "$PROJ"')
    lines.append("")

    # Launch runs
    lines.append(
        "# ── Launch runs ────────────────────────────────────────────────────────────"
    )
    pid_vars: list[str] = []
    for run in m.contract.runs:
        pid_var = f"PID_{run.label.replace('-', '_')}"
        pid_vars.append(pid_var)
        cmd_params = _build_run_cmd(run, m, cfg_only=False)

        lines.append(f"# ── Run {run.label}: {run.condition}")
        lines.append('nohup "$PYTHON" "$PROJ/run.py" \\')
        for i, param in enumerate(cmd_params):
            lines.append(f"    {param} \\")
        lines.append(
            f'    > "$LOG_DIR/{run.log_path or f"run_{run.label}.log"}" 2>&1 &'
        )
        lines.append(f"{pid_var}=$!")
        lines.append(
            f'echo "Run {run.label} started: PID=${pid_var}  DB={run.db}  '
            f'pipeline={run.pipeline}"'
        )
        lines.append("")

    # Summary
    lines.append('echo ""')
    lines.append(
        'echo "================================================================"'
    )
    lines.append(f'echo "All {len(m.contract.runs)} runs launched."')

    pid_echo = "  ".join(
        f"{r.label}=$PID_{r.label.replace('-', '_')}" for r in m.contract.runs
    )
    lines.append(f'echo "PIDs: {pid_echo}"')

    # Write PIDs to file
    pid_file_content = " ".join(
        f"$PID_{r.label.replace('-', '_')}" for r in m.contract.runs
    )
    lines.append(f'echo "{pid_file_content}" > "$LOG_DIR/pids.txt"')
    lines.append("")

    # Verify PIDs alive
    lines.append(
        "# ── Verify all PIDs alive ──────────────────────────────────────────────────"
    )
    lines.append("sleep 5")
    lines.append("ALL_ALIVE=true")
    for run in m.contract.runs:
        pv = f"PID_{run.label.replace('-', '_')}"
        lines.append(
            f'kill -0 ${pv} 2>/dev/null || {{ echo "DEAD: {run.label} (PID=${pv})"; ALL_ALIVE=false; }}'
        )
    lines.append('if [ "$ALL_ALIVE" = "false" ]; then')
    lines.append('    echo "ABORT: not all runs alive. Check logs."')
    lines.append("    exit 1")
    lines.append("fi")
    lines.append('echo "All PIDs verified alive."')
    lines.append("")

    # Write PIDs to experiment.yaml via helper script
    lines.append(
        "# ── Record PIDs in experiment.yaml ────────────────────────────────────────"
    )
    label_list = " ".join(r.label for r in m.contract.runs)
    lines.append(
        f"gigaevo -e {experiment} manifest record-pids"
        f' --pids-file "$LOG_DIR/pids.txt"'
        f' --labels "{label_list}"'
    )
    lines.append("")

    # Watchdog launch hint
    lines.append('echo ""')
    lines.append('echo "Launch watchdog:"')
    lines.append(f'echo "  nohup $PYTHON {exp_dir_rel}/run_watchdog.py \\\\"')
    lines.append(f'echo "      > {exp_dir_rel}/watchdog.log 2>&1 &"')
    lines.append(
        'echo "================================================================"'
    )

    return "\n".join(lines) + "\n"


def _format_hydra_value(value) -> str:
    """Render a Python value as a Hydra override RHS."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _build_run_cmd(run, manifest, *, cfg_only: bool) -> list[str]:
    """Build run.py command-line parameters for a run.

    Policy: every key in contract.config.extra is emitted as a Hydra
    override verbatim. The generator imposes no defaults — absent keys
    fall through to whatever the Hydra config hierarchy provides.
    Run-scoped params (problem, pipeline, db, model) and contract-level
    params (max_generations) are emitted explicitly from their respective
    homes; nothing is inferred.

    When ``contract.config.task_group`` is set, ``experiment=<task_group>``
    is emitted as the FIRST override. This swaps the ``experiment:`` slot
    in ``config/config.yaml``'s defaults list — Hydra composes the task
    group file (which inherits ``base``) and all subsequent CLI overrides
    (pipeline, extras, extra_overrides) win via Hydra's normal resolution.
    """
    params: list[str] = []

    task_group = manifest.contract.config.task_group
    if task_group:
        params.append(f"experiment={task_group}")

    # All Hydra overrides (including problem.name, pipeline, redis.db,
    # max_generations, model_name, llm_base_url) come from the manifest:
    # - contract.config.extra: shared across runs
    # - runs[].extra_overrides: per-run
    # Nothing hardcoded here — the manifest is the source of truth.
    for key, value in manifest.contract.config.extra.items():
        params.append(f"{key}={_format_hydra_value(value)}")

    # Extra per-run overrides from experiment.yaml (e.g. prompt_fetcher config)
    # Single-quote any override containing ${...} Hydra interpolation refs
    # to prevent bash from expanding them as shell variables (KF-02).
    if run.extra_overrides:
        for ov in run.extra_overrides:
            if "${" in ov:
                params.append(f"'{ov}'")
            else:
                params.append(ov)

    if cfg_only:
        params.append("--cfg job")

    return params
