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

    # Collect all server IPs for NO_PROXY
    no_proxy_hosts = ["localhost", "127.0.0.1", "api.github.com"] + m.contract.servers
    no_proxy = ",".join(no_proxy_hosts)

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
    lines.append(f'export NO_PROXY="{no_proxy}"')
    lines.append('export no_proxy="$NO_PROXY"')
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

        # Env prefix for per-run chain URL (only if run has a specific URL;
        # skip if null — the global export from custom_env handles shared LB)
        env_prefix = ""
        if run.chain_url:
            chain_url_env_var = m.contract.config.extra.get(
                "chain_url_env_var", "CHAIN_URL"
            )
            env_prefix = f'{chain_url_env_var}="{run.chain_url}" '

        lines.append(f"# ── Run {run.label}: {run.condition}")
        lines.append(f'{env_prefix}nohup "$PYTHON" "$PROJ/run.py" \\')
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
    lines.append('echo "  NO_PROXY=\\"$NO_PROXY\\" no_proxy=\\"$NO_PROXY\\" \\\\"')
    lines.append(f'echo "  nohup $PYTHON {exp_dir_rel}/run_watchdog.py \\\\"')
    lines.append(f'echo "      > {exp_dir_rel}/watchdog.log 2>&1 &"')
    lines.append(
        'echo "================================================================"'
    )

    return "\n".join(lines) + "\n"


def _build_run_cmd(run, manifest, *, cfg_only: bool) -> list[str]:
    """Build run.py command-line parameters for a run."""
    x = manifest.contract.config.extra
    params = [
        f"problem.name={run.problem_name}",
        f"pipeline={run.pipeline}",
        "prompts=default",
        f"redis.db={run.db}",
        f"stage_timeout={x.get('stage_timeout', 3000)}",
        f"dag_timeout={x.get('dag_timeout', 7200)}",
        f"max_generations={manifest.contract.max_generations}",
        f"max_mutations_per_generation={x.get('max_mutations_per_generation', 8)}",
        f"max_elites_per_generation={x.get('max_elites_per_generation', 8)}",
        f"num_parents={x.get('num_parents', 1)}",
        f"model_name={run.model_name}",
        f'llm_base_url="{run.mutation_url}"',
    ]

    if x.get("mutation_mode"):
        params.append(f"mutation_mode={x['mutation_mode']}")

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
