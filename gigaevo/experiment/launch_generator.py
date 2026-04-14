#!/usr/bin/env python3
"""Generate launch.sh from experiment.yaml manifest.

Eliminates hand-written launch scripts that drift from the manifest.
All values (servers, runs, config, custom_env) come from experiment.yaml.

DEPRECATED: Use `gigaevo -e task/name generate-launch` instead.

Usage (legacy):
    PYTHONPATH=. $GIGAEVO_PYTHON tools/experiment/generate_launch.py --experiment hover/feedback_softfit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Derive project root from this file's location
PROJ_PATH = str(Path(__file__).resolve().parent.parent.parent)

# Ensure repo root is on sys.path so `tools.experiment.manifest` is importable
# when invoked via the `gigaevo` console script.
if PROJ_PATH not in sys.path:
    sys.path.insert(0, PROJ_PATH)

from tools.experiment.manifest import experiment_dir, load_manifest  # noqa: E402

PYTHON_PATH = os.environ.get(
    "GIGAEVO_PYTHON", "python3"
)


def generate(experiment: str) -> str:
    m = load_manifest(experiment)
    exp_dir_rel = f"experiments/{experiment}"

    # Collect all server IPs for NO_PROXY
    no_proxy_hosts = ["localhost", "127.0.0.1", "api.github.com"] + m.servers
    no_proxy = ",".join(no_proxy_hosts)

    lines: list[str] = []

    # Header
    lines.append("#!/usr/bin/env bash")
    lines.append("# GENERATED from experiment.yaml — do not edit manually.")
    lines.append(
        f"# Regenerate: PYTHONPATH=. $GIGAEVO_PYTHON tools/experiment/generate_launch.py --experiment {experiment}"
    )
    lines.append("#")
    lines.append(f"# Experiment: {m.name}")
    lines.append(f"# Branch: {m.branch}")
    if m.pr_number:
        lines.append(f"# PR: #{m.pr_number}")
    if m.prereg_commit:
        lines.append(f"# Pre-reg commit: {m.prereg_commit}")
    lines.append("#")
    lines.append(f"# Runs: {', '.join(r.label for r in m.runs)}")
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
    if m.custom_env:
        lines.append(
            "# Task-specific environment variables (from experiment.yaml custom_env)"
        )
        for key, val in m.custom_env.items():
            lines.append(f'export {key}="{val}"')
        lines.append("")

    # Banner
    lines.append(
        'echo "================================================================"'
    )
    lines.append(
        f"echo \"{m.name} experiment launch — $(date -u '+%Y-%m-%d %H:%M UTC')\""
    )
    if m.prereg_commit:
        lines.append(f'echo "Pre-reg commit: {m.prereg_commit}"')
    for run in m.runs:
        lines.append(f'echo "{run.label}: {run.condition} — pipeline={run.pipeline}"')
    lines.append(
        'echo "================================================================"'
    )
    lines.append('echo ""')
    lines.append("")

    # Preflight call
    lines.append(
        "# ── Preflight check (hard gate) ──────────────────────────────────────────"
    )
    lines.append(
        f'PYTHONPATH="$PROJ" "$PYTHON" "$PROJ/tools/experiment/preflight_check.py" --experiment {experiment}'
    )
    lines.append('echo ""')
    lines.append("")

    # Config verification
    lines.append(
        "# ── Config verification (--cfg job) ───────────────────────────────────────"
    )
    for run in m.runs:
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
    for run in m.runs:
        pid_var = f"PID_{run.label.replace('-', '_')}"
        pid_vars.append(pid_var)
        cmd_params = _build_run_cmd(run, m, cfg_only=False)

        # Env prefix for per-run chain URL (only if run has a specific URL;
        # skip if null — the global export from custom_env handles shared LB)
        env_prefix = ""
        if run.chain_url:
            chain_url_env_var = m.config.get("chain_url_env_var", "CHAIN_URL")
            env_prefix = f'{chain_url_env_var}="{run.chain_url}" '
        # Per-run env vars (e.g. GIGAEVO_SOFT_FITNESS=1 for soft fitness IV)
        if run.run_env:
            for key, val in run.run_env.items():
                env_prefix += f'{key}="{val}" '

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
    lines.append(f'echo "All {len(m.runs)} runs launched."')

    pid_echo = "  ".join(f"{r.label}=$PID_{r.label.replace('-', '_')}" for r in m.runs)
    lines.append(f'echo "PIDs: {pid_echo}"')

    # Write PIDs to file
    pid_file_content = " ".join(f"$PID_{r.label.replace('-', '_')}" for r in m.runs)
    lines.append(f'echo "{pid_file_content}" > "$LOG_DIR/pids.txt"')
    lines.append("")

    # Verify PIDs alive
    lines.append(
        "# ── Verify all PIDs alive ──────────────────────────────────────────────────"
    )
    lines.append("sleep 5")
    lines.append("ALL_ALIVE=true")
    for run in m.runs:
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
    label_list = " ".join(r.label for r in m.runs)
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
    c = manifest.config
    params = [
        f"problem.name={run.problem_name}",
        f"pipeline={run.pipeline}",
        "prompts=default",
        f"redis.db={run.db}",
        f"stage_timeout={c.get('stage_timeout', 3000)}",
        f"dag_timeout={c.get('dag_timeout', 7200)}",
        f"max_generations={manifest.max_generations}",
        f"max_mutations_per_generation={c.get('max_mutations_per_generation', 8)}",
        f"max_elites_per_generation={c.get('max_elites_per_generation', 8)}",
        f"num_parents={c.get('num_parents', 1)}",
        f"model_name={run.model_name}",
        f'llm_base_url="{run.mutation_url}"',
    ]

    if c.get("mutation_mode"):
        params.append(f"mutation_mode={c['mutation_mode']}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate launch.sh from experiment.yaml"
    )
    parser.add_argument(
        "--experiment", required=True, help="e.g. hover/feedback_softfit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print to stdout instead of writing"
    )
    args = parser.parse_args()

    content = generate(args.experiment)

    if args.dry_run:
        print(content)
    else:
        out_path = experiment_dir(args.experiment) / "launch.sh"
        out_path.write_text(content)
        out_path.chmod(0o755)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
