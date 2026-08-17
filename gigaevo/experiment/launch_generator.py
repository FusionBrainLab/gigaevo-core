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
    "GIGAEVO_PYTHON",
    "python3",
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
            chain_url_env_var = m.contract.config.effective_overrides.get(
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

    # Watchdog launch hint (CLI path: loads .env, wires Telegram + plugin).
    lines.append('echo ""')
    lines.append('echo "Launch watchdog (CLI — loads .env, wires Telegram):"')
    lines.append(f'echo "  nohup gigaevo -e {experiment} watchdog \\\\"')
    lines.append(f'echo "      > {exp_dir_rel}/watchdog.log 2>&1 &"')
    lines.append(
        'echo "================================================================"'
    )

    return "\n".join(lines) + "\n"


# Keys consumed by the hand-rolled "base params" block below — must not be
# re-emitted by the generic shared_overrides sweep or they would appear twice.
_BUILTIN_EMITTED = frozenset(
    {
        "storage",
        "stage_timeout",
        "dag_timeout",
        "num_parents",
        "mutation_mode",
    }
)

# Keys in contract.config.shared_overrides that are NOT Hydra overrides.
# These drive launch.sh preamble/postamble behavior, not the run.py CLI.
_NOT_HYDRA = frozenset({"chain_url_env_var"})


def _build_run_cmd(
    run, manifest, *, cfg_only: bool, shell_escape: bool = True
) -> list[str]:
    """Build run.py command-line parameters for a run.

    shell_escape: when True, wrap ${...} overrides in single-quotes so bash
    doesn't variable-expand them (KF-02). Set False when handing the list to
    subprocess.run() without shell=True — literal single-quotes would end up
    inside the argument and Hydra's override lexer would reject it.

    Override precedence (lowest → highest):
      1. ``contract.config.task_group`` — emitted FIRST as
         ``experiment=<task_group>`` so ``config/experiment/<task_group>.yaml``
         loads as the task-level tradition before anything else overrides it.
      2. Built-in defaults from this file (``stage_timeout=3000`` etc).
      3. ``contract.config.effective_overrides`` — merged view of flat legacy
         ``flat_overrides`` (model_extra) + nested ``shared_overrides`` dict;
         nested wins on key conflict (I-00). Any non-builtin, non-blacklisted
         key lands on the command line as ``key=value``.
      4. ``run.extra_overrides`` — per-run, wins over everything above because
         Hydra treats later CLI args as overrides of earlier ones.
    """
    x = manifest.contract.config.effective_overrides
    params: list[str] = []
    # Task-group first so every later override can win over it.
    task_group = manifest.contract.config.task_group
    if task_group:
        params.append(f"experiment={task_group}")
    params.extend(
        [
            f"problem.name={run.problem_name}",
            f"pipeline={run.pipeline}",
            # The manifest contract identifies runs by Redis prefix + DB and
            # has no durable disk output path. Make that ownership explicit
            # instead of inheriting the framework's direct-run disk default.
            "storage=redis",
            "prompts=default",
            f"redis.db={run.db}",
            f"stage_timeout={x.get('stage_timeout', 3000)}",
            f"dag_timeout={x.get('dag_timeout', 7200)}",
            f"max_mutants={manifest.contract.max_generations}",
            f"num_parents={x.get('num_parents', 1)}",
            f"model_name={run.model_name}",
            f'llm_base_url="{run.mutation_url}"'
            if shell_escape
            else f"llm_base_url={run.mutation_url}",
        ]
    )

    if x.get("mutation_mode"):
        params.append(f"mutation_mode={x['mutation_mode']}")

    # Forward any additional shared_overrides keys as Hydra overrides. Built-ins
    # already landed above; _NOT_HYDRA keys steer launch.sh, not run.py. This
    # is the fix for I-00: before, a user writing
    # ``contract.config.shared_overrides.stopper: max_mutants`` was silently
    # dropped.
    for key, val in x.items():
        if key in _BUILTIN_EMITTED or key in _NOT_HYDRA or val is None:
            continue
        ov = f"{key}={val}"
        if shell_escape and "${" in ov:
            params.append(f"'{ov}'")
        else:
            params.append(ov)

    if run.extra_overrides:
        for ov in run.extra_overrides:
            # Writing `\${...}` in experiment.yaml is a no-op — ruamel.yaml
            # strips the backslash — so this single-quote wrap is the ONLY
            # defense against bash variable-expanding `${...}` to empty (I-04).
            if shell_escape and "${" in ov:
                params.append(f"'{ov}'")
            else:
                params.append(ov)

    if cfg_only:
        params.append("--cfg job")

    return params
