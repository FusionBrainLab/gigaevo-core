"""Resolve Hydra config per run ahead of launch — Phase 3 of the Config-Override
Integrity Pipeline. See .claude/plans/humble-weaving-shamir.md.

For each run in the manifest, invokes ``python run.py ... --cfg job``,
captures stdout, parses with OmegaConf, persists to
``experiments/<exp>/cfg_run_<label>.yaml``, and records a sha256 fingerprint
for every Hydra config file that could have contributed to the resolution.

Public surface:
    - ``DryRunResult`` — dataclass with ``resolved``, ``fingerprint``, ``cli_args``
    - ``dry_run(experiment)`` — run the compile per run and return the result

The subprocess invocation is behind a private seam ``_invoke_run_py_cfg_job``
so tests can stub it without spinning up Hydra.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from omegaconf import OmegaConf

from gigaevo.config.resolvers import register_resolvers
from gigaevo.experiment import manifest as _manifest_mod
from gigaevo.experiment.launch_generator import _build_run_cmd
from gigaevo.experiment.manifest import experiment_dir, load_manifest

# Register project resolvers (eval/get_object/merge/len/ref) so that
# OmegaConf.to_container(..., resolve=True) can resolve ${get_object:...}
# interpolations present in resolved --cfg job output. Safe to call repeatedly:
# OmegaConf.register_new_resolver is idempotent when replace=False is omitted
# (it no-ops on re-registration of the same name).
try:
    register_resolvers()
except ValueError:
    # Resolvers already registered in this process — fine.
    pass


def _hydra_stub_resolver(path: str) -> str:
    """Stub for ``${hydra:...}`` references in re-loaded --cfg job dumps.

    The real Hydra resolver is only registered inside a Hydra application
    context; when we re-load ``cfg_run_<label>.yaml`` the config contains
    entries like ``log_dir: ${hydra:runtime.output_dir}`` and
    ``prompts.dir: ${hydra:runtime.cwd}/gigaevo/prompts/...`` that cannot be
    resolved outside Hydra. The preview only needs values it can diff
    against ``contract.config.pinned``; ``hydra:`` paths are never pinned.
    Return a stable placeholder so resolution succeeds without affecting
    pin-matching.
    """
    return f"<hydra:{path}>"


try:
    OmegaConf.register_new_resolver("hydra", _hydra_stub_resolver)
except ValueError:
    pass


def _ref_stub_resolver(path: str) -> str:
    """Stub for ``${ref:...}`` that returns a placeholder instead of instantiating.

    The real resolver (``gigaevo.config.resolvers._ref_resolver``) calls
    ``hydra.utils.instantiate`` on the referenced node — fine inside a
    running Hydra application but fatal in dry-run, where the resolved
    value must round-trip through ``OmegaConf.to_container`` as a
    primitive. A live ``RedisProgramStorage`` instance is not a primitive
    and triggers ``UnsupportedValueType``.

    Pinned paths never point into a ``${ref:...}``-resolved node, so
    returning a stable placeholder preserves pin-matching semantics.
    """
    return f"<ref:{path}>"


OmegaConf.register_new_resolver("ref", _ref_stub_resolver, replace=True)


# Set placeholder env vars for ${oc.env:...} references that appear in the
# default LLM/logging configs. The preview only needs pin-match values;
# secrets don't matter.
for _env_key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.setdefault(_env_key, "<dry-run-stub>")

PYTHON_PATH_DEFAULT = "python3"


@dataclass(frozen=True)
class DryRunResult:
    """Outcome of a dry-run compile for one experiment."""

    resolved: dict[str, dict[str, Any]] = field(default_factory=dict)
    """label -> resolved OmegaConf dict."""

    fingerprint: dict[str, str] = field(default_factory=dict)
    """relative config path -> sha256 hex."""

    cli_args: dict[str, list[str]] = field(default_factory=dict)
    """label -> CLI arg list passed to run.py (for provenance tracking)."""


def dry_run(
    experiment: str,
    *,
    python_path: str = PYTHON_PATH_DEFAULT,
    max_workers: int = 4,
    timeout: float = 120,
) -> DryRunResult:
    """Compile resolved Hydra config for every run in ``experiment``.

    Args:
        experiment: Experiment name (e.g. ``heilbron/k5-budget-v2``).
        python_path: Python interpreter to invoke ``run.py`` with.
        max_workers: Parallel subprocesses (bounded for NFS friendliness).
        timeout: Per-subprocess timeout in seconds.

    Raises:
        RuntimeError: if any subprocess fails or Hydra compose errors out.
    """
    manifest = load_manifest(experiment)
    exp_dir = experiment_dir(experiment)
    proj = _manifest_mod.PROJ
    run_py = proj / "run.py"

    # Step 1: Build CLI args per run (reuses launch_generator logic so task_group,
    # extras, and per-run overrides are all handled consistently).
    cli_args: dict[str, list[str]] = {}
    for run in manifest.contract.runs:
        cli_args[run.label] = _build_run_cmd(
            run, manifest, cfg_only=True, shell_escape=False
        )

    # Step 2: Invoke run.py in parallel and persist.
    resolved: dict[str, dict[str, Any]] = {}

    def _one(run_label: str, args: list[str]) -> tuple[str, dict[str, Any]]:
        full_args = [python_path, str(run_py), *args]
        stdout = _invoke_run_py_cfg_job(full_args, cwd=proj, timeout=timeout)
        out_path = exp_dir / f"cfg_run_{run_label}.yaml"
        out_path.write_text(stdout)
        doc = OmegaConf.to_container(OmegaConf.load(out_path), resolve=True)
        if not isinstance(doc, dict):
            raise RuntimeError(
                f"run {run_label}: --cfg job output did not parse as a mapping; "
                f"got {type(doc).__name__}. Check {out_path} for Hydra errors."
            )
        return run_label, doc  # type: ignore[return-value]

    if max_workers <= 1 or len(cli_args) == 1:
        for label, args in cli_args.items():
            label, doc = _one(label, args)
            resolved[label] = doc
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(cli_args))
        ) as pool:
            futures = [
                pool.submit(_one, label, args) for label, args in cli_args.items()
            ]
            for fut in concurrent.futures.as_completed(futures):
                label, doc = fut.result()  # re-raises worker exceptions
                resolved[label] = doc

    # Step 3: Fingerprint Hydra-reachable config files.
    fingerprint = _fingerprint_config(manifest)

    return DryRunResult(
        resolved=resolved,
        fingerprint=fingerprint,
        cli_args=cli_args,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _invoke_run_py_cfg_job(
    args: list[str],
    cwd: Path,
    timeout: float = 120,
) -> str:
    """Invoke the command in ``args`` and return stdout.

    Mockable seam. Raises ``RuntimeError`` with a tail of stderr on failure.
    ``args[0]`` is the python interpreter, ``args[1]`` is run.py path,
    remainder are Hydra overrides including ``--cfg job``.
    """
    # Multi-word args like ``--cfg job`` must be split when passed to subprocess.
    flat_args: list[str] = []
    for a in args:
        if a.startswith("--") and " " in a:
            flat_args.extend(a.split(" ", 1))
        else:
            flat_args.append(a)

    result = subprocess.run(
        flat_args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-500:]
        raise RuntimeError(
            f"run.py --cfg job failed (exit {result.returncode}):\n{tail}"
        )
    return result.stdout


def _fingerprint_config(manifest) -> dict[str, str]:
    """Hash config files that could contribute to the resolved config.

    Scope (in priority order, following Hydra's defaults chain):
      1. ``config/config.yaml`` — root entry point
      2. ``config/experiment/*.yaml`` — group file(s); always include the
         chosen task_group and always include ``base.yaml`` since config.yaml
         defaults to it
      3. ``config/constants/*.yaml`` — small, cross-cutting, always composed
      4. ``config/pipeline/<pipeline>.yaml`` — the pipeline each run chose

    Returns a map of ``relative posix path`` -> ``sha256 hex``. Paths use
    forward slashes for platform-independent stability.
    """
    out: dict[str, str] = {}
    proj = _manifest_mod.PROJ

    def _add(path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        try:
            rel = path.resolve().relative_to(proj.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    cfg_root = proj / "config"
    _add(cfg_root / "config.yaml")

    for experiment_file in (cfg_root / "experiment").glob("*.yaml"):
        _add(experiment_file)

    for constant_file in (cfg_root / "constants").glob("*.yaml"):
        _add(constant_file)

    pipelines_seen: set[str] = set()
    for run in manifest.contract.runs:
        if run.pipeline in pipelines_seen:
            continue
        pipelines_seen.add(run.pipeline)
        _add(cfg_root / "pipeline" / f"{run.pipeline}.yaml")

    return out
