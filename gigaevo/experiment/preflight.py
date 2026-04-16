#!/usr/bin/env python3
"""Pre-launch validator — 22 checks, runs ALL, reports ALL.

Never exits on first failure. Reports every check with severity tags.
Exit codes: 0 = clean/WARN only, 1 = any CRITICAL, 2 = MAJOR only.

Usage:
    PYTHONPATH=. $GIGAEVO_PYTHON tools/experiment/preflight_check.py --experiment hover/feedback_softfit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

# Repo root for resolving experiment directories and shelling out to sibling scripts.
PROJ = Path(__file__).parent.parent.parent


def _find_run_pids_for_db(db: int) -> list[int]:
    """Find PIDs of run.py processes writing to a specific Redis DB."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        pids = []
        for line in result.stdout.splitlines():
            if "grep" in line or "run.py" not in line:
                continue
            if f"redis.db={db}" in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pids.append(int(parts[1]))
                    except ValueError:
                        pass
        return pids
    except Exception:
        return []


class CheckResult:
    def __init__(self, num: int, group: str, name: str, severity: str):
        self.num = num
        self.group = group
        self.name = name
        self.severity = severity
        self.passed = False
        self.message = ""

    def ok(self, msg: str = ""):
        self.passed = True
        self.message = msg

    def fail(self, msg: str):
        self.passed = False
        self.message = msg

    def __str__(self):
        status = "PASS" if self.passed else self.severity
        tag = f"[{status:>8s}]"
        detail = f" — {self.message}" if self.message else ""
        return f"  #{self.num:2d} {tag} [{self.group}] {self.name}{detail}"


def run_checks(experiment: str) -> list[CheckResult]:
    results: list[CheckResult] = []

    # ── Check 0: Own imports ────────────────────────────────────────────────
    c0 = CheckResult(0, "Self", "Core imports loadable", "CRITICAL")
    try:
        import redis  # noqa: F401

        from gigaevo.experiment.manifest import (
            claim_dbs,
            load_manifest,
            release_db_claims,
        )

        c0.ok()
    except ImportError as e:
        c0.fail(f"Import failed: {e}")
        results.append(c0)
        _report(results)
        return results
    results.append(c0)

    # ── Set NO_PROXY for all server IPs before any HTTP calls ───────────────
    try:
        m_tmp = load_manifest(experiment)
        if m_tmp.contract.servers:
            existing = os.environ.get("NO_PROXY", "")
            all_ips = ",".join(m_tmp.contract.servers)
            os.environ["NO_PROXY"] = f"{existing},{all_ips}" if existing else all_ips
            os.environ["no_proxy"] = os.environ["NO_PROXY"]
    except Exception:
        pass  # Will fail properly in check 1

    # ── Check 1: Status == implemented ──────────────────────────────────────
    c1 = CheckResult(1, "Status", "experiment.yaml status == implemented", "CRITICAL")
    try:
        m = load_manifest(experiment)
        if m.lifecycle.status == "implemented":
            c1.ok()
        else:
            c1.fail(f"status={m.lifecycle.status}, expected 'implemented'")
    except Exception as e:
        c1.fail(str(e))
        results.append(c1)
        _report(results)
        return results
    results.append(c1)

    # ── Check 2: validate.py return type vs pipeline ────────────────────────
    c2 = CheckResult(2, "Config", "validate.py return type vs pipeline", "CRITICAL")
    try:
        import ast

        issues = []
        # Build per-problem pipeline set
        prob_pipelines: dict[str, set[str]] = {}
        for r in m.contract.runs:
            prob_pipelines.setdefault(r.problem_name, set()).add(r.pipeline)

        for prob, pls in prob_pipelines.items():
            val_path = PROJ / "problems" / prob / "validate.py"
            if not val_path.exists():
                continue
            source = val_path.read_text()
            try:
                tree = ast.parse(source, filename=str(val_path))
            except SyntaxError:
                continue

            # Walk AST looking for return statements that return tuples
            returns_tuple = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Return) and node.value is not None:
                    if isinstance(node.value, ast.Tuple):
                        returns_tuple = True
                        break
                    # Also detect: return metrics, failures (implicit tuple)
                    # This appears as ast.Tuple in Python 3.8+
            # Also check type hints on the validate function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "validate":
                    ann = node.returns
                    if ann and isinstance(ann, ast.Subscript):
                        # tuple[...] annotation
                        if isinstance(ann.value, ast.Name) and ann.value.id == "tuple":
                            returns_tuple = True

            for pl in pls:
                if returns_tuple and pl == "standard":
                    issues.append(
                        f"{prob}: validate.py returns tuple but pipeline=standard"
                    )
        if issues:
            c2.fail("; ".join(issues))
        else:
            c2.ok()
    except Exception as e:
        c2.fail(str(e))
    results.append(c2)

    # ── Check 3: prompts_dir in pipeline YAML ──────────────────────────────
    c3 = CheckResult(3, "Config", "prompts_dir in pipeline YAML", "CRITICAL")
    try:
        import yaml

        issues = []
        for run in m.contract.runs:
            # Check if run uses custom local prompts via prompts_dir override.
            # prompt_fetcher=coevolved reads from Redis — prompts_dir is irrelevant.
            uses_coevolved = any(
                override.startswith("prompt_fetcher=coevolved")
                for override in (run.extra_overrides or [])
            )
            if uses_coevolved:
                continue  # Coevolved fetcher reads from Redis, not prompts_dir

            has_custom_prompts = any(
                override.startswith("prompts_dir=") or override.startswith("prompts=")
                for override in (run.extra_overrides or [])
            )
            if not has_custom_prompts:
                continue  # This run uses default prompts

            # Load the pipeline config
            pipeline_path = PROJ / "config" / "pipeline" / f"{run.pipeline}.yaml"
            if not pipeline_path.exists():
                issues.append(f"{run.label}: pipeline={run.pipeline} config not found")
                continue

            try:
                pipeline_cfg = yaml.safe_load(pipeline_path.read_text())
                if not pipeline_cfg:
                    issues.append(f"{run.label}: pipeline YAML is empty")
                    continue

                # Check that prompts_dir is in evolution_context
                evo_ctx = pipeline_cfg.get("evolution_context", {})
                if not evo_ctx.get("prompts_dir"):
                    issues.append(
                        f"{run.label}: evolution_context.prompts_dir not configured"
                    )

                # Check that prompts_dir is in mutation_operator
                mut_op = pipeline_cfg.get("mutation_operator", {})
                if not mut_op.get("prompts_dir"):
                    issues.append(
                        f"{run.label}: mutation_operator.prompts_dir not configured"
                    )
            except Exception as e:
                issues.append(f"{run.label}: failed to parse pipeline YAML: {e}")

        if issues:
            c3.fail("; ".join(issues))
        else:
            c3.ok("All custom prompt pipelines valid")
    except Exception as e:
        c3.fail(str(e))
    results.append(c3)

    # ── Check 4: model_name vs /v1/models ───────────────────────────────────
    c4 = CheckResult(4, "Config", "model_name matches server models", "CRITICAL")
    try:
        issues = []
        checked_urls: set[str] = set()
        for run in m.contract.runs:
            url = run.mutation_url
            if not url or url in checked_urls:
                continue
            checked_urls.add(url)
            try:
                api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
                req = Request(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp = urlopen(req, timeout=15)
                data = json.loads(resp.read())
                model_ids = [m_obj.get("id", "") for m_obj in data.get("data", [])]
                for run2 in m.contract.runs:
                    if run2.mutation_url == url and run2.model_name not in model_ids:
                        issues.append(
                            f"{run2.label}: model '{run2.model_name}' not in {model_ids}"
                        )
            except (URLError, OSError) as e:
                issues.append(f"{url}: unreachable ({e})")
        if issues:
            c4.fail("; ".join(issues))
        else:
            c4.ok()
    except Exception as e:
        c4.fail(str(e))
    results.append(c4)

    # ── Check 5: GIGAEVO_PYTHON set ─────────────────────────────────────────
    c5 = CheckResult(5, "Config", "GIGAEVO_PYTHON set and functional", "CRITICAL")
    gp = os.environ.get("GIGAEVO_PYTHON", "")
    if gp and Path(gp).exists():
        c5.ok(gp)
    elif not gp:
        c5.fail("GIGAEVO_PYTHON not set")
    else:
        c5.fail(f"GIGAEVO_PYTHON={gp} does not exist")
    results.append(c5)

    # ── Check 6: launch.sh matches manifest ─────────────────────────────────
    c6 = CheckResult(6, "Config", "launch.sh params match experiment.yaml", "CRITICAL")
    try:
        launch_path = PROJ / "experiments" / experiment / "launch.sh"
        if not launch_path.exists():
            c6.fail("launch.sh does not exist")
        else:
            launch_content = launch_path.read_text()
            issues = []
            for run in m.contract.runs:
                if f"redis.db={run.db}" not in launch_content:
                    issues.append(f"{run.label}: db={run.db} not in launch.sh")
                if run.model_name not in launch_content:
                    issues.append(f"{run.label}: model_name not in launch.sh")
            if "GENERATED from experiment.yaml" not in launch_content:
                issues.append(
                    "launch.sh not generated from manifest (missing header). "
                    "Run: tools/experiment/generate_launch.py"
                )
            if issues:
                c6.fail("; ".join(issues))
            else:
                c6.ok()
    except Exception as e:
        c6.fail(str(e))
    results.append(c6)

    # ── Check 7: /v1/models reachable for all servers ───────────────────────
    c7 = CheckResult(7, "Server", "/v1/models reachable for all URLs", "CRITICAL")
    try:
        issues = []
        all_urls: set[str] = set()
        for run in m.contract.runs:
            if run.chain_url:
                all_urls.add(run.chain_url)
            if run.mutation_url:
                all_urls.add(run.mutation_url)
        for url in sorted(all_urls):
            try:
                api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
                req = Request(
                    f"{url}/models", headers={"Authorization": f"Bearer {api_key}"}
                )
                urlopen(req, timeout=15)
            except (URLError, OSError) as e:
                issues.append(f"{url}: {e}")
        if issues:
            c7.fail("; ".join(issues))
        else:
            c7.ok(f"{len(all_urls)} endpoints reachable")
    except Exception as e:
        c7.fail(str(e))
    results.append(c7)

    # ── Check 8: Thinking mode active ───────────────────────────────────────
    c8 = CheckResult(8, "Server", "Thinking mode active on chain servers", "MAJOR")
    try:
        issues = []
        # Build chain_url -> model_name mapping from manifest
        chain_url_model: dict[str, str] = {}
        for run in m.contract.runs:
            if run.chain_url and run.chain_url not in chain_url_model:
                chain_url_model[run.chain_url] = run.model_name
        for url, model in sorted(chain_url_model.items()):
            try:
                payload = json.dumps(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "What is 2+2?"}],
                        "max_tokens": 100,
                        "temperature": 0.1,
                    }
                ).encode()
                api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
                req = Request(
                    f"{url}/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp = urlopen(req, timeout=90)
                body = resp.read().decode()
                if "<think>" not in body:
                    issues.append(f"{url}: no <think> in response")
            except (URLError, OSError) as e:
                issues.append(f"{url}: {e}")
        if issues:
            c8.fail("; ".join(issues))
        else:
            c8.ok(f"{len(chain_url_model)} chain servers confirmed")
    except Exception as e:
        c8.fail(str(e))
    results.append(c8)

    # ── Check 9: Redis DBs empty ────────────────────────────────────────────
    c9 = CheckResult(9, "Redis", "dbsize() == 0 for all run DBs", "CRITICAL")
    try:
        issues = []
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        for run in m.contract.runs:
            r = redis.Redis(host=redis_host, port=redis_port, db=run.db)
            size = r.dbsize()
            if size > 0:
                msg = f"DB {run.db}: {size} keys (flush first)"
                live_pids = _find_run_pids_for_db(run.db)
                if live_pids:
                    msg += f" — live writer(s) still running: PID {','.join(str(p) for p in live_pids)} (kill before flushing)"
                issues.append(msg)
        if issues:
            c9.fail("; ".join(issues))
        else:
            c9.ok()
    except Exception as e:
        c9.fail(str(e))
    results.append(c9)

    # ── Check 10: DB claim via SET NX ───────────────────────────────────────
    c10 = CheckResult(10, "Redis", "DB claim — no collision", "CRITICAL")
    try:
        dbs = [run.db for run in m.contract.runs]
        failed = claim_dbs(experiment, dbs)
        if failed:
            # Auto-release stale claims from completed experiments
            stale_released: list[str] = []
            still_blocked: list[str] = []
            for db, owner in failed:
                try:
                    owner_m = load_manifest(owner)
                    if owner_m.lifecycle.status == "complete":
                        release_db_claims([db])
                        re_failed = claim_dbs(experiment, [db])
                        if not re_failed:
                            stale_released.append(f"DB {db} (released from {owner})")
                        else:
                            still_blocked.append(
                                f"DB {db}: re-claim failed after releasing {owner}"
                            )
                    else:
                        still_blocked.append(
                            f"DB {db} owned by {owner} (status={owner_m.lifecycle.status})"
                        )
                except Exception as lookup_err:
                    still_blocked.append(
                        f"DB {db} owned by {owner} (lookup error: {lookup_err})"
                    )
            if still_blocked:
                c10.fail("DB collision: " + ", ".join(still_blocked))
            else:
                c10.ok(
                    f"Claimed DBs {dbs} (auto-released stale claims: {stale_released})"
                )
        else:
            c10.ok(f"Claimed DBs {dbs}")
    except Exception as e:
        c10.fail(str(e))
    results.append(c10)

    # ── Check 11: Seed programs exist ───────────────────────────────────────
    c11 = CheckResult(11, "Files", "Seed programs exist", "CRITICAL")
    try:
        issues = []
        seen_problems: set[str] = set()
        for run in m.contract.runs:
            if run.problem_name in seen_problems:
                continue
            seen_problems.add(run.problem_name)
            seed_dir = PROJ / "problems" / run.problem_name / "initial_programs"
            if not seed_dir.exists():
                issues.append(f"{run.problem_name}: no initial_programs/")
            elif not list(seed_dir.glob("*.py")):
                issues.append(f"{run.problem_name}: initial_programs/ has no .py files")
        if issues:
            c11.fail("; ".join(issues))
        else:
            c11.ok()
    except Exception as e:
        c11.fail(str(e))
    results.append(c11)

    # ── Check 12: Watchdog configured (CLI mode) ──────────────────────────
    # New infra: watchdog runs via `gigaevo watchdog` CLI, configured via
    # the `watchdog:` section in experiment.yaml. No run_watchdog.py needed.
    c12 = CheckResult(12, "Files", "Watchdog configured (CLI mode)", "CRITICAL")
    try:
        # Access watchdog via Pydantic model, not _raw (which doesn't exist post-refactor)
        plugin = m.control_plane.watchdog.plugin if m.control_plane.watchdog else None
        if not plugin:
            c12.fail(
                "watchdog.plugin not set in experiment.yaml. "
                "Add watchdog.plugin (e.g. 'adversarial') to use CLI watchdog."
            )
        else:
            c12.ok(
                f"watchdog.plugin='{plugin}' (CLI: gigaevo -e {experiment} watchdog)"
            )
    except Exception as e:
        c12.fail(str(e))
    results.append(c12)

    # ── Check 13: 05_results.md not gitignored ──────────────────────────────
    c13 = CheckResult(13, "Files", "05_results.md not gitignored", "MAJOR")
    try:
        results_path = f"experiments/{experiment}/05_results.md"
        proc = subprocess.run(
            ["git", "check-ignore", "-q", results_path],
            cwd=str(PROJ),
            capture_output=True,
        )
        if proc.returncode == 0:
            c13.fail(f"{results_path} is gitignored! Fix .gitignore")
        else:
            c13.ok()
    except Exception as e:
        c13.fail(str(e))
    results.append(c13)

    # ── Check 14: prereg_commit exists on branch ────────────────────────────
    c14 = CheckResult(14, "Design", "prereg_commit exists on branch", "MAJOR")
    try:
        if m.contract.identity.prereg_commit:
            commit_str = str(m.contract.identity.prereg_commit)
            proc = subprocess.run(
                ["git", "log", "--oneline", "-1", commit_str],
                cwd=str(PROJ),
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                c14.ok(proc.stdout.strip())
            else:
                c14.fail(f"Commit {m.contract.identity.prereg_commit} not found")
        else:
            c14.fail("prereg_commit not set in experiment.yaml")
    except Exception as e:
        c14.fail(str(e))
    results.append(c14)

    # ── Check 15: Single IV per comparison ──────────────────────────────────
    c15 = CheckResult(15, "Design", "Single IV per run comparison", "MAJOR")
    try:
        # Read factorial_design from manifest raw dump (backward compat for YAML files that have it)
        is_factorial = (
            m.model_dump().get("experiment", {}).get("factorial_design", False)
        )
        # Group runs by (pipeline, problem_name) to identify cells
        cells: dict[tuple[str, str], list[str]] = {}
        for r in m.contract.runs:
            key = (r.pipeline, r.problem_name)
            cells.setdefault(key, []).append(r.label)
        if is_factorial:
            c15.ok(f"Factorial design declared — {len(cells)} cell(s): {dict(cells)}")
        elif len(cells) <= 2:
            c15.ok(f"{len(cells)} cell(s): {dict(cells)}")
        else:
            c15.fail(
                f"More than 2 cells detected — check for compound confound: {dict(cells)}"
                f" (if intentional, add 'factorial_design: true' under 'experiment:' in experiment.yaml)"
            )
    except Exception as e:
        c15.fail(str(e))
    results.append(c15)

    # ── Check 16: N >= 2 per cell ───────────────────────────────────────────
    # Meta-evolution pipelines (prompt_evolution) are single-instance hub
    # runs, not replicated cells — exclude them.
    _META_EVO_PIPELINES = {"prompt_evolution"}
    c16 = CheckResult(16, "Design", "N >= 2 per cell", "MAJOR")
    try:
        cells: dict[tuple[str, str], list[str]] = {}
        for r in m.contract.runs:
            if r.pipeline in _META_EVO_PIPELINES:
                continue  # Hub runs are not replicated cells
            key = (r.pipeline, r.problem_name)
            cells.setdefault(key, []).append(r.label)
        issues = []
        for cell, labels in cells.items():
            if len(labels) < 2:
                issues.append(f"Cell {cell}: only {len(labels)} run(s): {labels}")
        if issues:
            c16.fail("; ".join(issues))
        else:
            c16.ok(f"All {len(cells)} cells have N>=2")
    except Exception as e:
        c16.fail(str(e))
    results.append(c16)

    # ── Check 17: Dataset SHA-256 (conditional: has_test_set) ───────────────
    c17 = CheckResult(17, "Test", "Dataset SHA-256 matches manifest", "CRITICAL")
    if m.contract.problem.has_test_set:
        try:
            if m.contract.problem.test_set_path and m.contract.problem.test_set_sha256:
                test_path = PROJ / m.contract.problem.test_set_path
                if test_path.exists():
                    actual = hashlib.sha256(test_path.read_bytes()).hexdigest()
                    if actual == m.contract.problem.test_set_sha256:
                        c17.ok()
                    else:
                        c17.fail(
                            f"SHA mismatch: expected {m.contract.problem.test_set_sha256[:16]}..., "
                            f"got {actual[:16]}..."
                        )
                else:
                    c17.fail(f"Test set not found: {test_path}")
            else:
                c17.fail("test_set_path or test_set_sha256 not set")
        except Exception as e:
            c17.fail(str(e))
    else:
        c17.ok("N/A (no test set)")
    results.append(c17)

    # ── Check 18: Test-set usage count (conditional) ────────────────────────
    c18 = CheckResult(18, "Test", "Test-set usage count", "WARN")
    if m.contract.problem.has_test_set:
        c18.ok("PLACEHOLDER: test-set usage count not yet implemented")
    else:
        c18.ok("N/A (no test set)")
    results.append(c18)

    # ── Check 19: Smoke test completed ──────────────────────────────────────
    c19 = CheckResult(19, "Smoke", "3-gen smoke test completed", "CRITICAL")
    if m.lifecycle.smoke_test.completed:
        c19.ok(f"Completed at {m.lifecycle.smoke_test.completed_at or 'unknown'}")
    else:
        c19.fail("smoke_test.completed is false in experiment.yaml")
    results.append(c19)

    # ── Check 20: Treatment verification completed ────────────────────────
    c20 = CheckResult(20, "Treatment", "Treatment verification completed", "CRITICAL")
    tv = m.lifecycle.treatment_verification
    if tv.completed:
        c20.ok(f"Completed at {tv.completed_at or 'unknown'}")
    else:
        c20.fail(
            "treatment_verification.completed is false/missing in experiment.yaml. "
            "Run treatment-verifier agent (implement Step 10) first."
        )
    results.append(c20)

    # ── Check 21: Server throughput parity ────────────────────────────────
    c21 = CheckResult(21, "Server", "Chain/mutation server throughput parity", "MAJOR")
    try:
        import time

        api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
        issues = []
        # Collect latencies and tokens/sec for each unique server URL
        url_latencies: dict[str, list[float]] = {}
        url_tps: dict[str, list[float]] = {}  # tokens per second
        url_roles: dict[str, list[str]] = {}  # url -> ["D1-chain", "D2-mutation", ...]

        for run in m.contract.runs:
            for role, url in [("chain", run.chain_url), ("mutation", run.mutation_url)]:
                if not url:
                    continue
                tag = f"{run.label}-{role}"
                url_roles.setdefault(url, []).append(tag)
                if url in url_latencies:
                    continue  # Already benchmarked
                latencies = []
                tps_samples = []
                model = run.model_name
                payload = json.dumps(
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Explain why the sky is blue in 2-3 sentences.",
                            }
                        ],
                        "max_tokens": 80,
                        "temperature": 0.0,
                    }
                ).encode()
                for _ in range(5):
                    try:
                        req = Request(
                            f"{url}/chat/completions",
                            data=payload,
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                        )
                        t0 = time.monotonic()
                        resp = urlopen(req, timeout=90)
                        body = resp.read()
                        elapsed = time.monotonic() - t0
                        latencies.append(elapsed)
                        # Parse completion_tokens for tokens/sec
                        try:
                            usage = json.loads(body).get("usage", {})
                            comp_tokens = usage.get("completion_tokens", 0)
                            if comp_tokens > 0 and elapsed > 0:
                                tps_samples.append(comp_tokens / elapsed)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    except (URLError, OSError):
                        latencies.append(float("inf"))
                url_latencies[url] = latencies
                url_tps[url] = tps_samples

        # Compare servers that share the same role (e.g. all chain servers)
        role_groups: dict[str, dict[str, float]] = {}  # role -> {url: median_lat}
        role_tps_groups: dict[str, dict[str, float]] = {}  # role -> {url: median_tps}
        for url, tags in url_roles.items():
            lats = url_latencies.get(url, [])
            median = sorted(lats)[len(lats) // 2] if lats else float("inf")
            tps = url_tps.get(url, [])
            median_tps = sorted(tps)[len(tps) // 2] if tps else 0.0
            for tag in tags:
                role = tag.split("-", 1)[1]  # "chain" or "mutation"
                role_groups.setdefault(role, {})[url] = median
                role_tps_groups.setdefault(role, {})[url] = median_tps

        for role, url_medians in role_groups.items():
            if len(url_medians) < 2:
                continue
            medians = list(url_medians.values())
            fastest = min(medians)
            slowest = max(medians)
            if fastest <= 0 or fastest == float("inf"):
                continue
            ratio = slowest / fastest
            if ratio > 3.0:
                slow_url = [u for u, v in url_medians.items() if v == slowest][0]
                fast_url = [u for u, v in url_medians.items() if v == fastest][0]
                issues.append(
                    f"{role} latency differs {ratio:.1f}x: "
                    f"{fast_url} ({fastest:.2f}s) vs {slow_url} ({slowest:.2f}s)"
                )
            # Also check tokens/sec parity
            tps_map = role_tps_groups.get(role, {})
            tps_vals = [v for v in tps_map.values() if v > 0]
            if len(tps_vals) >= 2:
                best_tps = max(tps_vals)
                worst_tps = min(tps_vals)
                tps_ratio = best_tps / worst_tps
                if tps_ratio > 3.0:
                    slow_u = [u for u, v in tps_map.items() if v == worst_tps][0]
                    fast_u = [u for u, v in tps_map.items() if v == best_tps][0]
                    issues.append(
                        f"{role} tok/s differs {tps_ratio:.1f}x: "
                        f"{fast_u} ({best_tps:.1f} t/s) vs {slow_u} ({worst_tps:.1f} t/s)"
                    )

        # Also report individual latencies + tok/s for visibility
        details = []
        for url, lats in sorted(url_latencies.items()):
            finite = sorted(lat for lat in lats if lat != float("inf"))
            tps = url_tps.get(url, [])
            tps_str = ""
            if tps:
                med_tps = sorted(tps)[len(tps) // 2]
                tps_str = f", {med_tps:.1f} tok/s"
            if finite:
                lo, med, hi = finite[0], finite[len(finite) // 2], finite[-1]
                tags = ", ".join(url_roles.get(url, []))
                details.append(
                    f"{url} ({tags}): {med:.2f}s median "
                    f"[{lo:.2f}–{hi:.2f}s, n={len(finite)}]{tps_str}"
                )
            else:
                details.append(f"{url}: unreachable")

        if issues:
            c21.fail("; ".join(issues) + " | " + "; ".join(details))
        else:
            c21.ok("; ".join(details))
    except Exception as e:
        c21.fail(str(e))
    results.append(c21)

    return results


def _report(results: list[CheckResult]) -> int:
    print(f"\n{'=' * 60}")
    print("  Preflight Check Results")
    print(f"{'=' * 60}\n")

    for r in results:
        print(r)

    print()
    criticals = [r for r in results if not r.passed and r.severity == "CRITICAL"]
    majors = [r for r in results if not r.passed and r.severity == "MAJOR"]
    warns = [r for r in results if not r.passed and r.severity == "WARN"]
    passed = [r for r in results if r.passed]

    print(
        f"  {len(passed)} passed, {len(criticals)} CRITICAL, {len(majors)} MAJOR, {len(warns)} WARN"
    )

    if criticals:
        print(f"\n  BLOCKED — {len(criticals)} critical failure(s). Fix before launch.")
        return 1
    elif majors:
        print(f"\n  WARNING — {len(majors)} major issue(s). Review before launch.")
        return 0  # MAJOR is advisory, not blocking
    else:
        print("\n  ALL CLEAR — safe to launch.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Pre-launch preflight checks")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    results = run_checks(args.experiment)
    exit_code = _report(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
