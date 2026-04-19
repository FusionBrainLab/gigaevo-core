"""Minimal principled preflight checks.

Only checks that the schema/Pydantic can't enforce and that target real
operator failure modes. ~10 checks replacing the 22-check ``preflight.py``.

Exit codes: 0 = clean/WARN only, 1 = any CRITICAL.
"""

from __future__ import annotations

from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from gigaevo.experiment.dry_run import dry_run
from gigaevo.experiment.manifest import load_manifest

PROJ = Path(__file__).resolve().parent.parent.parent


class Severity(StrEnum):
    CRITICAL = "CRITICAL"
    WARN = "WARN"


class CheckResult:
    __slots__ = ("name", "severity", "passed", "message")

    def __init__(self, name: str, severity: Severity):
        self.name = name
        self.severity = severity
        self.passed = False
        self.message = ""

    def ok(self, msg: str = "") -> None:
        self.passed = True
        self.message = msg

    def fail(self, msg: str) -> None:
        self.passed = False
        self.message = msg

    @property
    def is_blocking(self) -> bool:
        return not self.passed and self.severity == Severity.CRITICAL

    def __str__(self) -> str:
        tag = "PASS" if self.passed else self.severity.value
        detail = f" -- {self.message}" if self.message else ""
        return f"  [{tag:>8s}] {self.name}{detail}"


def run_checks(experiment: str) -> list[CheckResult]:
    """Run all preflight checks. Returns list of CheckResult."""
    results: list[CheckResult] = []

    # 1. Status gate
    c = CheckResult("Status == implemented", Severity.CRITICAL)
    try:
        m = load_manifest(experiment)
        if m.lifecycle.status == "implemented":
            c.ok()
        else:
            c.fail(f"status={m.lifecycle.status}, expected 'implemented'")
    except Exception as e:
        c.fail(str(e))
        results.append(c)
        return results
    results.append(c)
    if not c.passed:
        return results

    # Proxy (HTTPS_PROXY / NO_PROXY) is inherited from the user's shell/.env —
    # the preflight no longer builds NO_PROXY from manifest fields.

    # 2. GIGAEVO_PYTHON
    _check_gigaevo_python(results)

    # 3. LLM endpoint reachable (merged config)
    _check_llm_reachable(results, m)

    # 4. Model IDs match
    _check_model_ids(results, m)

    # 4b. Resolved Hydra config matches declared pins
    _check_resolved_config_matches_pinned(results, m, experiment)

    # 4c. Config fingerprint stable (active only on re-launch)
    _check_config_fingerprint_stable(results, m, experiment)

    # 5. Redis DBs empty
    _check_redis_empty(results, m)

    # 6. DB claims
    _check_db_claims(results, m, experiment)

    # 7. Seed programs
    _check_seed_programs(results, m)

    # 8. Test-set SHA
    _check_test_set_sha(results, m)

    # 9. Smoke test
    _check_smoke_test(results, m)

    # 10. Treatment verification
    _check_treatment_verification(results, m)

    return results


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_gigaevo_python(results: list[CheckResult]) -> None:
    c = CheckResult("GIGAEVO_PYTHON set and executable", Severity.CRITICAL)
    gp = os.environ.get("GIGAEVO_PYTHON", "")
    if gp and Path(gp).exists():
        c.ok(gp)
    elif not gp:
        c.fail("GIGAEVO_PYTHON not set")
    else:
        c.fail(f"GIGAEVO_PYTHON={gp} does not exist")
    results.append(c)


def _resolve_run_overrides(m, run) -> dict:
    """Merge ``contract.config.extra`` with ``run.extra_overrides`` (run wins).

    ``extra_overrides`` is a list of Hydra-style ``key=value`` strings; keys
    containing dots (e.g. ``opponent_provider.cache_ttl``) are preserved as
    flat string keys — the preflight does not expand dotted paths.
    """
    merged: dict = dict(m.contract.config.extra or {})
    for ov in run.extra_overrides or []:
        if "=" not in ov:
            continue
        key, _, value = ov.partition("=")
        key = key.strip().lstrip("+").lstrip("~")
        if not key:
            continue
        merged[key] = value.strip()
    return merged


def _check_llm_reachable(results: list[CheckResult], m) -> None:
    """Probe ``{merged.llm_base_url}/models`` for each run.

    Proxy behavior (HTTPS_PROXY / NO_PROXY) comes from the user's shell/.env —
    we don't reach into the manifest. A probe failure that looks proxy-related
    hints at a shell misconfiguration; all other failures are reported as-is.
    """
    c = CheckResult("LLM endpoint reachable (/v1/models)", Severity.CRITICAL)
    try:
        issues: list[str] = []
        all_urls: set[str] = set()
        for run in m.contract.runs:
            merged = _resolve_run_overrides(m, run)
            url = merged.get("llm_base_url")
            if url:
                all_urls.add(str(url))
        api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
        for url in sorted(all_urls):
            try:
                req = Request(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                urlopen(req, timeout=15)
            except (URLError, OSError) as e:
                issues.append(f"{url}: {e}")
        if issues:
            c.fail(
                "; ".join(issues)
                + " (if behind a corporate proxy, check HTTPS_PROXY/NO_PROXY in your shell/.env)"
            )
        elif not all_urls:
            c.ok("no llm_base_url set — skipping")
        else:
            c.ok(f"{len(all_urls)} endpoint(s) reachable")
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_model_ids(results: list[CheckResult], m) -> None:
    c = CheckResult("Model IDs match server /v1/models", Severity.CRITICAL)
    try:
        issues: list[str] = []
        cache: dict[str, list[str]] = {}
        api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
        for run in m.contract.runs:
            merged = _resolve_run_overrides(m, run)
            url = merged.get("llm_base_url")
            model_name = merged.get("model_name")
            if not url or not model_name:
                continue
            url = str(url)
            if url not in cache:
                try:
                    req = Request(
                        f"{url}/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    resp = urlopen(req, timeout=15)
                    data = json.loads(resp.read())
                    cache[url] = [obj.get("id", "") for obj in data.get("data", [])]
                except (URLError, OSError) as e:
                    cache[url] = []
                    issues.append(f"{run.label}: {url} unreachable ({e})")
                    continue
            if model_name not in cache[url]:
                issues.append(f"{run.label}: model '{model_name}' not in {cache[url]}")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok()
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_redis_empty(results: list[CheckResult], m) -> None:
    c = CheckResult("Redis DBs empty", Severity.CRITICAL)
    try:
        import redis

        issues: list[str] = []
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        for run in m.contract.runs:
            r = redis.Redis(host=redis_host, port=redis_port, db=run.db)
            size = r.dbsize()
            if size > 0:
                issues.append(f"DB {run.db}: {size} keys (flush first)")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok()
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_db_claims(results: list[CheckResult], m, experiment: str) -> None:
    c = CheckResult("DB claims -- no collision", Severity.CRITICAL)
    try:
        from gigaevo.experiment.manifest import claim_dbs, release_db_claims

        dbs = [run.db for run in m.contract.runs]
        failed = claim_dbs(experiment, dbs)
        if failed:
            still_blocked: list[str] = []
            for db, owner in failed:
                try:
                    owner_m = load_manifest(owner)
                    if owner_m.lifecycle.status == "complete":
                        release_db_claims(owner, [db])
                        re_failed = claim_dbs(experiment, [db])
                        if re_failed:
                            still_blocked.append(
                                f"DB {db}: re-claim failed after releasing {owner}"
                            )
                    else:
                        still_blocked.append(
                            f"DB {db} owned by {owner} (status={owner_m.lifecycle.status})"
                        )
                except Exception as e:
                    still_blocked.append(
                        f"DB {db} owned by {owner} (lookup error: {e})"
                    )
            if still_blocked:
                c.fail("DB collision: " + ", ".join(still_blocked))
            else:
                c.ok(f"Claimed DBs {dbs} (auto-released stale claims)")
        else:
            c.ok(f"Claimed DBs {dbs}")
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_seed_programs(results: list[CheckResult], m) -> None:
    c = CheckResult("Seed programs exist", Severity.CRITICAL)
    try:
        issues: list[str] = []
        seen: set[str] = set()
        for run in m.contract.runs:
            if run.problem_name in seen:
                continue
            seen.add(run.problem_name)
            seed_dir = PROJ / "problems" / run.problem_name / "initial_programs"
            if not seed_dir.exists():
                issues.append(f"{run.problem_name}: no initial_programs/")
            elif not list(seed_dir.glob("*.py")):
                issues.append(f"{run.problem_name}: initial_programs/ has no .py files")
        if issues:
            c.fail("; ".join(issues))
        else:
            c.ok()
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_test_set_sha(results: list[CheckResult], m) -> None:
    c = CheckResult("Test-set SHA-256 matches", Severity.CRITICAL)
    if not m.contract.problem.has_test_set:
        c.ok("N/A (no test set)")
        results.append(c)
        return
    try:
        if m.contract.problem.test_set_path and m.contract.problem.test_set_sha256:
            test_path = PROJ / m.contract.problem.test_set_path
            if test_path.exists():
                actual = hashlib.sha256(test_path.read_bytes()).hexdigest()
                if actual == m.contract.problem.test_set_sha256:
                    c.ok()
                else:
                    c.fail(
                        f"SHA mismatch: expected {m.contract.problem.test_set_sha256[:16]}..., "
                        f"got {actual[:16]}..."
                    )
            else:
                c.fail(f"Test set not found: {test_path}")
        else:
            c.fail("test_set_path or test_set_sha256 not set")
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_smoke_test(results: list[CheckResult], m) -> None:
    c = CheckResult("Smoke test completed", Severity.CRITICAL)
    if m.lifecycle.smoke_test.completed:
        c.ok(f"Completed at {m.lifecycle.smoke_test.completed_at or 'unknown'}")
    else:
        c.fail("smoke_test.completed is false")
    results.append(c)


def _check_treatment_verification(results: list[CheckResult], m) -> None:
    c = CheckResult("Treatment verification completed", Severity.CRITICAL)
    tv = m.lifecycle.treatment_verification
    if tv.completed:
        c.ok(f"Completed at {tv.completed_at or 'unknown'}")
    else:
        c.fail("treatment_verification.completed is false/missing")
    results.append(c)


def _check_resolved_config_matches_pinned(
    results: list[CheckResult], m, experiment: str
) -> None:
    """Diff resolved Hydra config against contract/run pinned assertions.

    Runs ``dry_run`` to compile the resolved config per run, then iterates
    over ``contract.config.pinned`` merged with each run's ``pinned`` delta.
    Any missing key or value mismatch is a CRITICAL — the experiment's
    declared contract is not what Hydra will actually produce.
    """
    c = CheckResult("Resolved config matches pinned contract", Severity.CRITICAL)
    contract_pins = dict(m.contract.config.pinned or {})
    runs_with_pins = [r for r in m.contract.runs if r.pinned]
    if not contract_pins and not runs_with_pins:
        c.ok("no pins declared — skipping (add contract.config.pinned to assert)")
        results.append(c)
        return

    try:
        result = dry_run(experiment)
    except Exception as e:  # Hydra compose / subprocess failure
        c.fail(f"dry_run failed: {e}")
        results.append(c)
        return

    violations: list[str] = []
    for run in m.contract.runs:
        resolved = result.resolved.get(run.label, {})
        pins = {**contract_pins, **(run.pinned or {})}
        for path, expected in pins.items():
            actual = _lookup_dotted(resolved, path)
            if actual is _MISSING:
                violations.append(f"{run.label}: '{path}' not in resolved config")
                continue
            if not _values_equal(actual, expected):
                violations.append(
                    f"{run.label}: {path} = {actual!r} (pinned {expected!r})"
                )

    if violations:
        c.fail("; ".join(violations))
    else:
        total = sum(len({**contract_pins, **(r.pinned or {})}) for r in m.contract.runs)
        c.ok(f"{total} pin assertion(s) satisfied across {len(m.contract.runs)} run(s)")
    results.append(c)


def _check_config_fingerprint_stable(
    results: list[CheckResult], m, experiment: str
) -> None:
    """Compare current Hydra config file digests against the recorded fingerprint.

    Active only on re-launches (``lifecycle.launch.config_fingerprint`` non-empty).
    Any drift in a hashed file → CRITICAL. Fresh launches pass trivially since
    there is nothing to compare against yet.
    """
    c = CheckResult("Config fingerprint stable (re-launch)", Severity.CRITICAL)
    recorded = dict(m.lifecycle.launch.config_fingerprint or {})
    if not recorded:
        c.ok("fresh launch — no fingerprint recorded yet")
        results.append(c)
        return

    try:
        result = dry_run(experiment)
    except Exception as e:
        c.fail(f"dry_run failed: {e}")
        results.append(c)
        return

    current = result.fingerprint
    drifted: list[str] = []
    missing: list[str] = []
    for path, digest in recorded.items():
        cur = current.get(path)
        if cur is None:
            missing.append(path)
        elif cur != digest:
            drifted.append(f"{path}: {digest[:12]}... -> {cur[:12]}...")

    if drifted or missing:
        parts: list[str] = []
        if drifted:
            parts.append("drift: " + "; ".join(drifted))
        if missing:
            parts.append("missing from current scan: " + ", ".join(missing))
        c.fail(" | ".join(parts))
    else:
        c.ok(f"{len(recorded)} config file(s) unchanged since recorded launch")
    results.append(c)


# Sentinel for missing nested keys (None is a valid pinned value).
_MISSING = object()


def _lookup_dotted(doc: dict, path: str):
    """Follow ``a.b.c`` into a nested dict, returning _MISSING on gap."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _values_equal(a, b) -> bool:
    """Compare with type coercion for int/float so 1 == 1.0."""
    if type(a) is type(b):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


# ---------------------------------------------------------------------------
# Report + CLI entry point
# ---------------------------------------------------------------------------


def report(results: list[CheckResult]) -> int:
    """Print results and return exit code (0 = pass, 1 = critical failures)."""
    print(f"\n{'=' * 60}")
    print("  Preflight Check Results")
    print(f"{'=' * 60}\n")

    for r in results:
        print(r)

    criticals = [r for r in results if r.is_blocking]
    passed = [r for r in results if r.passed]
    warns = [r for r in results if not r.passed and r.severity == Severity.WARN]

    print(f"\n  {len(passed)} passed, {len(criticals)} CRITICAL, {len(warns)} WARN")

    if criticals:
        print(
            f"\n  BLOCKED -- {len(criticals)} critical failure(s). Fix before launch."
        )
        return 1
    print("\n  ALL CLEAR -- safe to launch.")
    return 0
