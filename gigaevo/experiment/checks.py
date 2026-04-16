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

    # Set NO_PROXY before HTTP checks
    if m.contract.servers:
        existing = os.environ.get("NO_PROXY", "")
        all_ips = ",".join(m.contract.servers)
        os.environ["NO_PROXY"] = f"{existing},{all_ips}" if existing else all_ips
        os.environ["no_proxy"] = os.environ["NO_PROXY"]

    # 2. GIGAEVO_PYTHON
    _check_gigaevo_python(results)

    # 3. Servers reachable
    _check_servers_reachable(results, m)

    # 4. Model IDs match
    _check_model_ids(results, m)

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


def _check_servers_reachable(results: list[CheckResult], m) -> None:
    c = CheckResult("Servers reachable (/v1/models)", Severity.CRITICAL)
    try:
        issues: list[str] = []
        all_urls: set[str] = set()
        for run in m.contract.runs:
            if run.chain_url:
                all_urls.add(run.chain_url)
            if run.mutation_url:
                all_urls.add(run.mutation_url)
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
            c.fail("; ".join(issues))
        else:
            c.ok(f"{len(all_urls)} endpoints reachable")
    except Exception as e:
        c.fail(str(e))
    results.append(c)


def _check_model_ids(results: list[CheckResult], m) -> None:
    c = CheckResult("Model IDs match server /v1/models", Severity.CRITICAL)
    try:
        issues: list[str] = []
        checked_urls: set[str] = set()
        api_key = (m.contract.custom_env or {}).get("OPENAI_API_KEY", "None")
        for run in m.contract.runs:
            url = run.mutation_url
            if not url or url in checked_urls:
                continue
            checked_urls.add(url)
            try:
                req = Request(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp = urlopen(req, timeout=15)
                data = json.loads(resp.read())
                model_ids = [obj.get("id", "") for obj in data.get("data", [])]
                for run2 in m.contract.runs:
                    if run2.mutation_url == url and run2.model_name not in model_ids:
                        issues.append(
                            f"{run2.label}: model '{run2.model_name}' not in {model_ids}"
                        )
            except (URLError, OSError) as e:
                issues.append(f"{url}: unreachable ({e})")
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
                        release_db_claims([db])
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
