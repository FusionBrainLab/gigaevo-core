"""
Watchdog for adversarial co-evolution pilot — monitors both populations
and posts status + fitness plots to PR #169.

Pop A (optimizer, DB 1) vs Pop B (deceptive landscape, DB 2)
Synchronized via MainRunSyncHook (lockstep generation advancement).

Usage:
    export PYTHONPATH="$PROJ"
    nohup python problems/adversarial/run_watchdog.py \
        > /tmp/adversarial_optimizer/watchdog.log 2>&1 &
"""

import os

_SERVERS = [
    "INTERNAL_IP",
    "INTERNAL_IP",
    "INTERNAL_IP",
    "INTERNAL_IP",
    "INTERNAL_IP",
    "INTERNAL_IP",
    "api.github.com",
]
_existing = os.environ.get("NO_PROXY", "")
for _s in _SERVERS:
    if _s not in _existing:
        _existing = ",".join(filter(None, [_existing, _s]))
os.environ["NO_PROXY"] = _existing
os.environ["no_proxy"] = _existing

import base64  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import urllib.request  # noqa: E402

import redis  # noqa: E402

PROJ = Path(__file__).parent.parent.parent
PR_NUMBER = "169"
REPO = "KhrulkovV/gigaevo-core-internal"
PYTHON = sys.executable

RUNS = [
    {
        "label": "A",
        "role": "optimizer",
        "db": 1,
        "prefix": "adversarial/optimizer/pop_a",
        "max_gen": 5,
    },
    {
        "label": "B",
        "role": "landscape",
        "db": 2,
        "prefix": "adversarial/optimizer/pop_b",
        "max_gen": 5,
    },
]

_last_gen: dict[str, int] = {run["label"]: -1 for run in RUNS}


def log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{ts}] {msg}", flush=True)


def get_generation(db: int, prefix: str) -> int:
    try:
        r = redis.Redis(host="localhost", port=6379, db=db)
        raw = r.hget(f"{prefix}:run_state", "engine:total_generations")
        if raw:
            return int(raw)
    except Exception:
        pass
    return 0


def get_redis_metric(db: int, prefix: str, metric_suffix: str) -> float | None:
    try:
        r = redis.Redis(host="localhost", port=6379, db=db)
        key = f"{prefix}:metrics:history:program_metrics:{metric_suffix}"
        if r.type(key) == b"list":
            raw = r.lindex(key, -1)
            if raw:
                return json.loads(raw)["v"]
    except Exception:
        pass
    return None


def get_archive_size(db: int, prefix: str) -> int:
    try:
        r = redis.Redis(host="localhost", port=6379, db=db)
        return r.hlen(f"{prefix}:archive")
    except Exception:
        return 0


def _get_token() -> str | None:
    try:
        token_file = Path.home() / ".config/gh/hosts.yml"
        text = token_file.read_text()
        m = re.search(r"oauth_token:\s*(\S+)", text)
        return m.group(1) if m else None
    except Exception:
        return None


def generate_plot(check_num: int) -> Path | None:
    plot_dir = PROJ / "problems/adversarial/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_args = []
    for run in RUNS:
        run_args.extend(
            ["--run", f"{run['prefix']}@{run['db']}:{run['label']} ({run['role']})"]
        )
    try:
        subprocess.run(
            [
                PYTHON,
                str(PROJ / "tools/comparison.py"),
                *run_args,
                "--annotate-frontier",
                "--output-folder",
                str(plot_dir),
                "--title",
                "Adversarial Co-Evolution: Optimizer vs Landscape",
            ],
            cwd=str(PROJ),
            env={**os.environ, "PYTHONPATH": str(PROJ)},
            capture_output=True,
            timeout=120,
            check=True,
        )
        png = plot_dir / "evolution_runs_comparison.png"
        if png.exists():
            stamped = plot_dir / f"comparison_check_{check_num:03d}.png"
            shutil.copy2(png, stamped)
            log(f"Plot generated: {stamped.name}")
            return stamped
    except Exception as e:
        log(f"Plot generation failed: {e}")
    return None


def upload_plot_to_github(png_path: Path, token: str, check_num: int) -> str | None:
    PUBLIC_REPO = "KhrulkovV/gigaevo-plots"
    repo_path = f"adversarial-optimizer/{png_path.name}"
    api_url = f"https://api.github.com/repos/{PUBLIC_REPO}/contents/{repo_path}"
    content_b64 = base64.b64encode(png_path.read_bytes()).decode()
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    sha = None
    try:
        req = urllib.request.Request(
            api_url, headers={"Authorization": f"token {token}"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            sha = json.loads(resp.read())["sha"]
    except Exception:
        pass

    payload: dict = {
        "message": f"chore: adversarial pilot watchdog plot check {check_num:03d} — {now}",
        "content": content_b64,
    }
    if sha:
        payload["sha"] = sha

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        raw_url = f"https://raw.githubusercontent.com/{PUBLIC_REPO}/main/{repo_path}"
        log(f"Plot uploaded: {raw_url}")
        return raw_url
    except Exception as e:
        log(f"Plot upload failed: {e}")
        return None


def post_pr_comment(body: str) -> None:
    token = _get_token()
    if not token:
        log("No GitHub token found — skipping PR comment")
        return
    payload = json.dumps({"body": body}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/issues/{PR_NUMBER}/comments",
        data=payload,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        log("PR comment posted")
    except Exception as e:
        log(f"PR post failed: {e}")


def check_processes_alive() -> dict[str, bool]:
    """Check if Pop A and Pop B processes are still running."""
    alive = {}
    try:
        result = subprocess.run(
            ["pgrep", "-f", "adversarial/optimizer/pop_a"],
            capture_output=True,
            timeout=5,
        )
        alive["A"] = result.returncode == 0
    except Exception:
        alive["A"] = False
    try:
        result = subprocess.run(
            ["pgrep", "-f", "adversarial/optimizer/pop_b"],
            capture_output=True,
            timeout=5,
        )
        alive["B"] = result.returncode == 0
    except Exception:
        alive["B"] = False
    return alive


def make_status_body(
    run_states: list[dict], check_num: int, plot_url: str | None
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = ""
    for s in run_states:
        fit_str = f"{s['fitness'] * 100:.1f}%" if s["fitness"] is not None else "—"
        status = "ALIVE" if s["alive"] else "DEAD"
        if s["stalled"] and s["alive"]:
            status = "STALLED"
        if s["gen"] >= s["max_gen"]:
            status = "COMPLETE"
        archive = s["archive_size"]
        rows += (
            f"| {s['label']} ({s['role']}) "
            f"| {s['gen']} / {s['max_gen']} "
            f"| {fit_str} "
            f"| {archive} "
            f"| {status} |\n"
        )

    # Arms race summary
    fitnesses = {
        s["label"]: s["fitness"] for s in run_states if s["fitness"] is not None
    }
    arms_block = ""
    if "A" in fitnesses and "B" in fitnesses:
        a_fit = fitnesses["A"] * 100
        b_fit = fitnesses["B"] * 100
        arms_block = (
            f"\n**Arms race:** Optimizer fitness {a_fit:.1f}% vs "
            f"Landscape deceptiveness {b_fit:.1f}%\n"
            f"*(Healthy co-evolution: both should increase over time)*\n"
        )

    alerts = []
    for s in run_states:
        if s["stalled"] and s["alive"]:
            alerts.append(f"- Run {s['label']} stalled at gen={s['gen']}")
        if not s["alive"] and s["gen"] < s["max_gen"]:
            alerts.append(f"- Run {s['label']} process DEAD at gen={s['gen']}")
    alert_block = ""
    if alerts:
        alert_block = "\n**ALERTS:**\n" + "\n".join(alerts) + "\n"

    plot_block = (
        f"\n![Fitness curves check {check_num}]({plot_url})\n"
        if plot_url
        else "\n*(plot unavailable)*\n"
    )

    all_complete = all(s["gen"] >= s["max_gen"] for s in run_states)
    footer = (
        "*All runs complete!*"
        if all_complete
        else "*Posted by adversarial watchdog — monitoring every 30 min*"
    )

    return f"""### Adversarial Co-Evolution Status #{check_num} — {now}

| Population | Progress | Fitness | Archive | Status |
|-----------|----------|---------|---------|--------|
{rows}
> **Pop A** (optimizer): fitness = proximity to global optimum on adversarial landscapes
> **Pop B** (landscape): fitness = deceptiveness (how far optimizers land from global optimum)
{arms_block}{alert_block}{plot_block}
{footer}"""


def run_check(check_num: int) -> bool:
    """Run one status check. Returns True if all runs complete."""
    alive = check_processes_alive()

    run_states = []
    for run in RUNS:
        gen = get_generation(run["db"], run["prefix"])
        fitness = get_redis_metric(run["db"], run["prefix"], "valid_frontier_fitness")
        archive = get_archive_size(run["db"], run["prefix"])
        stalled = (gen == _last_gen[run["label"]]) and gen > 0
        _last_gen[run["label"]] = gen

        run_states.append(
            {
                "label": run["label"],
                "role": run["role"],
                "max_gen": run["max_gen"],
                "gen": gen,
                "fitness": fitness,
                "archive_size": archive,
                "stalled": stalled,
                "alive": alive.get(run["label"], False),
            }
        )
        log(
            f"Run {run['label']} ({run['role']}): gen={gen}/{run['max_gen']}, "
            f"fitness={f'{fitness:.3f}' if fitness is not None else 'N/A'}, "
            f"archive={archive}, alive={alive.get(run['label'], False)}"
        )

    token = _get_token()
    plot_url = None
    if token:
        png_path = generate_plot(check_num)
        if png_path:
            plot_url = upload_plot_to_github(png_path, token, check_num)

    post_pr_comment(make_status_body(run_states, check_num, plot_url))

    return all(s["gen"] >= s["max_gen"] for s in run_states)


if __name__ == "__main__":
    log(f"Adversarial watchdog started — PR #{PR_NUMBER}")
    check_num = 0

    import signal

    _shutdown = False

    def _sigterm(signum, frame):
        global _shutdown
        _shutdown = True
        log("SIGTERM received — shutting down")

    signal.signal(signal.SIGTERM, _sigterm)

    while not _shutdown:
        check_num += 1
        try:
            complete = run_check(check_num)
            if complete:
                log("All runs complete — exiting")
                break
        except Exception as e:
            log(f"Check failed: {e}")

        # Sleep in small intervals to allow SIGTERM handling
        for _ in range(1800 // 10):  # 30 min in 10s chunks
            if _shutdown:
                break
            import time

            time.sleep(10)

    log("Watchdog exiting")
