"""Cohn conjecture campaign: the calibrated champion at protocol P* over the
noteworthy (d, N) cases supplied by Henry Cohn (email, 2026-07) — cases with a
conjectured exact optimal inner product and no known proof of optimality. Any
strict improvement of mu below the catalogue value is a candidate disproof of
that conjecture; near-exact matches are independent evidence the conjecture is
sharp. Protocol and knobs are identical to the paper's P* re-blast (reblast.py):
R unbounded, M=10, sigma 1->1e-6, fresh=5, dry_patience=0, 3540 s wall,
best-of-3 seeds.

  fetch              -> politely pre-download all case configs into cohn_cache/
  run   --shard i/N  -> grade shard i of N at P*, write conjecture_raw/shard_iofN.json
  merge              -> best-of-seeds per case, margin bands, candidate disproofs,
                        write conjecture_results/{results.json,packings_champion.npz}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

_PROB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROB)

from squeeze_eval import _cost, _init_worker, _squeeze_one  # noqa: E402

CHAMPION = os.path.join(_PROB, "initial_programs", "champion_gen11_199c6110.py")
CASES_JSON = Path(_PROB) / "cohn_conjecture_cases.json"
RAW = Path(_PROB) / "conjecture_raw"
OUT = Path(_PROB) / "conjecture_results"

# relative-gain bands: below NOISE is numerical jitter, not a candidate disproof
NOISE, WEAK, STRONG = 1e-9, 1e-6, 1e-4


def _cases(a) -> list[tuple[int, int]]:
    data = json.loads(Path(a.cases).read_text())
    return [tuple(x) for x in data["cases"]]


def cmd_fetch(a):
    import cohn_catalogue as cc

    cases = _cases(a)
    misses = [
        (d, n) for d, n in cases if not (cc.CACHE_DIR / f"packing_{d}_{n}.txt").exists()
    ]
    print(f"fetch: {len(cases)} cases, {len(misses)} not cached", flush=True)
    for i, (d, n) in enumerate(misses, 1):
        for attempt in range(3):
            try:
                cc.load(d, n)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"  ({d},{n}) FAILED: {e}", flush=True)
                else:
                    time.sleep(5.0 * (attempt + 1))
        if i % 25 == 0 or i == len(misses):
            print(f"  [{i}/{len(misses)}]", flush=True)
        time.sleep(a.delay)


def _run_task(task):
    res = _squeeze_one(task)
    res["seed"] = task[-1]
    return res


def cmd_run(a):
    cases = _cases(a)
    seeds = [int(s) for s in a.seeds.split(",")]
    tasks = [
        (
            "champion",
            os.path.abspath(a.program),
            d,
            n,
            a.restarts,
            a.b_steps,
            a.hi,
            a.lo,
            a.wall,
            1e-9,
            a.fresh_every,
            a.dry_patience,
            sd,
        )
        for d, n in cases
        for sd in seeds
    ]
    tasks.sort(key=lambda t: _cost(t[2], t[3]), reverse=True)  # long poles first
    si, sn = (int(x) for x in a.shard.split("/"))
    tasks = [t for i, t in enumerate(tasks) if i % sn == si]
    RAW.mkdir(parents=True, exist_ok=True)

    print(
        f"conjecture run shard {si}/{sn}: {len(tasks)} tasks "
        f"({len(cases)} cases x {len(seeds)} seeds, this shard) | {a.workers}w | "
        f"wall<={a.wall:.0f}s R<={a.restarts} M={a.b_steps} sigma {a.hi:g}->{a.lo:g} "
        f"fresh={a.fresh_every} dry={a.dry_patience}",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results, done, improved = [], 0, 0
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(1,)) as pool:
        for res in pool.imap_unordered(_run_task, tasks):
            res.pop("chain_bests", None)
            results.append(res)
            done += 1
            if res["improved"]:
                improved += 1
                print(
                    f"  IMPROVED ({res['d']},{res['n']}) seed {res['seed']}: "
                    f"mu {res['mu_cohn']:.12f} -> {res['mu_best']:.12f} "
                    f"(rel {res['gain']:.3e})",
                    flush=True,
                )
            if done % 20 == 0 or done == len(tasks):
                print(
                    f"  [{done}/{len(tasks)}] improved-tasks={improved} "
                    f"{(time.perf_counter() - t0) / 60:.1f} min",
                    flush=True,
                )

    out = RAW / f"shard_{si}of{sn}.json"
    out.write_text(json.dumps({"shard": a.shard, "seeds": seeds, "results": results}))
    print(f"wrote {out}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)


def cmd_merge(a):
    cases = _cases(a)
    raw = []
    for p in sorted(RAW.glob("shard_*.json")):
        raw.extend(json.loads(p.read_text())["results"])
    if not raw:
        raise SystemExit("no conjecture_raw/shard_*.json to merge")

    best: dict[tuple, dict] = {}
    for r in raw:
        k = (r["d"], r["n"])
        if k not in best or r["mu_best"] < best[k]["mu_best"]:
            best[k] = r

    missing = [c for c in cases if c not in best]
    if missing:
        print(f"WARNING: {len(missing)} cases have no results yet: {missing[:10]}...")

    rows, packs = [], {}
    for d, n in sorted(best):
        r = dict(best[(d, n)])
        bx = r.pop("best_X", None)
        if r["improved"] and bx is not None:
            packs[f"{d}_{n}"] = np.asarray(bx, dtype=np.float64)
        r["abs_margin"] = r["mu_cohn"] - r["mu_best"]
        rows.append(r)

    def band(r):
        g = r["gain"]
        if g >= STRONG:
            return "strong"
        if g >= WEAK:
            return "moderate"
        if g >= NOISE:
            return "weak"
        return "none"

    counts = {
        b: sum(1 for r in rows if band(r) == b)
        for b in ("strong", "moderate", "weak", "none")
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "protocol": "P* wall=3540s best-of-seeds",
                "bands": counts,
                "results": rows,
            }
        )
    )
    np.savez(OUT / "packings_champion.npz", **packs)

    print(f"cases graded: {len(rows)}/{len(cases)}   bands: {counts}", flush=True)
    print(f"saved {len(packs)} improved packings -> {OUT / 'packings_champion.npz'}")
    cand = sorted((r for r in rows if r["gain"] >= WEAK), key=lambda r: -r["gain"])
    if not cand:
        print("no candidate disproofs (all gains below the weak band)")
    for r in cand[:30]:
        print(
            f"  ({r['d']:>2},{r['n']:>4}) mu {r['mu_cohn']:.12f} -> {r['mu_best']:.12f} "
            f"abs {r['abs_margin']:.3e} rel {r['gain']:.3e} [{band(r)}]",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASES_JSON))
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--delay", type=float, default=1.0)
    f.set_defaults(fn=cmd_fetch)

    r = sub.add_parser("run")
    r.add_argument("--program", default=CHAMPION)
    r.add_argument("--wall", type=float, default=3540.0)
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--workers", type=int, default=160)
    r.add_argument("--restarts", type=int, default=100000)
    r.add_argument("--b-steps", type=int, default=10)
    r.add_argument("--hi", type=float, default=1.0)
    r.add_argument("--lo", type=float, default=1e-6)
    r.add_argument("--fresh-every", type=int, default=5)
    r.add_argument("--dry-patience", type=int, default=0)
    r.add_argument("--shard", default="0/1")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("merge")
    m.set_defaults(fn=cmd_merge)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
