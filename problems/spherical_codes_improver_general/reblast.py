"""Best-of-K-seed max-effort re-blast of the headroom configs at the calibrated
protocol P* (R unbounded, M=10, sigma 1->1e-6, fresh=5, dry_patience=0), at a
larger per-config wall than the original 1800s/1-seed test blast.

Only the configs with demonstrated headroom are re-graded; the proven-optimal
("dead") configs are carried forward at Cohn (gain 0 for every program), so the
full90 head-to-head stays complete and fair (identical protocol on every config
that can move).

Two subcommands so the same harness fans across NFS-mounted machines:

  run   --shard i/N  -> grade shard i of N, write reblast_raw/shard_iofN.json
  merge              -> best-of-K over all raw shards, carry dead configs,
                        write <out-dir>/squeeze_{champion,E7,E8}.json + packings

Single box: `run --shard 0/1` then `merge`.
Two boxes:  `run --shard 0/2` (A) + `run --shard 1/2` (B), then `merge` once both done.
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

PROGRAMS = {
    "champion": "initial_programs/champion_gen11_199c6110.py",
    "E7": "initial_programs/paper_evolved.py",
    "E8": "initial_programs/paper_evolved_plus.py",
}
RAW = Path(_PROB) / "reblast_raw"


def _tasks(head, seeds, a):
    out = []
    for name, path in PROGRAMS.items():
        ap = os.path.abspath(path)
        for d, n in head:
            for sd in seeds:
                out.append(
                    (
                        name,
                        ap,
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
                )
    out.sort(key=lambda t: _cost(t[2], t[3]), reverse=True)  # long poles first
    return out


def cmd_run(a):
    part = json.loads(Path(a.headroom).read_text())
    head = [tuple(x) for x in part["headroom"]]
    seeds = [int(s) for s in a.seeds.split(",")]
    si, sn = (int(x) for x in a.shard.split("/"))
    tasks = [t for i, t in enumerate(_tasks(head, seeds, a)) if i % sn == si]
    RAW.mkdir(parents=True, exist_ok=True)

    print(
        f"reblast run shard {si}/{sn}: {len(tasks)} tasks "
        f"({len(PROGRAMS)} progs x {len(head)} configs x {len(seeds)} seeds, "
        f"this shard) | {a.workers}w | wall<={a.wall:.0f}s "
        f"R<={a.restarts} M={a.b_steps} sigma {a.hi:g}->{a.lo:g} "
        f"fresh={a.fresh_every} dry={a.dry_patience}",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results, done = [], 0
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(1,)) as pool:
        for res in pool.imap_unordered(_squeeze_one, tasks):
            res.pop("chain_bests", None)
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(
                    f"  [{done}/{len(tasks)}] {(time.perf_counter() - t0) / 60:.1f} min",
                    flush=True,
                )

    out = RAW / f"shard_{si}of{sn}.json"
    out.write_text(json.dumps({"shard": a.shard, "seeds": seeds, "results": results}))
    print(f"wrote {out}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)


def cmd_merge(a):
    part = json.loads(Path(a.headroom).read_text())
    head = {tuple(x) for x in part["headroom"]}
    dead = [tuple(x) for x in part["dead"]]
    out_dir = Path(_PROB) / a.out_dir

    raw = []
    for p in sorted(RAW.glob("shard_*.json")):
        raw.extend(json.loads(p.read_text())["results"])
    if not raw:
        raise SystemExit("no reblast_raw/shard_*.json to merge")

    # best-of-K per (program, config): lowest mu_best wins; keep its packing.
    best: dict[tuple, dict] = {}
    for r in raw:
        k = (r["name"], r["d"], r["n"])
        if k not in best or r["mu_best"] < best[k]["mu_best"]:
            best[k] = r

    prev = {
        name: {
            (r["d"], r["n"]): r
            for r in json.loads((out_dir / f"squeeze_{name}.json").read_text())[
                "results"
            ]
        }
        for name in PROGRAMS
    }

    for name in PROGRAMS:
        rows, packs = [], {}
        for d, n in sorted(head | set(dead)):
            if (d, n) in head:
                r = dict(best[(name, d, n)])
            else:
                r = dict(prev[name][(d, n)])  # carry-forward proven-optimal
            bx = r.pop("best_X", None)
            if (
                name == "champion"
                and (d, n) in head
                and r.get("improved")
                and bx is not None
            ):
                packs[f"{d}_{n}"] = np.asarray(bx, dtype=np.float64)
            rows.append(r)
        gains = [
            max(0.0, (r["mu_cohn"] - r["mu_best"]) / (abs(r["mu_cohn"]) or 1.0))
            for r in rows
        ]
        improved = sum(r["mu_best"] < r["mu_cohn"] - 1e-15 for r in rows)
        valid = sum(1 for r in rows)
        payload = {
            "results": rows,
            "improved": improved,
            "valid": valid,
            "mean_gain_pct": 100.0 * float(np.mean(gains)),
        }
        (out_dir / f"squeeze_{name}.json").write_text(json.dumps(payload))
        print(
            f"{name:9s} full90 {100 * np.mean(gains):.4f}%  improved {improved}/{len(rows)}",
            flush=True,
        )
        if name == "champion":
            np.savez(out_dir / "packings_champion.npz", **packs)
            print(f"  saved {len(packs)} champion packings", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--wall", type=float, default=3540.0)
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--workers", type=int, default=160)
    r.add_argument("--restarts", type=int, default=100000)
    r.add_argument("--b-steps", type=int, default=10)
    r.add_argument("--hi", type=float, default=1.0)
    r.add_argument("--lo", type=float, default=1e-6)
    r.add_argument("--fresh-every", type=int, default=5)
    r.add_argument("--dry-patience", type=int, default=0)
    r.add_argument("--headroom", default="/tmp/headroom61.json")
    r.add_argument("--shard", default="0/1")
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("merge")
    m.add_argument("--headroom", default="/tmp/headroom61.json")
    m.add_argument("--out-dir", default="squeeze_test")
    m.set_defaults(fn=cmd_merge)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
