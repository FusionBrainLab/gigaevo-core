"""Dump raw point configurations (unit-vector matrices) for every (d, N), one
artifact per program, for downstream visualization done separately.

Reads the best-of-K winner per (program, config) from reblast_raw/shard_*.json
(headroom configs) and falls back to the frozen Cohn config for the proven-
optimal ("dead") configs and for any config a program failed to improve.

Outputs under configs_raw/:
  cohn.npz       key "{d}_{n}" -> (N, d) Cohn baseline unit vectors
  champion.npz   key "{d}_{n}" -> (N, d) champion final config (best-of-K)
  E7.npz, E8.npz same, for the paper programs
  index.json     per-config mu_cohn and per-program mu_best + gain%

Run after the re-blast completes:
  python dump_configs.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

_PROB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROB)
import cohn_catalogue as cc  # noqa: E402

PROGRAMS = ("champion", "E7", "E8")
RAW = Path(_PROB) / "reblast_raw"
OUT = Path(_PROB) / "configs_raw"


def main() -> None:
    part = json.loads(Path("/tmp/headroom61.json").read_text())
    all_cfgs = sorted(
        {tuple(x) for x in part["headroom"]} | {tuple(x) for x in part["dead"]}
    )

    raw = []
    for p in sorted(RAW.glob("shard_*.json")):
        raw.extend(json.loads(p.read_text())["results"])
    if not raw:
        raise SystemExit("no reblast_raw/shard_*.json found")

    # best-of-K winner per (program, config): lowest mu_best, keep its best_X
    best: dict[tuple, dict] = {}
    for r in raw:
        k = (r["name"], r["d"], r["n"])
        if k not in best or r["mu_best"] < best[k]["mu_best"]:
            best[k] = r

    OUT.mkdir(parents=True, exist_ok=True)
    cohn_packs: dict[str, np.ndarray] = {}
    prog_packs: dict[str, dict[str, np.ndarray]] = {p: {} for p in PROGRAMS}
    index: dict[str, dict] = {}

    for d, n in all_cfgs:
        key = f"{d}_{n}"
        Xc, mu_c = cc.load_frozen(d, n)
        cohn_packs[key] = np.asarray(Xc, dtype=np.float64)
        index[key] = {"d": d, "n": n, "mu_cohn": float(mu_c)}
        for p in PROGRAMS:
            rec = best.get((p, d, n))
            if rec is not None and rec.get("best_X") is not None and rec["improved"]:
                X = np.asarray(rec["best_X"], dtype=np.float64)
                mu = float(rec["mu_best"])
            else:
                X, mu = (
                    np.asarray(Xc, dtype=np.float64),
                    float(mu_c),
                )  # no improvement -> Cohn
            prog_packs[p][key] = X
            gain = 100.0 * max(0.0, (mu_c - mu) / (abs(mu_c) or 1.0))
            index[key][p] = {
                "mu_best": mu,
                "gain_pct": gain,
                "improved": mu < mu_c - 1e-15,
            }

    np.savez(OUT / "cohn.npz", **cohn_packs)
    for p in PROGRAMS:
        np.savez(OUT / f"{p}.npz", **prog_packs[p])
    (OUT / "index.json").write_text(json.dumps(index, indent=2))

    n_imp = {p: sum(1 for k in index if index[k][p]["improved"]) for p in PROGRAMS}
    print(f"wrote {len(all_cfgs)} configs/program to {OUT}")
    print(f"  cohn.npz + {', '.join(p + '.npz' for p in PROGRAMS)} + index.json")
    for p in PROGRAMS:
        print(f"  {p}: improved {n_imp[p]}/{len(all_cfgs)}")
    print("each .npz: key '{d}_{n}' -> (N, d) float64 unit vectors")


if __name__ == "__main__":
    main()
