"""UMAP 2x2 panel: qualitative Cohn-vs-champion structure on extreme-gain codes.

For each of 4 selected (d, N) configs (largest champion improvement by default),
embed the union of the N Cohn unit vectors and the N champion unit vectors with a
single UMAP fit (shared embedding so the two are comparable), then overlay them in
one subplot. Four configs -> a 2x2 palette, mirroring the paper's UMAP figure.

Run with the isolated venv python (has umap-learn):
  /tmp/umap_venv/bin/python make_umap.py \
      --packings squeeze_test/packings_champion.npz \
      --results squeeze_test/squeeze_champion.json \
      --out report/fig_umap_2x2.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import umap  # noqa: E402

_PROB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROB)

_GREEN = "#2ca02c"  # Initial / Cohn (matches paper Fig. 12)
_BLUE = "#1f77b4"  # ImprovEvolve (ours)


def _whitegrid(ax) -> None:
    """seaborn-whitegrid look without the seaborn dependency."""
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bcbcbc")
    ax.tick_params(labelsize=8, color="#bcbcbc")


def _pick_configs(results_json: Path, packings, k: int) -> list[tuple[int, int]]:
    data = json.loads(results_json.read_text())
    rows = [r for r in data["results"] if f"{r['d']}_{r['n']}" in packings.files]
    rows.sort(
        key=lambda r: max(
            0.0, (r["mu_cohn"] - r["mu_best"]) / (abs(r["mu_cohn"]) or 1.0)
        ),
        reverse=True,
    )
    return [(r["d"], r["n"]) for r in rows[:k]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packings", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument(
        "--configs", default=None, help="comma d:n list; default = top-4 gains"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="report/fig_umap_2x2.png")
    args = ap.parse_args()

    import cohn_catalogue as cc

    z = np.load(args.packings)
    if args.configs:
        cfgs = [
            (int(a), int(b)) for a, b in (t.split(":") for t in args.configs.split(","))
        ]
    else:
        cfgs = _pick_configs(Path(args.results), z, 4)
    if not cfgs:
        raise SystemExit("no improved configs with saved packings to embed")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    axes = axes.ravel()
    for idx, (ax, (d, n)) in enumerate(zip(axes, cfgs)):
        key = f"{d}_{n}"
        Xc, mu_c = cc.load_frozen(d, n)
        Xe = np.asarray(z[key], dtype=np.float64)
        mu_e = float(np.max((Xe @ Xe.T)[np.triu_indices(n, 1)]))
        U = np.vstack([Xc, Xe])
        nn = min(15, n - 1)
        emb = umap.UMAP(
            n_neighbors=nn, min_dist=0.1, metric="cosine", random_state=args.seed
        ).fit_transform(U)
        ec, ee = emb[:n], emb[n:]
        _whitegrid(ax)
        ax.scatter(
            ec[:, 0],
            ec[:, 1],
            s=34,
            marker="o",
            c=_GREEN,
            alpha=0.8,
            linewidths=0.4,
            edgecolors="white",
            label="Cohn (initial)",
            zorder=3,
        )
        ax.scatter(
            ee[:, 0],
            ee[:, 1],
            s=34,
            marker="s",
            c=_BLUE,
            alpha=0.8,
            linewidths=0.4,
            edgecolors="white",
            label="ImprovEvolve (ours)",
            zorder=3,
        )
        gain = 100.0 * max(0.0, (mu_c - mu_e) / (abs(mu_c) or 1.0))
        ax.set_title(
            f"(d={d}, N={n})  $\\mu$ {mu_c:.5f}$\\to${mu_e:.5f}  (+{gain:.3f}%)",
            fontsize=10,
        )
        if idx >= 2:
            ax.set_xlabel("UMAP Dimension 1", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("UMAP Dimension 2", fontsize=9)
        ax.legend(
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc",
            loc="lower left",
            fontsize=8,
            markerscale=1.2,
        )
    for ax in axes[len(cfgs) :]:
        ax.axis("off")
    fig.suptitle(
        "UMAP projection of unit vectors: Cohn vs ImprovEvolve champion "
        "(largest-improvement configurations)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("UMAP OK ->", args.out, "configs:", cfgs)


if __name__ == "__main__":
    main()
