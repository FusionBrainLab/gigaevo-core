"""Assemble report/report.tex (a self-contained paper-update report) and leave it
ready for pdflatex. Pulls headline numbers from the full90 squeeze JSONs, turns the
90-row tabular into a multipage longtable, and lays out every figure with captions.

  /tmp/umap_venv/bin/python build_report_pdf.py   # writes report/report.tex
  pdflatex -output-directory report report/report.tex
"""

from __future__ import annotations

import json
from pathlib import Path

_PROB = Path(__file__).resolve().parent
RES = _PROB / "squeeze_test"
REP = _PROB / "report"


def _load(name):
    d = json.loads((RES / f"squeeze_{name}.json").read_text())
    d["by_cfg"] = {(r["d"], r["n"]): r for r in d["results"]}
    return d


def _gain(r):
    return max(0.0, (r["mu_cohn"] - r["mu_best"]) / (abs(r["mu_cohn"]) or 1.0))


def main():
    progs = {n: _load(n) for n in ("E7", "E8", "champion")}
    dims = sorted({d for (d, _) in progs["champion"]["by_cfg"]})

    # per-dimension mean gain % per program
    perdim = {}
    for d in dims:
        row = {}
        for n, p in progs.items():
            rs = [r for (dd, _), r in p["by_cfg"].items() if dd == d]
            row[n] = 100.0 * sum(_gain(r) for r in rs) / len(rs)
        perdim[d] = row

    head = {
        n: (
            100.0 * sum(_gain(r) for r in p["results"]) / len(p["results"]),
            p["improved"],
            p["valid"],
            len(p["results"]),
        )
        for n, p in progs.items()
    }

    # body rows from the generated tabular (between \midrule and the summary line)
    raw = (REP / "table_full90.tex").read_text().splitlines()
    body = [
        ln
        for ln in raw
        if ln.endswith(r"\\")
        and "&" in ln
        and "mathrm" not in ln
        and "multicolumn" not in ln
    ]

    longtable = [
        r"\begin{longtable}{rrrrrr}",
        r"\caption{Full 90-configuration head-to-head: minimal achievable maximum pairwise "
        r"inner product $\mu=\max_{i<j}\langle x_i,x_j\rangle$ (lower is better) under the "
        r"calibrated protocol $P^\star$. \textbf{Bold} = strictly beats the Cohn catalogue. "
        r"Summary row: mean relative improvement over Cohn (\# configs improved).}\\",
        r"\toprule",
        r"$d$ & $N$ & $\mu_{\mathrm{Cohn}}$ & $\mu_{\mathrm{E7}}$ & $\mu_{\mathrm{E8}}$ & "
        r"$\mu_{\textbf{ours}}$ \\",
        r"\midrule \endfirsthead",
        r"\toprule $d$ & $N$ & $\mu_{\mathrm{Cohn}}$ & $\mu_{\mathrm{E7}}$ & $\mu_{\mathrm{E8}}$ & "
        r"$\mu_{\textbf{ours}}$ \\ \midrule \endhead",
        *body,
        r"\midrule",
        r"\multicolumn{3}{r}{\emph{mean rel.\ improvement (\# improved)}} & "
        + rf"{head['E7'][0]:.4f}\% ({head['E7'][1]}/90) & "
        + rf"{head['E8'][0]:.4f}\% ({head['E8'][1]}/90) & "
        + rf"\textbf{{{head['champion'][0]:.4f}\% ({head['champion'][1]}/90)}} \\",
        r"\bottomrule",
        r"\end{longtable}",
    ]

    perdim_tbl = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$d$ & E7 & E8 & \textbf{ours} \\",
        r"\midrule",
    ]
    for d in dims:
        row = perdim[d]
        best = max(row, key=row.get)
        cells = []
        for n in ("E7", "E8", "champion"):
            s = f"{row[n]:.4f}"
            cells.append(rf"\textbf{{{s}}}" if n == best else s)
        perdim_tbl.append(f"{d} & " + " & ".join(cells) + r" \\")
    perdim_tbl += [
        r"\midrule",
        rf"all & {head['E7'][0]:.4f} & {head['E8'][0]:.4f} & "
        rf"\textbf{{{head['champion'][0]:.4f}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    def fig(name, caption, width=0.78):
        if not (REP / name).exists():
            return ""
        return (
            r"\begin{figure}[H]\centering"
            + "\n"
            + rf"\includegraphics[width={width}\linewidth]{{{name}}}"
            + "\n"
            + rf"\caption{{{caption}}}"
            + "\n"
            + r"\end{figure}"
        )

    ratio = head["champion"][0] / max(head["E7"][0], 1e-9)
    figs = "\n".join(
        [
            fig(
                "fig_perdim_gain.png",
                "Per-dimension mean relative improvement over the Cohn catalogue for E7, E8 and our champion (full90 test).",
            ),
            fig(
                "fig_gain_sorted.png",
                "Per-configuration improvement of the champion over Cohn, sorted; bars above zero are configs where we strictly beat the catalogue.",
            ),
            fig(
                "fig_sweep_shape.png",
                "Validation-panel calibration: best-over-seed mean gain as a function of noising steps $M$ and noise schedule. Smaller $M$ (more restarts per wall) wins.",
                0.62,
            ),
        ]
    )
    qual = "\n".join(
        [
            fig(
                "fig_umap_2x2.png",
                r"UMAP (cosine) of the unit vectors for the four configurations with the largest champion improvement: Cohn (green circles) vs.\ ImprovEvolve champion (blue squares), a single shared embedding per panel. The two distributions overlap because both are near-uniform on the same sphere $S^{d-1}$: every point is in fact relocated (mean displacement $\approx 0.3$--$0.6$, up to near-antipodal), but the improvement lives in the fine pairwise-extreme arrangement rather than the marginal density, so the catalogue and the improved code occupy the same embedded region.",
                0.92,
            ),
            fig(
                "fig_tail_16_296.png",
                r"Pairwise inner-product right tail, Cohn vs champion, $(d{=}16,N{=}296)$. Dashed lines mark $\mu=\max_{i<j}\langle x_i,x_j\rangle$.",
                0.62,
            ),
            fig(
                "fig_tail_13_244.png",
                r"Pairwise inner-product right tail, $(d{=}13,N{=}244)$.",
                0.62,
            ),
            fig(
                "fig_tail_13_90.png",
                r"Pairwise inner-product right tail, $(d{=}13,N{=}90)$.",
                0.62,
            ),
            fig(
                "fig_tail_15_380.png",
                r"Pairwise inner-product right tail, $(d{=}15,N{=}380)$.",
                0.62,
            ),
        ]
    )
    doc = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,longtable,graphicx,amsmath,xcolor,float}
\graphicspath{{@@GPATH@@/}}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{\vspace{-2.5em}ImprovEvolve on the Cohn Catalogue: a Calibrated General Spherical-Code Improver}
\author{}\date{}
\begin{document}\maketitle\vspace{-3em}

\section*{Summary}
We evolved a general spherical-code improver (dimensions $d\in[8,16]$) and benchmark it
against the Cohn catalogue and the two prior paper programs (E7, E8). The headline metric is
the mean relative improvement of the minimal achievable maximum pairwise inner product
$\mu=\max_{i<j}\langle x_i,x_j\rangle$ over Cohn, floored at $0$ per configuration:
$\;\%=100\cdot\mathrm{mean}_{\text{configs}}\max\!\big(0,(\mu_{\mathrm{Cohn}}-\mu_{\text{best}})/|\mu_{\mathrm{Cohn}}|\big).$
Parameters were tuned on a 14-configuration validation \emph{panel} (the high-headroom
configs that steered evolution) and then applied unchanged to the full 90-configuration
\emph{test} set, identically for every program (a fair head-to-head).

\textbf{Headline (full90 test, protocol $P^\star$, per-config wall $3540$\,s, best of 3 seeds).}
Our champion reaches \textbf{@@CH@@\%} mean relative improvement over
Cohn (@@CH_IMP@@/90 configs improved, @@CH_VALID@@/90 valid),
\textbf{@@RATIO@@$\times$ the best prior program} (E7 @@E7@@\%, E8 @@E8@@\%).
The advantage concentrates in the high-headroom dimensions $d=13$ and $d=16$.

\section*{Calibration (validation panel)}
\textbf{Metric replication.} Under the canonical grader (\texttt{validate.py}; single monotone
$L$-BFGS walk, $R{=}1$, $B{=}10$, seed 42) at the evolution's true per-config budget
($3540$\,s) the champion scores $0.4645\%$ on the panel --- reproducing the plotted $\sim$0.46.
Fitness is wall-limited and monotone in compute ($0.4125\%$ at $600$\,s $\to 0.4645\%$ at $3540$\,s).

\textbf{Protocol sweep.} Sweeping the basin-hopping knobs on the panel
(noising steps $M$, noise schedule $\sigma_{\max}{\to}\sigma_{\min}$, fresh-restart period,
restarts) selects $P^\star$: \emph{unbounded restarts, $M{=}10$, $\sigma\!:\!1{\to}10^{-6}$,
fresh every 5}. Fewer noising steps per restart let more restarts fit a fixed wall, yielding
more basin hops; the fresh-restart period is irrelevant; high base-seed variance at short walls
collapses as the wall grows.

\textbf{Beating canonical at equal budget.} At an identical $3540$\,s per-config wall on the
panel, $P^\star$ (restart-heavy) scores $\mathbf{0.4797\%}$ vs the canonical single-chain
$0.4645\%$ --- a $+3.3\%$ relative gain purely from the search protocol.

\section*{Per-dimension improvement (full90, \% over Cohn)}
\begin{center}@@PERDIM@@\end{center}

@@FIGS@@

\section*{Qualitative structure (extreme-gain configurations)}
@@QUAL@@

\clearpage
\section*{Full 90-configuration table}
{\footnotesize
@@LONGTABLE@@
}

\end{document}
"""
    doc = (
        doc.replace("@@GPATH@@", str(REP))
        .replace("@@CH@@", f"{head['champion'][0]:.4f}")
        .replace("@@CH_IMP@@", str(head["champion"][1]))
        .replace("@@CH_VALID@@", str(head["champion"][2]))
        .replace("@@RATIO@@", f"{ratio:.2f}")
        .replace("@@E7@@", f"{head['E7'][0]:.4f}")
        .replace("@@E8@@", f"{head['E8'][0]:.4f}")
        .replace("@@PERDIM@@", "\n".join(perdim_tbl))
        .replace("@@FIGS@@", figs)
        .replace("@@QUAL@@", qual)
        .replace("@@LONGTABLE@@", "\n".join(longtable))
    )
    (REP / "report.tex").write_text(doc)
    print("wrote", REP / "report.tex")
    print(
        f"headline: ours {head['champion'][0]:.4f}%  E7 {head['E7'][0]:.4f}%  E8 {head['E8'][0]:.4f}%  ({ratio:.2f}x)"
    )


if __name__ == "__main__":
    main()
