#!/usr/bin/env python3
"""
Analyze push experiment test eval results.

Computes binomial 95% CIs, gate verdicts, and cross-run comparisons from
experiments/hotpotqa/push/test_evals/results.json.

Usage:
    PYTHONPATH=. python tools/analyze_test_results.py \
        experiments/hotpotqa/push/test_evals/results.json
"""

import argparse
import json
import math
from pathlib import Path
import sys

# ── Binomial CI (Wilson score interval) ──────────────────────────────────────


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Gate verdicts ─────────────────────────────────────────────────────────────


def verdict_gepa(test_em: float, val_test_gap: float | None) -> str:
    """GEPA threshold verdict (pre-registered in 01_design.md §8)."""
    if test_em >= 0.630:
        return "STRONG POSITIVE"
    if test_em >= 0.623:
        return "POSITIVE"
    if test_em >= 0.615:
        gap_ok = val_test_gap is not None and val_test_gap <= 0.015
        return "SUGGESTIVE" if gap_ok else "NULL (gap too large for SUGGESTIVE)"
    return "NULL"


def verdict_gate_c(test_em_b: float, gap_b: float, gap_o: float = 0.007) -> str:
    """Gate C: Run B gap reduction vs. Run O (gap_O ~0.7pp)."""
    delta = gap_o - gap_b
    if delta >= 0.020 and test_em_b >= 0.600:
        return "POSITIVE"
    if 0.010 <= delta < 0.020 and test_em_b >= 0.600:
        return "SUGGESTIVE"
    if abs(delta) < 0.010:
        return "NULL"
    if gap_b > gap_o:
        return "NEGATIVE (gap_B > gap_O)"
    return "NULL"


def verdict_run_a(test_em_a: float, ref: float = 0.6167) -> str:
    """Run A vs. Run F reference (F1+default+300, test EM = 61.67%)."""
    if test_em_a >= 0.623:
        return "POSITIVE"
    if test_em_a >= ref:
        return "SUGGESTIVE"
    if test_em_a >= 0.610:
        return "NULL"
    return "NEGATIVE (NLP prompts harm F1 runs)"


# ── McNemar's test ────────────────────────────────────────────────────────────


def mcnemar(correct_a: list[int], correct_b: list[int]) -> tuple[float, float]:
    """McNemar's test comparing two binary-correct arrays on the same test set.

    Returns (chi2_statistic, p_value). Uses continuity correction (b+c >= 25 recommended).
    """
    assert len(correct_a) == len(correct_b), "Arrays must have equal length"
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a == 0 and bb == 1)
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    # With continuity correction (Yates)
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Approximate p-value from chi2 distribution (df=1) using survival function
    p = _chi2_sf(chi2, df=1)
    return (chi2, p)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of chi2 distribution. Approximate via regularized gamma."""
    # For df=1: P(chi2 > x) = erfc(sqrt(x/2))
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    # Fallback: not implemented for df != 1
    raise NotImplementedError(f"Only df=1 supported, got df={df}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", help="Path to test_evals/results.json")
    args = parser.parse_args()

    path = Path(args.results_json)
    if not path.exists():
        print(f"ERROR: {path} not found — run run_test_eval.sh first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    runs = {
        k: v
        for k, v in data.items()
        if k not in ("preregistration_commit", "evaluation_date_utc")
    }

    # ── Per-run summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print("PUSH EXPERIMENT — TEST EVAL ANALYSIS")
    print("=" * 70)
    print(f"Results file: {path}")
    print(f"Pre-registration commit: {data.get('preregistration_commit', 'unknown')}")
    print()

    run_data = {}
    for label in sorted(runs):
        r = runs[label]
        n = r["n_test_samples"]
        k = round(r["test_em"] * n)
        lo, hi = wilson_ci(k, n)
        val_em = r.get("val_em")
        gap = r.get("val_test_gap")

        run_data[label] = r
        run_data[label]["_ci_lo"] = lo
        run_data[label]["_ci_hi"] = hi

        print(f"Run {label}:")
        print(f"  condition     : {r.get('chain_url', '?')}")
        print(
            f"  program_id    : {r.get('program_id', '?')}  (iteration {r.get('iteration', '?')})"
        )
        print(
            f"  val_EM        : {val_em * 100:.2f}%"
            if val_em is not None
            else "  val_EM        : N/A"
        )
        print(f"  test_EM       : {r['test_em'] * 100:.2f}%  ({k}/{n})")
        print(f"  95% CI        : [{lo * 100:.2f}%, {hi * 100:.2f}%]")
        print(
            f"  val-test gap  : {gap * 100:+.2f}pp"
            if gap is not None
            else "  val-test gap  : N/A"
        )
        print(f"  fail rate     : {r.get('extraction_failure_rate', 0) * 100:.1f}%")
        print()

    # ── Gate verdicts ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("GATE VERDICTS")
    print("=" * 70)

    if "C" in run_data:
        r = run_data["C"]
        verdict = verdict_gepa(r["test_em"], r.get("val_test_gap"))
        print(f"[PRIMARY] Run C GEPA gate: {verdict}")
        print(f"  test EM = {r['test_em'] * 100:.2f}%  (GEPA = 62.3%)")
        if r.get("val_test_gap") is not None:
            print(
                f"  val-test gap = {r['val_test_gap'] * 100:+.2f}pp  (threshold for SUGGESTIVE: < 1.5pp)"
            )
    else:
        print("[PRIMARY] Run C: not in results.json")

    print()

    if "B" in run_data:
        r = run_data["B"]
        gap_b = r.get("val_test_gap", 0.0)
        verdict = verdict_gate_c(r["test_em"], gap_b)
        print(f"Gate C (Run B, EM+600 gap reduction): {verdict}")
        print(
            f"  test EM = {r['test_em'] * 100:.2f}%  gap_B = {gap_b * 100:+.2f}pp  gap_O ~0.70pp"
        )
    else:
        print("Gate C (Run B): not in results.json")

    print()

    if "A" in run_data:
        r = run_data["A"]
        verdict = verdict_run_a(r["test_em"])
        print(f"Run A (F1+NLP+300 vs. Run F = 61.67%): {verdict}")
        print(f"  test EM = {r['test_em'] * 100:.2f}%  Run F = 61.67%")
    else:
        print("Run A: not in results.json")

    print()

    if "D" in run_data:
        r = run_data["D"]
        delta_dc = (
            (r["test_em"] - run_data["C"]["test_em"]) * 100 if "C" in run_data else None
        )
        delta_da = (
            (r["test_em"] - run_data["A"]["test_em"]) * 100 if "A" in run_data else None
        )
        print(f"Run D (F1+NLP+600, Amendment 3): test EM = {r['test_em'] * 100:.2f}%")
        if delta_dc is not None:
            print(f"  D - C (NLP effect at 600/F1) = {delta_dc:+.2f}pp")
        if delta_da is not None:
            print(f"  D - A (val-N effect for F1+NLP) = {delta_da:+.2f}pp")
    else:
        print("Run D: not in results.json")

    # ── McNemar pairwise ──────────────────────────────────────────────────────
    pairs_with_correct = [
        (lx, ly)
        for lx in run_data
        for ly in run_data
        if lx < ly
        and "per_sample_correct" in run_data[lx]
        and "per_sample_correct" in run_data[ly]
        and len(run_data[lx]["per_sample_correct"])
        == len(run_data[ly]["per_sample_correct"])
    ]

    if pairs_with_correct:
        print()
        print("=" * 70)
        print("MCNEMAR PAIRWISE TESTS (per_sample_correct, continuity-corrected)")
        print(
            "Note: observational only — experiment not powered for pairwise inference"
        )
        print("=" * 70)
        for lx, ly in pairs_with_correct:
            cx = run_data[lx]["per_sample_correct"]
            cy = run_data[ly]["per_sample_correct"]
            chi2, p = mcnemar(cx, cy)
            sig = " *" if p < 0.05 else ""
            print(f"  {lx} vs {ly}: chi2={chi2:.2f}  p={p:.3f}{sig}")

    print()
    print("=" * 70)
    print("Binomial CIs use Wilson score interval (z=1.96, N=300).")
    print("All verdicts per pre-registration 01_design.md §8.")
    print("=" * 70)


if __name__ == "__main__":
    main()
