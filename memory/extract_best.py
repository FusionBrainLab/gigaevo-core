#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract top-K program(s) with highest metric_fitness from a CSV and write to a txt file."
    )
    parser.add_argument("--csv_path", default="outputs/circles_26_2_qwen3_gemini_flash.csv", help="Path to input CSV")
    parser.add_argument(
        "-o",
        "--out",
        default="memory/best_programs_26_5.txt",
        help="Output txt path (default: best_programs_heilbron.txt)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="If set, only consider rows where metric_is_valid is truthy.",
    )
    parser.add_argument(
        "--write",
        choices=["code", "full", "id_name_code"],
        default="code",
        help="What to write for each program (default: code).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of top programs to write (default: 5).",
    )
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help="If set, include all rows tied at the cutoff (may write more than top-k).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path)

    required = {"program_id", "name", "code", "metric_fitness", "metric_is_valid"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing required columns: {sorted(missing)}", file=sys.stderr)
        return 2

    # Ensure numeric fitness (coerce bad values to NaN)
    df["metric_fitness"] = pd.to_numeric(df["metric_fitness"], errors="coerce")

    # Optional: filter to valid-only rows
    if args.valid_only:
        # Treat values like True/1/"true"/"True"/"yes" as valid; also allow numeric nonzero.
        valid_str = df["metric_is_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
        valid_num = pd.to_numeric(df["metric_is_valid"], errors="coerce").fillna(0) != 0
        df = df[valid_str | valid_num]

    # Drop rows without fitness
    df = df.dropna(subset=["metric_fitness"])
    if df.empty:
        print("No rows with a usable metric_fitness after filtering.", file=sys.stderr)
        return 1

    top_k = max(1, int(args.top_k))

    # Sort by fitness descending; stable sort keeps input order for equal fitness
    sorted_df = df.sort_values("metric_fitness", ascending=False, kind="mergesort")

    if args.include_ties:
        # Include all rows tied at the cutoff rank
        cutoff_idx = min(top_k, len(sorted_df)) - 1
        cutoff = sorted_df.iloc[cutoff_idx]["metric_fitness"]
        best = sorted_df[sorted_df["metric_fitness"] >= cutoff].copy()
    else:
        # Exactly top_k rows
        best = sorted_df.head(top_k).copy()

    if best.empty:
        print("No rows selected for output.", file=sys.stderr)
        return 1

    best_fit = best["metric_fitness"].max()

    lines = []
    header = (
        f"# Top programs by metric_fitness\n"
        f"# Requested top_k = {top_k}\n"
        f"# include_ties = {bool(args.include_ties)}\n"
        f"# Best metric_fitness in output = {best_fit}\n"
        f"# Num programs written = {len(best)}\n"
    )
    lines.append(header)

    for i, row in best.reset_index(drop=True).iterrows():
        lines.append(f"\n=== PROGRAM {i+1}/{len(best)} ===\n")
        if args.write == "code":
            lines.append(str(row["code"]) + "\n")
        elif args.write == "id_name_code":
            lines.append(f"program_id: {row['program_id']}\n")
            lines.append(f"name: {row['name']}\n")
            lines.append(f"metric_fitness: {row['metric_fitness']}\n")
            lines.append("code:\n")
            lines.append(str(row["code"]) + "\n")
        else:  # full
            # Write all columns in a readable way, then the code
            lines.append("metadata:\n")
            for col in best.columns:
                if col == "code":
                    continue
                lines.append(f"- {col}: {row[col]}\n")
            lines.append("\ncode:\n")
            lines.append(str(row["code"]) + "\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {len(best)} program(s) to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
