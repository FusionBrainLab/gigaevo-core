"""Run a frozen-graph × evaluator matrix over several estimator seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

from problems.tabular_dag_baselines.compare import EVALUATOR_MODULES


def _graph_argument(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("graph must be NAME=PATH") from exc
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("graph must be NAME=PATH")
    return name, Path(raw_path).expanduser().resolve()


def _summarize(rows: list[dict], field: str) -> dict[str, dict[str, float]]:
    payloads = [row.get(field, {}) for row in rows]
    keys = sorted(set.intersection(*(set(payload) for payload in payloads)))
    summary = {}
    for key in keys:
        values = [payload[key] for payload in payloads]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            continue
        numeric = [float(value) for value in values]
        summary[key] = {
            "mean": statistics.fmean(numeric),
            "sample_std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            "n_seeds": len(numeric),
        }
    return summary


def _run_cell(
    *,
    evaluator: str,
    graph_name: str,
    graph_path: Path,
    seed: int,
    phase: str,
    root: Path,
) -> dict:
    output = root / f"{evaluator}__{graph_name}__seed{seed}.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "problems.tabular_dag_baselines.compare",
            "--evaluator",
            evaluator,
            "--graph",
            str(graph_path),
            "--phase",
            phase,
            "--seed",
            str(seed),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{evaluator}/{graph_name}/seed={seed} failed with "
            f"rc={completed.returncode}:\n{completed.stderr[-4000:]}"
        )
    result = json.loads(output.read_text())
    result["graph"] = graph_name
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="append", type=_graph_argument, required=True)
    parser.add_argument(
        "--evaluator",
        action="append",
        choices=tuple(EVALUATOR_MODULES),
        required=True,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--phase", choices=("both", "cv", "test"), default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph_names = [name for name, _ in args.graph]
    if len(graph_names) != len(set(graph_names)):
        parser.error("graph names must be unique")
    if len(args.evaluator) != len(set(args.evaluator)):
        parser.error("evaluators must be unique")
    if any(seed < 0 for seed in args.seeds) or len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique non-negative integers")
    if args.workers < 1:
        parser.error("--workers must be positive")
    missing = [str(path) for _, path in args.graph if not path.is_file()]
    if missing:
        parser.error(f"graph files do not exist: {missing}")

    with tempfile.TemporaryDirectory(prefix="gigaevo-cross-eval-") as temp:
        temp_root = Path(temp)
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for evaluator in args.evaluator:
                for graph_name, graph_path in args.graph:
                    for seed in args.seeds:
                        future = executor.submit(
                            _run_cell,
                            evaluator=evaluator,
                            graph_name=graph_name,
                            graph_path=graph_path,
                            seed=seed,
                            phase=args.phase,
                            root=temp_root,
                        )
                        futures[future] = (evaluator, graph_name, seed)
            rows = [future.result() for future in as_completed(futures)]

    evaluator_order = {name: index for index, name in enumerate(args.evaluator)}
    graph_order = {name: index for index, name in enumerate(graph_names)}
    rows.sort(
        key=lambda row: (
            evaluator_order[row["evaluator"]],
            graph_order[row["graph"]],
            row["seed"],
        )
    )
    cells = {}
    for evaluator in args.evaluator:
        cells[evaluator] = {}
        for graph_name in graph_names:
            selected = [
                row
                for row in rows
                if row["evaluator"] == evaluator and row["graph"] == graph_name
            ]
            cells[evaluator][graph_name] = {
                "per_seed": selected,
                "cv_summary": _summarize(selected, "cv_metrics"),
                "test_summary": _summarize(selected, "test_metrics"),
            }

    result = {
        "protocol": {
            "phase": args.phase,
            "seeds": args.seeds,
            "seed_std": "sample standard deviation (ddof=1)",
            "feature_graphs_fixed": True,
            "only_estimator_seed_varied": True,
            "test_split_reused": args.phase in {"both", "test"},
            "workers": args.workers,
        },
        "graphs": {name: str(path) for name, path in args.graph},
        "evaluators": args.evaluator,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
