"""Evaluate one frozen FeatureGraph with one estimator and one model seed."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EVALUATOR_MODULES = {
    "catboost": "problems.dag_tab.validate",
    "tabm": "problems.tabular_dag_baselines.tabm.validate",
    "realmlp": "problems.tabular_dag_baselines.realmlp.validate",
    "tabicl": "problems.tabular_dag_baselines.tabicl.validate",
    "tabpfn": "problems.tabular_dag_baselines.tabpfn.validate",
    "lightgbm": "problems.tabular_dag_baselines.lightgbm.validate",
    "xgboost": "problems.tabular_dag_baselines.xgboost.validate",
}
SEED_ENV = {
    name: f"GIGAEVO_{name.upper()}_SEED"
    for name in EVALUATOR_MODULES
    if name != "catboost"
}


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _configure_seed(evaluator, evaluator_name: str, seed: int) -> str:
    if evaluator_name != "catboost":
        setting = SEED_ENV[evaluator_name]
        os.environ[setting] = str(seed)
        return setting

    # CatBoost's production recipe pins random_seed=0 directly. Keep that
    # recipe untouched and override only constructors in this comparison
    # process, matching the established CatBoost/TabM transfer audit.
    for name in ("CatBoostRegressor", "CatBoostClassifier"):
        constructor = getattr(evaluator, name)

        def seeded_constructor(*args, _constructor=constructor, **kwargs):
            kwargs["random_seed"] = seed
            return _constructor(*args, **kwargs)

        setattr(evaluator, name, seeded_constructor)
    return "experiment-local CatBoost constructor override"


def evaluate_cell(
    *, evaluator_name: str, graph_path: Path, phase: str, seed: int
) -> dict:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    graph_path = graph_path.expanduser().resolve()
    graph_bytes = graph_path.read_bytes()
    graph = json.loads(graph_bytes)

    # Environment-backed model configs read their seed inside validate/test,
    # so set it before importing the adapter.
    if evaluator_name != "catboost":
        os.environ[SEED_ENV[evaluator_name]] = str(seed)
    evaluator = importlib.import_module(EVALUATOR_MODULES[evaluator_name])
    seed_method = _configure_seed(evaluator, evaluator_name, seed)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"

    result = {
        "evaluator": evaluator_name,
        "seed": seed,
        "seed_method": seed_method,
        "graph_path": str(graph_path),
        "graph_file_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "revision": revision,
        "evaluator_module": str(Path(evaluator.__file__).resolve()),
    }

    # Some third-party estimators print progress to stdout. Keep stdout as a
    # clean machine-readable result stream and route their messages to stderr.
    with redirect_stdout(sys.stderr):
        if phase in {"both", "cv"}:
            started = time.perf_counter()
            metrics, artifact = evaluator.validate(graph)
            result["cv_seconds"] = time.perf_counter() - started
            result["cv_metrics"] = metrics
            result["cv_artifact"] = artifact
            if metrics.get("is_valid") != 1.0:
                return result

        if phase in {"both", "test"}:
            started = time.perf_counter()
            result["test_metrics"] = evaluator.score_on_test(graph)
            result["test_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", choices=tuple(EVALUATOR_MODULES), required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--phase", choices=("both", "cv", "test"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    result = evaluate_cell(
        evaluator_name=args.evaluator,
        graph_path=args.graph,
        phase=args.phase,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, default=_json_default)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(args.output)
    if result.get("cv_metrics", {}).get("is_valid", 1.0) != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
