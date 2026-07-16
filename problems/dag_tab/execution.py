"""Validation and execution of pandas feature-node bodies."""

from __future__ import annotations

import ast
import math
from textwrap import indent

import numpy as np
import pandas as pd

from .graph import FeatureGraph, FeatureNode


class FeatureExecutionError(RuntimeError):
    pass


_BLOCKED_CALLS = {
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_SPLIT_DEPENDENT_CALLS = {
    "agg",
    "aggregate",
    "corr",
    "count",
    "cov",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "expanding",
    "groupby",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "pct_change",
    "prod",
    "qcut",
    "quantile",
    "rank",
    "rolling",
    "shift",
    "std",
    "sum",
    "transform",
    "value_counts",
    "var",
}
_BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _df_column(target: ast.expr) -> str | None:
    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "df":
        return None
    key = target.slice
    return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None


def _df_accessed_column(node: ast.Subscript) -> str | None:
    return _df_column(node)


def normalize_node_code(code: str) -> str:
    stripped = code.strip()
    tree = ast.parse(stripped)
    final = tree.body[-1] if tree.body else None
    ends_with_return_df = (
        isinstance(final, ast.Return)
        and isinstance(final.value, ast.Name)
        and final.value.id == "df"
    )
    return stripped if ends_with_return_df else f"{stripped}\nreturn df"


def validate_node_code(node: FeatureNode) -> None:
    try:
        tree = ast.parse(node.code)
    except SyntaxError as exc:
        raise ValueError(f"node {node.id}: invalid Python: {exc}") from exc

    assigned: set[str] = set()
    accessed: set[str] = set()
    returns_df = False
    for item in ast.walk(tree):
        if isinstance(item, _BLOCKED_NODES):
            raise ValueError(f"node {node.id}: forbidden syntax {type(item).__name__}")
        if isinstance(item, ast.Attribute) and item.attr.startswith("_"):
            raise ValueError(f"node {node.id}: private attribute access is forbidden")
        if isinstance(item, ast.Name) and item.id.startswith("__"):
            raise ValueError(f"node {node.id}: dunder names are forbidden")
        if isinstance(item, ast.Call):
            if isinstance(item.func, ast.Name) and item.func.id in _BLOCKED_CALLS:
                raise ValueError(f"node {node.id}: call to {item.func.id} is forbidden")
            if isinstance(item.func, ast.Name) and item.func.id in {"max", "min", "sum"}:
                raise ValueError(
                    f"node {node.id}: split-dependent operation {item.func.id!r} "
                    "is forbidden; use row-wise numpy operations instead"
                )
            if (
                isinstance(item.func, ast.Attribute)
                and item.func.attr in _SPLIT_DEPENDENT_CALLS
            ):
                raise ValueError(
                    f"node {node.id}: split-dependent operation {item.func.attr!r} "
                    "is forbidden; node code must be row-wise"
                )
            if (
                isinstance(item.func, ast.Attribute)
                and isinstance(item.func.value, ast.Name)
                and item.func.value.id == "pd"
                and item.func.attr == "qcut"
            ):
                raise ValueError(
                    f"node {node.id}: split-dependent operation 'qcut' is forbidden; "
                    "node code must be row-wise"
                )
        if isinstance(item, ast.Subscript):
            column = _df_accessed_column(item)
            if column is not None and isinstance(item.ctx, ast.Load):
                accessed.add(column)
        if isinstance(item, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            assigned.update(filter(None, (_df_column(target) for target in targets)))
        if isinstance(item, ast.Return) and isinstance(item.value, ast.Name):
            returns_df = returns_df or item.value.id == "df"

    extra_assignments = assigned - set(node.output_cols)
    if extra_assignments:
        raise ValueError(
            f"node {node.id}: code assigns undeclared columns {sorted(extra_assignments)}"
        )
    undeclared_reads = accessed - set(node.input_cols)
    if undeclared_reads:
        raise ValueError(
            f"node {node.id}: code reads undeclared input columns {sorted(undeclared_reads)}"
        )
    missing = set(node.output_cols) - assigned
    if missing:
        raise ValueError(
            f"node {node.id}: code must explicitly assign df[column] for {sorted(missing)}"
        )
    if not returns_df:
        raise ValueError(f"node {node.id}: code must contain return df")


def _compile_transform(node: FeatureNode):
    validate_node_code(node)
    source = "def transform(df):\n" + indent(node.code, "    ")
    namespace = {
        "__builtins__": _ALLOWED_BUILTINS,
        "math": math,
        "np": np,
        "pd": pd,
    }
    exec(compile(source, f"<feature-node:{node.id}>", "exec"), namespace)
    return namespace["transform"]


def execute_graph(graph: FeatureGraph, frame: pd.DataFrame) -> pd.DataFrame:
    missing_raw = set(graph.raw_columns) - set(frame.columns)
    if missing_raw:
        raise FeatureExecutionError(f"missing raw columns: {sorted(missing_raw)}")

    result = frame.loc[:, graph.raw_columns].copy()
    original_index = result.index.copy()
    for node in graph.nodes:
        missing = set(node.input_cols) - set(result.columns)
        if missing:
            raise FeatureExecutionError(
                f"node {node.id}: missing inputs {sorted(missing)}"
            )
        before_columns = set(result.columns)
        before_frame = result.copy(deep=True)
        try:
            transformed = _compile_transform(node)(result.copy(deep=True))
        except Exception as exc:
            raise FeatureExecutionError(f"node {node.id}: {exc}") from exc
        if not isinstance(transformed, pd.DataFrame):
            raise FeatureExecutionError(f"node {node.id}: transform must return DataFrame")
        if len(transformed) != len(result) or not transformed.index.equals(original_index):
            raise FeatureExecutionError(f"node {node.id}: row count/index changed")
        missing_outputs = set(node.output_cols) - set(transformed.columns)
        if missing_outputs:
            raise FeatureExecutionError(
                f"node {node.id}: missing declared outputs {sorted(missing_outputs)}"
            )
        undeclared = set(transformed.columns) - before_columns - set(node.output_cols)
        if undeclared:
            raise FeatureExecutionError(
                f"node {node.id}: created undeclared columns {sorted(undeclared)}"
            )
        missing_existing = before_columns - set(transformed.columns)
        if missing_existing:
            raise FeatureExecutionError(
                f"node {node.id}: removed pre-existing columns {sorted(missing_existing)}"
            )
        changed_existing = [
            col
            for col in before_frame.columns
            if not transformed[col].equals(before_frame[col])
        ]
        if changed_existing:
            raise FeatureExecutionError(
                f"node {node.id}: modified pre-existing columns {changed_existing}"
            )
        for col in node.output_cols:
            values = transformed[col]
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                if not np.all(np.isfinite(numeric)):
                    raise FeatureExecutionError(
                        f"node {node.id}: output {col!r} contains NaN or inf"
                    )
        result = transformed

    output_cols = [*graph.raw_columns, *graph.output_columns]
    return result.loc[:, list(dict.fromkeys(output_cols))]
