"""Execution contracts for row-wise and fitted tabular feature nodes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from textwrap import indent

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .graph import FeatureGraph, FeatureNode, FeatureValueKind, TargetTransform


class FeatureExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphTriplet:
    """Aligned graph outputs produced under one fitted preprocessing contract."""

    fit: pd.DataFrame
    validation: pd.DataFrame
    query: pd.DataFrame


def _extract_target_function_body(
    code: str, allowed_names: set[str]
) -> tuple[str, ast.Module]:
    stripped = code.strip()
    tree = ast.parse(stripped)
    function_defs = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    if not function_defs:
        return stripped, tree
    non_imports = [
        item for item in tree.body if not isinstance(item, (ast.Import, ast.ImportFrom))
    ]
    if (
        len(function_defs) != 1
        or len(non_imports) != 1
        or function_defs[0].name not in allowed_names
    ):
        expected = "/".join(sorted(allowed_names))
        raise ValueError(
            f"code containing a function must contain only imports and def {expected}(...); "
            "move module-level setup inside transform"
        )
    if function_defs[0].decorator_list:
        raise ValueError("transform function cannot use decorators")
    body = [
        *[item for item in tree.body if isinstance(item, (ast.Import, ast.ImportFrom))],
        *function_defs[0].body,
    ]
    extracted = ast.Module(body=body, type_ignores=[])
    return ast.unparse(extracted).strip(), extracted


def normalize_node_code(code: str) -> str:
    stripped = code.strip()
    tree = ast.parse(stripped)
    transforms = _module_transforms(tree)
    _reject_transform_decorators(transforms)
    if not transforms:
        final = tree.body[-1] if tree.body else None
        return stripped if isinstance(final, ast.Return) else f"{stripped}\nreturn df"

    transform = transforms[-1]
    final = transform.body[-1] if transform.body else None
    if isinstance(final, ast.Return):
        return stripped
    transform.body.append(ast.Return(value=ast.Name(id="df", ctx=ast.Load())))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip()


def normalize_target_code(code: str, *, inverse: bool = False) -> str:
    names = {"inverse", "inverse_transform"} if inverse else {"transform"}
    stripped, _ = _extract_target_function_body(code, names)
    return stripped


def _module_transforms(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "transform"
    ]


def _reject_transform_decorators(transforms: list[ast.FunctionDef]) -> None:
    if any(transform.decorator_list for transform in transforms):
        raise ValueError("transform function cannot use decorators")


_ROUNDING_RTOL = 1e-9
_ABI_FRAME_NAMES = frozenset({"df", "df_fit"})
_AGGREGATE_ABI_NAMES = frozenset({"df_fit", "y_fit"})
_PANDAS_ONLY_TARGET_ATTRIBUTES = {
    "groupby",
    "iloc",
    "loc",
    "name",
    "to_frame",
    "values",
}


def _literal_selector_columns(selection: ast.expr) -> set[str] | None:
    if isinstance(selection, ast.Constant) and isinstance(selection.value, str):
        return {selection.value}
    if isinstance(selection, (ast.List, ast.Tuple)) and all(
        isinstance(item, ast.Constant) and isinstance(item.value, str)
        for item in selection.elts
    ):
        return {item.value for item in selection.elts}
    return None


def _frame_expression_source(value: ast.expr) -> ast.expr:
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "copy"
    ):
        return value.func.value
    return value


def _is_frame_expression(value: ast.expr) -> bool:
    source = _frame_expression_source(value)
    return isinstance(source, ast.Name) and source.id in _ABI_FRAME_NAMES


@dataclass(frozen=True)
class _ScopeBindings:
    bound_names: frozenset[str]
    literal_columns: dict[str, frozenset[str]]
    frame_aliases: frozenset[str]


_SCOPE_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _scope_body(scope: ast.AST) -> list[ast.AST]:
    if isinstance(scope, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
        return list(scope.body)
    if isinstance(scope, ast.Lambda):
        return [scope.body]
    raise TypeError(f"unsupported scope {type(scope).__name__}")


def _scope_bindings(scope: ast.AST, *, abi_parameters: bool = False) -> _ScopeBindings:
    """Collect exact single bindings without crossing nested function scopes.

    Only the entry-point ``transform`` receives the ABI frames as parameters, so
    a helper parameter spelled ``df`` shadows the ABI name instead of aliasing it.
    """

    nodes: list[ast.AST] = []
    stack = list(reversed(_scope_body(scope)))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, _SCOPE_NODES[1:]):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))

    binding_counts: dict[str, int] = {}
    assigned_values: dict[str, ast.expr] = {}
    value_binding_counts: dict[str, int] = {}
    rebound_away: set[str] = set()
    parameter_names: set[str] = set()

    def record(name: str, value: ast.expr | None = None) -> None:
        binding_counts[name] = binding_counts.get(name, 0) + 1
        if value is not None:
            assigned_values[name] = value

    def bind_value(name: str, value: ast.expr) -> None:
        assigned_values[name] = value
        value_binding_counts[name] = value_binding_counts.get(name, 0) + 1
        if name in _ABI_FRAME_NAMES and not _is_frame_expression(value):
            rebound_away.add(name)

    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        arguments = [
            *scope.args.posonlyargs,
            *scope.args.args,
            *scope.args.kwonlyargs,
        ]
        for argument in arguments:
            record(argument.arg)
            parameter_names.add(argument.arg)
        if scope.args.vararg is not None:
            record(scope.args.vararg.arg)
        if scope.args.kwarg is not None:
            record(scope.args.kwarg.arg)

    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            record(node.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bind_value(target.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                bind_value(node.target.id, node.value)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            bind_value(node.target.id, node.value)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                record(alias.asname or alias.name.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            record(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            record(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            for name in node.names:
                record(name)

    literal_columns: dict[str, frozenset[str]] = {}
    for name, value in assigned_values.items():
        if binding_counts.get(name) != 1:
            continue
        columns = _literal_selector_columns(value)
        if columns is not None:
            literal_columns[name] = frozenset(columns)

    # An ABI name bound to anything but another frame stops being the frame here.
    # Whether it still holds it depends on control flow, and crediting reads through
    # it invents columns that transcription would then add to input_cols, changing
    # the contract of a node that never mentioned them. Bindings whose value this
    # cannot see at all — loop targets, `with ... as`, `except ... as`, imports —
    # count the same way; only the ABI parameter itself is a declaration.
    for name in _ABI_FRAME_NAMES:
        declarations = 1 if name in parameter_names else 0
        opaque = (
            binding_counts.get(name, 0)
            - value_binding_counts.get(name, 0)
            - declarations
        )
        if opaque > 0:
            rebound_away.add(name)
    declared_aliases = parameter_names & _ABI_FRAME_NAMES if abi_parameters else set()
    frame_aliases: set[str] = declared_aliases - rebound_away
    reachable_frames = (
        set(_ABI_FRAME_NAMES - parameter_names) | frame_aliases
    ) - rebound_away
    changed = True
    while changed:
        changed = False
        for name, value in assigned_values.items():
            if binding_counts.get(name) != 1 or name in frame_aliases:
                continue
            source: ast.expr = value
            if (
                isinstance(source, ast.Call)
                and isinstance(source.func, ast.Attribute)
                and source.func.attr == "copy"
            ):
                source = source.func.value
            if isinstance(source, ast.Name) and source.id in reachable_frames:
                frame_aliases.add(name)
                reachable_frames.add(name)
                changed = True

    return _ScopeBindings(
        bound_names=frozenset(binding_counts),
        literal_columns=literal_columns,
        frame_aliases=frozenset(frame_aliases),
    )


class _LiteralFrameReadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.scopes: list[_ScopeBindings] = []

    def _visit_scope(self, node: ast.AST) -> None:
        entry_point = (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "transform"
            and len(self.scopes) == 1
        )
        self.scopes.append(_scope_bindings(node, abi_parameters=entry_point))
        for statement in _scope_body(node):
            self.visit(statement)
        self.scopes.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_scope(node)

    def _is_frame(self, name: str) -> bool:
        for scope in reversed(self.scopes):
            if name in scope.bound_names:
                return name in scope.frame_aliases
        return name in _ABI_FRAME_NAMES

    def _selector_columns(self, selection: ast.expr) -> set[str] | None:
        columns = _literal_selector_columns(selection)
        if columns is not None:
            return columns
        if not isinstance(selection, ast.Name):
            return None
        for scope in reversed(self.scopes):
            if selection.id in scope.bound_names:
                resolved = scope.literal_columns.get(selection.id)
                return None if resolved is None else set(resolved)
        return None

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load):
            value = node.value
            if isinstance(value, ast.Name) and self._is_frame(value.id):
                referenced = self._selector_columns(node.slice)
                if referenced is not None:
                    self.reads.update(referenced)
            elif (
                isinstance(value, ast.Attribute)
                and value.attr == "loc"
                and isinstance(value.value, ast.Name)
                and self._is_frame(value.value.id)
                and isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) == 2
            ):
                referenced = self._selector_columns(node.slice.elts[1])
                if referenced is not None:
                    self.reads.update(referenced)
        self.generic_visit(node)


def literal_frame_reads(code: str) -> set[str]:
    """Return the frame reads we can prove; unprovable forms yield no reads."""

    tree = ast.parse(code.strip())
    visitor = _LiteralFrameReadVisitor()
    visitor.visit(tree)
    return visitor.reads


def _exception_metadata(exc: BaseException, attribute: str) -> object | None:
    """Read exception metadata without letting the read replace the node's failure.

    Node code is ordinary Python and may raise an exception class whose attributes
    are properties that themselves raise; the hint is advisory, so an unreadable
    attribute means no hint, never a different error than the one the node hit.
    """

    try:
        return getattr(exc, attribute, None)
    except Exception:
        return None


def _abi_error_hint(node: FeatureNode, exc: BaseException) -> str:
    """Guidance for the two commonest ABI mistakes, read off the raised exception.

    Both were once proven from the syntax tree. Whether a name holds the ABI
    object depends on scope and control flow, so a source-level rule rejects
    valid shadowing and rebinding while still missing the invalid reads; the
    interpreter has already resolved the name by the time these are raised.
    """

    name = _exception_metadata(exc, "name")
    if not isinstance(name, str):
        return ""
    if (
        isinstance(exc, AttributeError)
        and isinstance(_exception_metadata(exc, "obj"), np.ndarray)
        and name in _PANDAS_ONLY_TARGET_ATTRIBUTES
    ):
        return (
            "; that object is a numpy.ndarray, which has no pandas attributes; wrap "
            "the array explicitly, as in pd.Series(np.asarray(y_fit), "
            "index=df_fit.index), when pandas alignment is required"
        )
    if (
        isinstance(exc, NameError)
        and node.kind == "rowwise"
        and name in _AGGREGATE_ABI_NAMES
    ):
        return (
            f"; rowwise code cannot reference {name}, which is only passed to "
            "aggregate nodes; use kind='aggregate'"
        )
    return ""


def validate_node_code(node: FeatureNode) -> None:
    """Reject obvious ABI mistakes without treating trusted Python as a sandbox."""

    try:
        source = _node_source(node)
        tree = ast.parse(source)
        compile(tree, f"<feature-node:{node.id}>", "exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"node {node.id}: invalid Python: {exc}") from exc


def _node_source(node: FeatureNode) -> str:
    stripped = node.code.strip()
    tree = ast.parse(stripped)
    transforms = _module_transforms(tree)
    _reject_transform_decorators(transforms)
    if transforms:
        return stripped
    arguments = "df" if node.kind == "rowwise" else "df_fit, y_fit, df"
    return f"def transform({arguments}):\n" + indent(stripped, "    ")


def _compile_transform(node: FeatureNode):
    validate_node_code(node)
    namespace = {"math": math, "np": np, "pd": pd}
    source = _node_source(node)
    exec(compile(source, f"<feature-node:{node.id}>", "exec"), namespace)
    return namespace["transform"]


def _compile_target_body(name: str, arguments: str, code: str):
    source = f"def {name}({arguments}):\n" + indent(code.strip(), "    ")
    namespace = {"math": math, "np": np, "pd": pd}
    try:
        exec(compile(source, f"<target-{name}>", "exec"), namespace)
    except (SyntaxError, ValueError) as exc:
        raise FeatureExecutionError(f"target {name}: invalid Python: {exc}") from exc
    return namespace[name]


def _validate_target_values(label: str, values, expected_length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != expected_length:
        raise FeatureExecutionError(
            f"target {label}: expected one-dimensional length {expected_length}, "
            f"got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise FeatureExecutionError(f"target {label}: output must be numeric")
    numeric = array.astype(float)
    if not np.isfinite(numeric).all():
        raise FeatureExecutionError(f"target {label}: output must be finite")
    return numeric


def transform_target(
    target: TargetTransform | None,
    y_fit: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    fit = np.asarray(y_fit)
    current = np.asarray(values)
    if target is None:
        return current.copy()
    transform = _compile_target_body("transform", "y_fit, y", target.code)
    try:
        result = transform(fit.copy(), current.copy())
    except Exception as exc:
        raise FeatureExecutionError(f"target transform: {exc}") from exc
    return _validate_target_values("transform", result, len(current))


def inverse_target(
    target: TargetTransform | None,
    y_fit: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    fit = np.asarray(y_fit)
    current = np.asarray(predictions)
    if target is None:
        return current.copy()
    inverse = _compile_target_body("inverse", "y_fit, predictions", target.inverse_code)
    try:
        result = inverse(fit.copy(), current.copy())
    except Exception as exc:
        raise FeatureExecutionError(f"target inverse: {exc}") from exc
    return _validate_target_values("inverse", result, len(current))


def assert_target_round_trip(target: TargetTransform, y_fit: np.ndarray) -> None:
    fit = np.asarray(y_fit, dtype=float)
    if fit.ndim != 1 or len(fit) == 0:
        raise FeatureExecutionError(
            "target round-trip probe requires non-empty 1D y_fit"
        )
    probe = fit[: min(len(fit), 64)].copy()
    transformed = transform_target(target, fit, probe)
    repeated = transform_target(target, fit, probe)
    if not np.array_equal(transformed, repeated, equal_nan=True):
        raise FeatureExecutionError("target transform is non-deterministic")
    restored = inverse_target(target, fit, transformed)
    if not np.allclose(restored, probe, rtol=1e-6, atol=1e-8):
        raise FeatureExecutionError(
            "target transform and inverse do not round-trip to the original target"
        )


def _validate_output_type(
    node: FeatureNode, column: str, values: pd.Series, kind: FeatureValueKind
) -> None:
    if kind in {"numerical", "ordinal"}:
        if not pd.api.types.is_numeric_dtype(values):
            raise FeatureExecutionError(
                f"node {node.id}: output {column!r} declares {kind} but has "
                f"non-numeric dtype {values.dtype}"
            )
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise FeatureExecutionError(
                f"node {node.id}: output {column!r} contains inf"
            )
        return

    if kind == "binary":
        non_missing = values.dropna()
        if not non_missing.isin([0, 1, False, True]).all():
            raise FeatureExecutionError(
                f"node {node.id}: binary output {column!r} contains values other than 0/1"
            )
        return

    if kind == "categorical":
        for value in values.dropna().tolist():
            if not isinstance(value, (str, int, np.integer, bool, np.bool_)):
                raise FeatureExecutionError(
                    f"node {node.id}: categorical output {column!r} contains "
                    f"unsupported value {value!r}"
                )
        return

    raise FeatureExecutionError(f"node {node.id}: unsupported output type {kind!r}")


def _execute_on_combined(
    graph: FeatureGraph,
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    y_fit: np.ndarray | None,
) -> GraphTriplet:
    lengths = (len(fit_frame), len(validation_frame), len(query_frame))
    frames = (fit_frame, validation_frame, query_frame)
    for label, frame in zip(("fit", "validation", "query"), frames):
        missing_raw = set(graph.raw_columns) - set(frame.columns)
        if missing_raw:
            raise FeatureExecutionError(
                f"{label} frame missing raw columns: {sorted(missing_raw)}"
            )

    if y_fit is not None and len(y_fit) != lengths[0]:
        raise FeatureExecutionError(
            f"y_fit length {len(y_fit)} does not match fit rows {lengths[0]}"
        )

    result = pd.concat(
        [frame.loc[:, graph.raw_columns] for frame in frames],
        ignore_index=True,
        copy=True,
    )
    fit_stop = lengths[0]
    original_index = result.index.copy()

    def frozen_target() -> np.ndarray | None:
        if y_fit is None:
            return None
        target = np.array(y_fit, copy=True)
        target.flags.writeable = False
        return target

    for node in graph.nodes:
        missing = set(node.input_cols) - set(result.columns)
        if missing:
            raise FeatureExecutionError(
                f"node {node.id}: missing inputs {sorted(missing)}"
            )
        node_input = result.loc[:, node.input_cols].copy()
        node_input_before = node_input.copy(deep=True)
        fit_input = result.iloc[:fit_stop].loc[:, node.input_cols].copy()
        try:
            if node.kind == "aggregate":
                transformed = _compile_transform(node)(
                    fit_input, frozen_target(), node_input
                )
            else:
                boundaries = (0, lengths[0], lengths[0] + lengths[1], sum(lengths))
                transformed_parts: list[pd.DataFrame] = []
                for start, stop in zip(boundaries, boundaries[1:]):
                    if start == stop:
                        continue
                    transformed_parts.append(
                        _compile_transform(node)(node_input.iloc[start:stop].copy())
                    )
                transformed = pd.concat(transformed_parts, axis=0)
        except KeyError as exc:
            raise FeatureExecutionError(
                f"node {node.id}: read unavailable input column {exc}"
            ) from exc
        except Exception as exc:
            raise FeatureExecutionError(
                f"node {node.id}: {exc}{_abi_error_hint(node, exc)}"
            ) from exc

        if not isinstance(transformed, pd.DataFrame):
            raise FeatureExecutionError(
                f"node {node.id}: transform must return DataFrame"
            )
        if len(transformed) != len(result) or not transformed.index.equals(
            original_index
        ):
            raise FeatureExecutionError(f"node {node.id}: row count/index changed")

        transformed_columns = set(transformed.columns)
        missing_outputs = set(node.output_cols) - transformed_columns
        if missing_outputs:
            raise FeatureExecutionError(
                f"node {node.id}: missing declared outputs {sorted(missing_outputs)}"
            )
        undeclared = transformed_columns - set(node.input_cols) - set(node.output_cols)
        if undeclared:
            raise FeatureExecutionError(
                f"node {node.id}: created undeclared columns {sorted(undeclared)}"
            )
        missing_existing = set(node.input_cols) - transformed_columns
        if missing_existing:
            raise FeatureExecutionError(
                f"node {node.id}: removed pre-existing columns {sorted(missing_existing)}"
            )
        changed_existing = [
            col
            for col in node.input_cols
            if not transformed[col].equals(node_input_before[col])
        ]
        if changed_existing:
            raise FeatureExecutionError(
                f"node {node.id}: modified pre-existing columns {changed_existing}"
            )

        for column in node.output_cols:
            _validate_output_type(
                node, column, transformed[column], node.output_type(column)
            )
        result[node.output_cols] = transformed[node.output_cols]

    output_cols = graph.estimator_columns
    fit_end = lengths[0]
    validation_end = fit_end + lengths[1]
    return GraphTriplet(
        fit=result.iloc[:fit_end].loc[:, output_cols].reset_index(drop=True),
        validation=result.iloc[fit_end:validation_end]
        .loc[:, output_cols]
        .reset_index(drop=True),
        query=result.iloc[validation_end:].loc[:, output_cols].reset_index(drop=True),
    )


def execute_graph_triplet(
    graph: FeatureGraph,
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    y_fit: np.ndarray | None = None,
) -> GraphTriplet:
    """Fit aggregate nodes on fit rows and jointly transform all three roles."""

    return _execute_on_combined(
        graph,
        fit_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
        query_frame.reset_index(drop=True),
        None if y_fit is None else np.asarray(y_fit),
    )


def execute_graph(graph: FeatureGraph, frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible single-frame execution for row-wise graphs and fixtures."""

    empty = frame.iloc[:0].copy()
    return execute_graph_triplet(graph, frame, empty, empty).fit


# Shares no positional landmark with the full frame: drops both endpoints so
# first/last-row broadcasts differ, strides so neighbour ops (shift/diff) differ.
_PROBE_SUBSET = slice(1, -1, 2)


def _perturbed_target(target: np.ndarray, index: int, scale: float) -> np.ndarray:
    """``target`` with row ``index`` moved to a value it cannot compare equal to.

    A perturbation the node cannot see is a probe that always passes: adding to a
    huge float rounds back to itself, and adding to NaN stays NaN.
    """

    if np.issubdtype(target.dtype, np.integer):
        perturbed = target.copy()
        classes = np.unique(perturbed)
        alternatives = classes[classes != perturbed[index]]
        perturbed[index] = (
            alternatives[0] if len(alternatives) else perturbed[index] + 1
        )
        return perturbed

    perturbed = target.astype(float)
    original = perturbed[index]
    for candidate in (original + scale, original * 2.0 + scale, 0.0, 1.0, -1.0):
        if np.isfinite(candidate) and candidate != original:
            perturbed[index] = candidate
            return perturbed
    raise FeatureExecutionError(
        f"cannot probe own-target invariance at fit row {index}: no representable "
        f"target value differs from {original}"
    )


def _value_scale(values: pd.Series) -> float:
    if not is_numeric_dtype(values):
        return 1.0
    magnitudes = np.abs(values.to_numpy(dtype=float))
    finite = magnitudes[np.isfinite(magnitudes)]
    return max(1.0, float(finite.max())) if finite.size else 1.0


def _values_differ(
    left: pd.Series, right: pd.Series, *, scale: float | None = None
) -> bool:
    """Compare numerically, allowing a step at the scale of floating-point rounding.

    A batch of a different size can take a different kernel, and the algebraically
    exact ``(group_sum - own) / (count - 1)`` leave-one-out form moves by the same
    one step. Both are honest features. A dependence strong enough to teach the
    model moves the value by a fraction of the target's own scale, orders above
    this bound, so nothing that matters hides under it.
    """

    if left.dtype != right.dtype:
        return True
    if not is_numeric_dtype(left):
        return not left.equals(right)
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    tolerance = _ROUNDING_RTOL * (_value_scale(left) if scale is None else scale)
    return not np.allclose(
        left_values, right_values, rtol=0.0, atol=tolerance, equal_nan=True
    )


def _different_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return [
        column for column in left.columns if _values_differ(left[column], right[column])
    ]


def assert_split_invariant(
    graph: FeatureGraph,
    frame: pd.DataFrame,
    y_fit: np.ndarray | None = None,
) -> None:
    """Enforce frozen-fit batch purity, determinism, and own-target invariance."""

    if len(frame) < 4:
        raise FeatureExecutionError(
            f"split-invariance probe needs at least 4 rows; got {len(frame)}"
        )
    fit = frame.reset_index(drop=True)
    full_query = frame.reset_index(drop=True)
    subset_query = frame.iloc[_PROBE_SUBSET].reset_index(drop=True)
    empty = frame.iloc[:0].copy()

    target = None if y_fit is None else np.asarray(y_fit)
    if target is not None and len(target) != len(fit):
        raise FeatureExecutionError(
            f"y_fit length {len(target)} does not match probe rows {len(fit)}"
        )

    full = execute_graph_triplet(graph, fit, empty, full_query, y_fit=target).query
    repeated = execute_graph_triplet(graph, fit, empty, full_query, y_fit=target).query
    nondeterministic = _different_columns(full, repeated)
    if nondeterministic:
        raise FeatureExecutionError(
            f"non-deterministic behavior: columns {sorted(nondeterministic)} change "
            "between identical executions; seed every stochastic operation"
        )

    subset = execute_graph_triplet(graph, fit, empty, subset_query, y_fit=target).query
    full_subset = full.iloc[_PROBE_SUBSET].reset_index(drop=True)
    diff_cols = _different_columns(full_subset, subset)
    if diff_cols:
        raise FeatureExecutionError(
            f"split-dependent behavior: columns {sorted(diff_cols)} change with batch "
            "composition; if this computation needs whole-column statistics, use an "
            "aggregate node and fit them on df_fit"
        )

    if target is None or not any(node.kind == "aggregate" for node in graph.nodes):
        return
    own_target_graph = graph.model_copy(
        update={
            "nodes": [
                node.model_copy(update={"is_output": True}) for node in graph.nodes
            ]
        }
    )
    generated_columns = [column for node in graph.nodes for column in node.output_cols]
    # Own-target probing happens at the fit-only shape because that is the shape the
    # scored path uses for the rows the model learns from; no appended batch exists
    # there to condition on. Sampling appended shapes here cannot certify the ones it
    # does not sample, so it is not what makes the probe sound — it reports a node
    # whose fit rows move with the batch, which is a confused node either way.
    appended_shapes = {
        f"{len(subset_query)} appended rows": subset_query,
        f"{len(full_query)} appended rows": full_query,
    }
    baseline_fit = execute_graph_triplet(
        own_target_graph, fit, empty, empty, y_fit=target
    ).fit
    for label, appended in appended_shapes.items():
        candidate = execute_graph_triplet(
            own_target_graph, fit, empty, appended, y_fit=target
        ).fit
        batch_dependent = _different_columns(
            baseline_fit.loc[:, generated_columns], candidate.loc[:, generated_columns]
        )
        if batch_dependent:
            raise FeatureExecutionError(
                f"batch-dependent fit rows: columns {sorted(batch_dependent)} differ "
                f"between no appended rows and {label}; fit rows must depend on df_fit "
                "and y_fit alone"
            )
    scale = max(1.0, float(np.nanstd(target)))
    column_scales = {
        column: _value_scale(baseline_fit[column]) for column in generated_columns
    }
    for index in range(len(fit)):
        perturbed = _perturbed_target(target, index, scale)
        candidate_fit = execute_graph_triplet(
            own_target_graph, fit, empty, empty, y_fit=perturbed
        ).fit
        changed = [
            column
            for column in generated_columns
            if _values_differ(
                pd.Series([baseline_fit.at[index, column]]),
                pd.Series([candidate_fit.at[index, column]]),
                scale=column_scales[column],
            )
        ]
        if changed:
            raise FeatureExecutionError(
                "own-target leakage: fit row "
                f"{index} columns {sorted(changed)} change when only its own y_fit changes; "
                "use leave-one-out or out-of-fold supervised features"
            )
