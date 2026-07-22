"""Schema-constrained mutations for FeatureGraph JSON genomes."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any, Literal

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)

from gigaevo.evolution.mutation.allowed_changes import (
    AllowedChanges,
    DiffSchema,
    DiffStructuredOutputBase,
)
from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import portable_json_schema
from problems.dag_tab.execution import (
    literal_frame_reads,
    normalize_node_code,
    normalize_target_code,
    validate_node_code,
)
from problems.dag_tab.graph import FeatureGraph, FeatureNode


class SetTargetTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["set"]
    code: str = Field(
        min_length=1,
        max_length=3000,
        description="Body of transform(y_fit, y); return a numeric 1D array.",
    )
    inverse_code: str = Field(
        min_length=1,
        max_length=3000,
        description=(
            "Body of inverse(y_fit, predictions); return original-scale predictions."
        ),
    )


class KeepTargetTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["keep"]


class DropTargetTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["drop"]


class DagTabDiffBase(DiffStructuredOutputBase):
    """Shared evidence fields plus graph-level edits and the child node array."""

    structural_intent: Literal[
        "local_edit", "extend_chain", "compose_chain", "simplify_graph"
    ] = Field(
        description=(
            "Topology intent for this mutation. local_edit may keep depth unchanged; "
            "extend_chain must build on a generated output and increase base-parent depth; "
            "compose_chain must produce at least one dependency path of depth >= 2; "
            "simplify_graph may deliberately remove nodes or dependencies."
        )
    )
    minimum_child_depth: int | None = Field(
        default=None,
        ge=2,
        le=16,
        description=(
            "Optional claimed minimum dependency depth for the complete child. Set this for "
            "a deliberate multi-stage composition; Python rejects a shallower transcription."
        ),
    )
    dropped_raw_columns: list[str] | None = Field(
        default=None,
        description=(
            "Complete child list of raw columns hidden from CatBoost but still available "
            "to feature nodes. Null preserves the base parent list."
        ),
    )
    target_change: (
        SetTargetTransform | KeepTargetTransform | DropTargetTransform | None
    ) = Field(
        default=None,
        description=(
            "Regression target transform edit. Null/keep preserves the base transform; "
            "drop removes it; set supplies transform and inverse bodies."
        ),
    )


_RATIONALE_MAX_LENGTH = 500
_RATIONALE_DESCRIPTION = (
    "Feature hypothesis: signal exposed and what is lost if omitted."
)

_INPUT_COLS_DESCRIPTION = (
    "Columns code reads. df and df_fit hold exactly these columns and nothing else, so an "
    "undeclared column is absent rather than ignored. Declarable: raw xN and output_cols of "
    "earlier entries. Literal df/df_fit reads synchronize input_cols and dependencies."
)
_DEPENDENCY_DESCRIPTION = (
    "Earlier entry ids whose output_cols code consumes. Order is topological; unresolved or "
    "forward ids are dropped. Only consumed generated outputs add depth; unused declared edges "
    "do not."
)
_EDIT_CODE_DESCRIPTION = (
    "Replacement body or module; the ABI follows the retained or edited node kind. Assign "
    "every output_cols, preserve rows, index, and inputs, create no other columns, return "
    "the frame, and seed randomness. Code, batch purity, determinism, and aggregate own-target "
    "invariance are validated."
)
_ROWWISE_CODE_DESCRIPTION = (
    "Body or module defining transform(df). Assign every output_cols, preserve rows, index, "
    "and inputs, create no other columns, return the frame, and seed randomness. Each row is "
    "independent; df_fit and y_fit are unavailable. Trusted Python and imports are allowed."
)
_AGGREGATE_CODE_DESCRIPTION = (
    "Body or module defining transform(df_fit, y_fit, df). Fit state only from frozen training "
    "df_fit/y_fit and query it for every row of df; exclude each fitting row's own y_fit "
    "contribution. Assign every output_cols, preserve rows, index, and inputs, create no other "
    "columns, return the frame, and seed randomness. y_fit is a 1D numpy.ndarray; trusted "
    "Python and "
    "imports are allowed."
)


class NodeEdits(BaseModel):
    """Fields that a retained parent node may change."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["rowwise", "aggregate"] | None = Field(
        default=None,
        description=(
            "Optional ABI override: rowwise uses transform(df) and computes each row "
            "independently; aggregate uses transform(df_fit, y_fit, df) and may fit state "
            "from the frozen training frame and target."
        ),
    )
    input_cols: list[str] | None = Field(
        default=None,
        min_length=1,
        description=_INPUT_COLS_DESCRIPTION,
    )
    output_cols: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Exact set of new feature columns the edited code assigns.",
    )
    output_types: (
        dict[str, Literal["numerical", "categorical", "binary", "ordinal"]] | None
    ) = Field(
        default=None,
        description="Semantic type for every output column; defaults to numerical.",
    )
    code: str | None = Field(
        default=None,
        min_length=1,
        max_length=6000,
        description=_EDIT_CODE_DESCRIPTION,
    )
    rationale: str | None = Field(
        default=None,
        min_length=1,
        max_length=_RATIONALE_MAX_LENGTH,
        description=_RATIONALE_DESCRIPTION,
    )
    is_output: bool | None = None


def _new_node_model(name: str, kind: str, kind_description: str, code_description: str):
    return create_model(
        name,
        __config__=ConfigDict(extra="forbid"),
        kind=(Literal[kind], Field(..., description=kind_description)),  # type: ignore[valid-type]
        id=(
            str,
            Field(
                ...,
                min_length=1,
                pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
                description="Dependency label; the child id is deterministically uniquified.",
            ),
        ),
        input_cols=(
            list[str],
            Field(..., min_length=1, description=_INPUT_COLS_DESCRIPTION),
        ),
        output_cols=(
            list[str],
            Field(..., min_length=1, description="Exact new columns code assigns."),
        ),
        output_types=(
            dict[str, Literal["numerical", "categorical", "binary", "ordinal"]],
            Field(
                default_factory=dict,
                description="Semantic type of each output; omitted outputs are numerical.",
            ),
        ),
        code=(
            str,
            Field(..., min_length=1, max_length=6000, description=code_description),
        ),
        rationale=(
            str,
            Field(
                ...,
                min_length=1,
                max_length=_RATIONALE_MAX_LENGTH,
                description=_RATIONALE_DESCRIPTION,
            ),
        ),
        dependencies=(
            list[str],
            Field(
                default_factory=list, max_length=16, description=_DEPENDENCY_DESCRIPTION
            ),
        ),
        is_output=(bool, False),
    )


NewRowwiseFeatureNode = _new_node_model(
    "NewRowwiseFeatureNode",
    "new_rowwise",
    "Create a rowwise child node with the transform(df) ABI.",
    _ROWWISE_CODE_DESCRIPTION,
)
NewAggregateFeatureNode = _new_node_model(
    "NewAggregateFeatureNode",
    "new_aggregate",
    "Create an aggregate child node that may fit state from df_fit/y_fit.",
    _AGGREGATE_CODE_DESCRIPTION,
)


def _node_entry_model(parent_ids: tuple[str, ...]):
    forms = [NewRowwiseFeatureNode, NewAggregateFeatureNode]
    if parent_ids:
        forms.insert(
            0,
            create_model(
                "KeepFeatureNode",
                __config__=ConfigDict(extra="forbid"),
                kind=(
                    Literal["keep"],
                    Field(
                        ...,
                        description=(
                            "Reuse the parent node selected by id, with optional edits."
                        ),
                    ),
                ),
                id=(
                    Literal[parent_ids],
                    Field(
                        ...,
                        description=(
                            "Parent lookup and dependency label; the child id is "
                            "deterministically uniquified."
                        ),
                    ),
                ),
                edits=(NodeEdits, Field(default_factory=NodeEdits)),
                dependencies=(
                    list[str] | None,
                    Field(
                        default=None,
                        max_length=16,
                        description=(
                            f"{_DEPENDENCY_DESCRIPTION} Null preserves resolvable parent edges; "
                            "[] removes them."
                        ),
                    ),
                ),
            ),
        )
    entry: Any = forms[0]
    for form in forms[1:]:
        entry |= form
    return entry


class AllowedDagTabChanges(AllowedChanges):
    """Compile a structured full-child diff into a validated FeatureGraph JSON."""

    def __init__(self, *, min_nodes: int = 1, max_nodes: int = 12):
        if not 1 <= min_nodes <= max_nodes <= 16:
            raise ValueError(
                f"invalid node bounds: min={min_nodes} max={max_nodes}; maximum is 16"
            )
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def build_schema(self, parents: dict[str, str]) -> DiffSchema:
        graphs = self._parse(parents)
        adapter = TypeAdapter(self._diff_model(graphs))
        schema = portable_json_schema(
            {**adapter.json_schema(), "title": "dag_tab_feature_graph_diff"}
        )

        def validate(payload: Any) -> Any:
            try:
                return adapter.validate_python(payload)
            except ValidationError as original:
                repaired = self._repair_payload(payload)
                if repaired is None:
                    raise
                try:
                    validated = adapter.validate_python(repaired)
                except ValidationError:
                    raise original
                logger.warning(
                    "dag_tab_feature_graph_diff: truncated overlong node rationale"
                )
                return validated

        return DiffSchema(json_schema=schema, validate=validate)

    @staticmethod
    def _repair_payload(payload: Any) -> dict | None:
        if not isinstance(payload, dict):
            return None
        repaired = deepcopy(payload)
        changed = False
        nodes = repaired.get("nodes")
        if not isinstance(nodes, list):
            return None
        for node in nodes:
            if not isinstance(node, dict):
                continue
            rationale_owner = node.get("edits") if node.get("kind") == "keep" else node
            if not isinstance(rationale_owner, dict):
                continue
            rationale = rationale_owner.get("rationale")
            if isinstance(rationale, str) and len(rationale) > _RATIONALE_MAX_LENGTH:
                rationale_owner["rationale"] = rationale[
                    :_RATIONALE_MAX_LENGTH
                ].rstrip()
                changed = True
        return repaired if changed else None

    @staticmethod
    def _addressed_nodes(
        graphs: dict[str, FeatureGraph],
    ) -> dict[str, list[FeatureNode]]:
        qualify_ids = len(graphs) > 1
        addressed: dict[str, list[FeatureNode]] = {}
        for namespace, graph in graphs.items():
            id_map = {
                node.id: f"{namespace}_{node.id}" if qualify_ids else node.id
                for node in graph.nodes
            }
            addressed[namespace] = [
                node.model_copy(
                    update={
                        "id": id_map[node.id],
                        "dependencies": [id_map[dep] for dep in node.dependencies],
                    }
                )
                for node in graph.nodes
            ]
        return addressed

    def render_parents(self, parents: dict[str, str]) -> str:
        graphs = self._parse(parents)
        addressed = self._addressed_nodes(graphs)
        blocks: list[str] = []
        for namespace, graph in graphs.items():
            lines = [
                f"=== Parent {namespace} ===",
                f"dataset: {graph.dataset}",
                f"raw_columns: {graph.raw_columns}",
                f"dropped_raw_columns: {graph.dropped_raw_columns}",
                f"target: {graph.target.model_dump() if graph.target else None}",
            ]
            for node in addressed[namespace]:
                lines.extend(
                    [
                        f"id={node.id} | kind={node.kind} | "
                        f"dependencies={node.dependencies} | inputs={node.input_cols} | "
                        f"outputs={node.output_cols} | output_types={node.output_types} | "
                        f"is_output={node.is_output}",
                        f"    rationale: {node.rationale}",
                        "    code:",
                        *[f"        {line}" for line in node.code.splitlines()],
                    ]
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def apply(self, diff: Any, parents: dict[str, str]) -> str:
        graphs = self._parse(parents)
        try:
            # The cap cannot ride on the schema: Gemini 400s on maxItems over an
            # anyOf union, so the description states it and this enforces it.
            if len(diff.nodes) > self.max_nodes:
                raise ValueError(
                    f"child graph has {len(diff.nodes)} nodes but at most "
                    f"{self.max_nodes} are allowed; merge or drop nodes"
                )
            child = self._transcribe(diff, graphs)
            reparsed = FeatureGraph.model_validate_json(child.to_json())
            for node in reparsed.nodes:
                validate_node_code(node)
                tree = ast.parse(node.code)
                transforms = [
                    statement
                    for statement in tree.body
                    if isinstance(statement, ast.FunctionDef)
                    and statement.name == "transform"
                ]
                body = transforms[-1].body if transforms else tree.body
                if not any(isinstance(statement, ast.Return) for statement in body):
                    raise ValueError(f"node {node.id}: code must return a DataFrame")
        except MutationError:
            raise
        except Exception as exc:
            raise MutationError(f"diff_apply_assertion: {exc}") from exc
        return reparsed.to_json()

    def describe(self) -> str:
        return (
            "TABULAR FEATURE-GRAPH DIFF\n"
            "- nodes is the complete child in topological order; omitted parent nodes are "
            "deleted.\n"
            "- base_parent selects the parent whose dataset and raw_columns are inherited; "
            "these fields cannot be changed. dropped_raw_columns is the complete child list "
            "hidden only at the CatBoost boundary; omitted preserves the base list.\n"
            "- kind=keep reuses a rendered parent node; kind=new_rowwise and "
            "kind=new_aggregate create nodes with their named ABI.\n"
            "- dependencies resolve backward by earlier entry id. Literal column reads "
            "synchronize input_cols and consumed edges.\n"
            "- structural_intent and minimum_child_depth are verified against depth from "
            "consumed generated columns.\n"
            "- target_change can keep, drop, or set an invertible regression target transform. "
            "Set bodies implement transform(y_fit, y) and inverse(y_fit, predictions).\n"
            "- Set is_output=true on every node whose generated columns should be passed "
            "to the fixed CatBoost estimator. At least one output node is required.\n"
            "- Declare output_types for generated categorical/binary/ordinal columns; "
            "omitted outputs are numerical. When keep edits output_cols without output_types, "
            "Python preserves types of retained outputs and types new outputs as numerical. "
            "NaN is allowed for numerical outputs, inf is not.\n"
            "- code is trusted Python. Imports are legal; output, row/index, purity, target, "
            "and determinism contracts are enforced."
        )

    def _parse(self, parents: dict[str, str]) -> dict[str, FeatureGraph]:
        if not parents:
            raise MutationError("dag_tab_validation_error: no parents provided")
        graphs: dict[str, FeatureGraph] = {}
        for namespace, code in parents.items():
            try:
                graph = FeatureGraph.model_validate_json(code)
                for node in graph.nodes:
                    validate_node_code(node)
            except Exception as exc:
                raise MutationError(
                    f"dag_tab_validation_error: parent {namespace}: {exc}"
                ) from exc
            graphs[namespace] = graph
        return graphs

    def _diff_model(self, graphs: dict[str, FeatureGraph]) -> type[DagTabDiffBase]:
        addressed = self._addressed_nodes(graphs)
        all_ids = tuple(node.id for nodes in addressed.values() for node in nodes)
        node_entry = _node_entry_model(all_ids)
        return create_model(
            "DagTabFeatureGraphDiff",
            __base__=DagTabDiffBase,
            base_parent=(Literal[tuple(graphs)], ...),
            nodes=(
                list[node_entry],  # type: ignore[valid-type]
                Field(
                    ...,
                    min_length=self.min_nodes,
                    description=(
                        "The complete child graph: exactly these entries, in topological "
                        f"order, at most {self.max_nodes} of them. Dependencies point "
                        "backward to ids of earlier entries; parent nodes omitted here "
                        "are deleted."
                    ),
                ),
            ),
        )

    def _transcribe(
        self, diff: DagTabDiffBase, graphs: dict[str, FeatureGraph]
    ) -> FeatureGraph:
        base = graphs[diff.base_parent]
        addressed = self._addressed_nodes(graphs)
        nodes_by_id = {node.id: node for nodes in addressed.values() for node in nodes}
        child_nodes: list[FeatureNode] = []
        emitted_node_ids: set[str] = set()
        child_id_by_entry_id: dict[str, str] = {}
        output_to_node_id: dict[str, str] = {}

        for entry in diff.nodes:
            if entry.kind == "keep":
                data = nodes_by_id[entry.id].model_dump()
                dependency_refs = (
                    data["dependencies"]
                    if entry.dependencies is None
                    else entry.dependencies
                )
                edits = entry.edits.model_dump(exclude_none=True)
                if "output_cols" in edits and "output_types" not in edits:
                    previous_types = data["output_types"]
                    edits["output_types"] = {
                        column: previous_types.get(column, "numerical")
                        for column in edits["output_cols"]
                    }
                    logger.warning(
                        "dag_tab_feature_graph_diff: synchronized output_types for "
                        "edited node {} after output_cols changed: {}",
                        data["id"],
                        edits["output_types"],
                    )
                data.update(edits)
            else:
                data = entry.model_dump(
                    include={
                        "id",
                        "input_cols",
                        "output_cols",
                        "output_types",
                        "code",
                        "rationale",
                        "is_output",
                    }
                )
                data["kind"] = entry.kind.removeprefix("new_")
                dependency_refs = entry.dependencies

            data["dependencies"] = list(
                dict.fromkeys(
                    child_id_by_entry_id[dependency]
                    for dependency in dependency_refs
                    if dependency in child_id_by_entry_id
                )
            )
            requested_id = entry.id
            suffix = 2
            while data["id"] in emitted_node_ids:
                data["id"] = f"{requested_id}_{suffix}"
                suffix += 1
            data["code"] = normalize_node_code(data["code"])
            literal_reads = literal_frame_reads(data["code"])
            # Repair only what resolves to a real column. A read this cannot place is
            # either a local that shadows the frame — valid code, and refusing it here
            # would delete the idea — or a genuine mistake, which execution reports
            # against the column that is actually missing.
            available_columns = set(base.raw_columns) | set(output_to_node_id)
            missing_declared_reads = (
                (literal_reads & available_columns)
                - set(data["input_cols"])
                - set(data["output_cols"])
            )
            if missing_declared_reads:
                repaired_input_cols = [
                    *data["input_cols"],
                    *sorted(missing_declared_reads),
                ]
                logger.warning(
                    "dag_tab_feature_graph_diff: synchronized input_cols from literal "
                    "code reads for node {}: {} -> {}",
                    data["id"],
                    data["input_cols"],
                    repaired_input_cols,
                )
                data["input_cols"] = repaired_input_cols

            inferred_dependencies = [
                output_to_node_id[column]
                for column in data["input_cols"]
                if column in output_to_node_id
            ]
            repaired_dependencies = list(
                dict.fromkeys([*data["dependencies"], *inferred_dependencies])
            )
            if repaired_dependencies != data["dependencies"]:
                logger.warning(
                    "dag_tab_feature_graph_diff: repaired missing dependencies for "
                    "node {}: {} -> {}",
                    data["id"],
                    data["dependencies"],
                    repaired_dependencies,
                )
                data["dependencies"] = repaired_dependencies

            node = FeatureNode.model_validate(data)
            child_nodes.append(node)
            emitted_node_ids.add(node.id)
            child_id_by_entry_id[entry.id] = node.id
            output_to_node_id.update({column: node.id for column in node.output_cols})

        target = base.target
        if diff.target_change is not None:
            if diff.target_change.kind == "drop":
                target = None
            elif diff.target_change.kind == "set":
                target = {
                    "code": normalize_target_code(diff.target_change.code),
                    "inverse_code": normalize_target_code(
                        diff.target_change.inverse_code, inverse=True
                    ),
                }

        child = FeatureGraph(
            schema_version=base.schema_version,
            dataset=base.dataset,
            raw_columns=base.raw_columns,
            dropped_raw_columns=(
                base.dropped_raw_columns
                if diff.dropped_raw_columns is None
                else diff.dropped_raw_columns
            ),
            target=target,
            nodes=child_nodes,
        )
        required_depth = diff.minimum_child_depth
        if diff.structural_intent == "compose_chain":
            required_depth = max(required_depth or 0, 2)
        elif diff.structural_intent == "extend_chain":
            required_depth = max(required_depth or 0, base.depth + 1, 2)
        if required_depth is not None and child.depth < required_depth:
            raise ValueError(
                f"structural_intent={diff.structural_intent} requires child depth "
                f">= {required_depth}, got {child.depth}; make a later node consume an "
                "earlier generated output and name that node in dependencies"
            )
        return child
