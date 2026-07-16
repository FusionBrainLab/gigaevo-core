"""Schema-constrained positional-slot mutations for FeatureGraph JSON genomes."""

from __future__ import annotations

from typing import Any, Literal

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
    model_validator,
)

from gigaevo.evolution.mutation.allowed_changes import (
    AllowedChanges,
    DiffSchema,
    DiffStructuredOutputBase,
)
from gigaevo.exceptions import MutationError
from gigaevo.llm.schema_compat import portable_json_schema
from problems.dag_tab.execution import normalize_node_code, validate_node_code
from problems.dag_tab.graph import FeatureGraph, FeatureNode


class DagTabDiffBase(DiffStructuredOutputBase):
    """Shared evidence fields plus dynamically generated graph slots."""


_NODE_CODE_DESCRIPTION = (
    "Concise pandas transform body. Explicitly create every declared output and "
    "end with `return df`; the runtime safely appends that final statement if omitted."
)


class NodeEdits(BaseModel):
    """Fields that a retained parent node may change."""

    model_config = ConfigDict(extra="forbid")

    input_cols: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Complete set of columns the edited code reads.",
    )
    output_cols: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Exact set of new feature columns the edited code assigns.",
    )
    code: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description=_NODE_CODE_DESCRIPTION,
    )
    rationale: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description=(
            "Counterfactual hypothesis: signal exposed, why this operation is appropriate, "
            "and what becomes worse or unavailable without the feature."
        ),
    )
    is_output: bool | None = None


def _slots_contiguous(diff: BaseModel) -> BaseModel:
    first_empty: str | None = None
    for name in type(diff).model_fields:
        if not name.startswith("slot_"):
            continue
        if getattr(diff, name) is None:
            first_empty = first_empty or name
        elif first_empty is not None:
            raise ValueError(
                f"{name} is filled after empty {first_empty}; slots must be contiguous"
            )
    return diff


def _parent_ids(namespace: str, graph: FeatureGraph) -> list[str]:
    return [f"{namespace.lower()}{index}" for index in range(1, len(graph.nodes) + 1)]


def _slot_models(position: int, parent_ids: tuple[str, ...]) -> tuple[type, type]:
    dependency_field: dict[str, Any] = {}
    if position > 1:
        dependency_ref: Any = Literal[
            tuple(f"slot_{index}" for index in range(1, position))
        ]
        dependency_field = {
            "dependencies": (
                list[dependency_ref],  # type: ignore[valid-type]
                Field(
                    default_factory=list,
                    max_length=position - 1,
                    description=(
                        "Earlier slots whose generated columns this node builds on: "
                        "list slot_k here, then read its output_cols in input_cols "
                        "and code to compose features from already-constructed "
                        "features. Leave empty when reading only raw xN columns."
                    ),
                ),
            )
        }

    keep = create_model(
        f"KeepFeatureNode{position}",
        __config__=ConfigDict(extra="forbid"),
        kind=(Literal["keep"], ...),
        id=(Literal[parent_ids], ...),
        edits=(NodeEdits, Field(default_factory=NodeEdits)),
        **dependency_field,
    )
    new = create_model(
        f"NewFeatureNode{position}",
        __config__=ConfigDict(extra="forbid"),
        kind=(Literal["new"], ...),
        id=(str, Field(..., min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")),
        input_cols=(
            list[str],
            Field(..., min_length=1, description="Complete set of columns code reads."),
        ),
        output_cols=(
            list[str],
            Field(
                ..., min_length=1, description="Exact set of new columns code assigns."
            ),
        ),
        code=(
            str,
            Field(
                ...,
                min_length=1,
                max_length=2000,
                description=_NODE_CODE_DESCRIPTION,
            ),
        ),
        rationale=(
            str,
            Field(
                ...,
                min_length=1,
                max_length=1000,
                description=(
                    "Counterfactual hypothesis: signal exposed, why the operation matches "
                    "it, and what is lost without this feature."
                ),
            ),
        ),
        is_output=(bool, False),
        **dependency_field,
    )
    return keep, new


class AllowedDagTabChanges(AllowedChanges):
    """Compile a structured full-child diff into a validated FeatureGraph JSON."""

    def __init__(self, *, min_nodes: int = 1, max_nodes: int = 8):
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
                repaired = self._compact_payload(payload)
                if repaired is None:
                    raise
                try:
                    validated = adapter.validate_python(repaired)
                except ValidationError:
                    raise original
                logger.warning(
                    "dag_tab_feature_graph_diff: repaired non-contiguous slot payload"
                )
                return validated

        return DiffSchema(json_schema=schema, validate=validate)

    @staticmethod
    def _compact_payload(payload: Any) -> dict | None:
        if not isinstance(payload, dict):
            return None
        rebuilt: dict[str, Any] = {}
        try:
            indices: dict[str, int] = {}
            for key in payload:
                if not (isinstance(key, str) and key.startswith("slot_")):
                    continue
                index = int(key.removeprefix("slot_"))
                if index < 1 or key != f"slot_{index}":
                    return None
                indices[key] = index
            names = sorted(indices, key=indices.__getitem__)
            filled = [
                (indices[key], payload[key])
                for key in names
                if payload[key] is not None
            ]
            old_indices = [index for index, _ in filled]
            if old_indices == list(range(1, len(old_indices) + 1)):
                return None
            remap = {
                old_index: new_index
                for new_index, (old_index, _) in enumerate(filled, start=1)
            }
            for new_index, (_, body) in enumerate(filled, start=1):
                if not isinstance(body, dict):
                    return None
                body = dict(body)
                dependencies = body.get("dependencies")
                if dependencies is not None:
                    if not isinstance(dependencies, list):
                        return None
                    remapped_dependencies: list[str] = []
                    for reference in dependencies:
                        if not isinstance(reference, str):
                            return None
                        old_dependency = int(reference.removeprefix("slot_"))
                        if (
                            reference != f"slot_{old_dependency}"
                            or old_dependency not in remap
                        ):
                            return None
                        remapped_dependencies.append(f"slot_{remap[old_dependency]}")
                    body["dependencies"] = remapped_dependencies
                rebuilt[f"slot_{new_index}"] = body
        except (TypeError, ValueError):
            return None
        result = {
            key: value for key, value in payload.items() if not key.startswith("slot_")
        }
        result.update(rebuilt)
        return result

    def render_parents(self, parents: dict[str, str]) -> str:
        graphs = self._parse(parents)
        blocks: list[str] = []
        for namespace, graph in graphs.items():
            lines = [
                f"=== Parent {namespace} ===",
                f"dataset: {graph.dataset}",
                f"raw_columns: {graph.raw_columns}",
            ]
            stable_ids = _parent_ids(namespace, graph)
            node_to_stable = {
                node.id: stable for node, stable in zip(graph.nodes, stable_ids)
            }
            for stable, node in zip(stable_ids, graph.nodes):
                dependencies = [node_to_stable[dep] for dep in node.dependencies]
                lines.extend(
                    [
                        f"{stable} | node_id={node.id} | deps={dependencies} | "
                        f"inputs={node.input_cols} | outputs={node.output_cols} | "
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
            child = self._transcribe(diff, graphs)
            reparsed = FeatureGraph.model_validate_json(child.to_json())
            for node in reparsed.nodes:
                validate_node_code(node)
        except MutationError:
            raise
        except Exception as exc:
            raise MutationError(f"diff_apply_assertion: {exc}") from exc
        return reparsed.to_json()

    def describe(self) -> str:
        return (
            "POSITIONAL-SLOT TABULAR FEATURE-GRAPH DIFF\n"
            f"- Emit the complete child as consecutive slot_1..slot_{self.max_nodes}; "
            f"fill {self.min_nodes}..{self.max_nodes} slots and set unused trailing "
            "slots to null.\n"
            "- base_parent selects the parent whose dataset and raw_columns are inherited; "
            "these fields cannot be changed.\n"
            "- kind=keep reuses a rendered parent node and may edit its feature contract; "
            "kind=new creates a node. Omitted parent nodes are deleted.\n"
            "- dependencies refer only to earlier slots in the NEW child. Generated "
            "input columns must be outputs of those dependencies; raw xN columns need no "
            "dependency. Reordering slots and changing dependencies rewires the graph.\n"
            "- Set is_output=true on every node whose generated columns should be passed "
            "to the fixed CatBoost estimator. At least one output node is required.\n"
            "- code is a pandas function body. Explicitly assign every output as "
            "df['name'] = ..., use only declared inputs, preserve rows/index, create no "
            "undeclared columns, and finish with return df. np and pd are available; "
            "imports, target access, files, network, eval, and exec are forbidden."
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
        all_ids = tuple(
            stable
            for namespace, graph in graphs.items()
            for stable in _parent_ids(namespace, graph)
        )
        fields: dict[str, Any] = {}
        for position in range(1, self.max_nodes + 1):
            keep, new = _slot_models(position, all_ids)
            fields[f"slot_{position}"] = (
                (keep | new, ...)
                if position <= self.min_nodes
                else (keep | new | None, None)
            )
        return create_model(
            "DagTabFeatureGraphDiff",
            __base__=DagTabDiffBase,
            base_parent=(Literal[tuple(graphs)], ...),
            **fields,
            __validators__={
                "check_contiguous": model_validator(mode="after")(_slots_contiguous)  # type: ignore[dict-item]
            },
        )

    def _transcribe(
        self, diff: DagTabDiffBase, graphs: dict[str, FeatureGraph]
    ) -> FeatureGraph:
        base = graphs[diff.base_parent]
        nodes_by_stable = {
            stable: node
            for namespace, graph in graphs.items()
            for stable, node in zip(_parent_ids(namespace, graph), graph.nodes)
        }
        child_nodes: list[FeatureNode] = []
        slot_to_node_id: dict[str, str] = {}

        for position in range(1, self.max_nodes + 1):
            slot_name = f"slot_{position}"
            slot = getattr(diff, slot_name)
            if slot is None:
                break
            dependency_refs = slot.dependencies if position > 1 else []
            dependencies = [slot_to_node_id[ref] for ref in dependency_refs]
            if slot.kind == "keep":
                data = nodes_by_stable[slot.id].model_dump()
                data.update(slot.edits.model_dump(exclude_none=True))
                data["dependencies"] = dependencies
            else:
                data = slot.model_dump(
                    include={
                        "id",
                        "input_cols",
                        "output_cols",
                        "code",
                        "rationale",
                        "is_output",
                    }
                )
                data["dependencies"] = dependencies
            data["code"] = normalize_node_code(data["code"])
            node = FeatureNode.model_validate(data)
            child_nodes.append(node)
            slot_to_node_id[slot_name] = node.id

        return FeatureGraph(
            schema_version=base.schema_version,
            dataset=base.dataset,
            raw_columns=base.raw_columns,
            nodes=child_nodes,
        )
