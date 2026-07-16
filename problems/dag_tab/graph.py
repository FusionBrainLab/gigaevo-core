"""Canonical JSON genome for tabular feature DAGs."""

from __future__ import annotations

import json
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeatureNode(BaseModel):
    """One pandas transformation in topological graph order."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="Stable semantic name for the feature transformation.",
    )
    input_cols: list[str] = Field(
        min_length=1,
        description=(
            "Complete set of raw or dependency-produced columns actually read by code."
        ),
    )
    output_cols: list[str] = Field(
        min_length=1,
        description=(
            "New columns invented by this node; code must assign exactly these columns."
        ),
    )
    code: str = Field(
        min_length=1,
        max_length=6000,
        description=(
            "Split-invariant row-wise pandas transform using only input_cols and creating "
            "only output_cols."
        ),
    )
    rationale: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "Counterfactual feature hypothesis: what signal this field exposes, why the "
            "chosen operation matches that mechanism, and what information or robustness "
            "would be lost if the field were omitted."
        ),
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Earlier nodes whose generated output columns this node consumes.",
    )
    is_output: bool = Field(
        default=False,
        description="Whether this node's outputs are exported to the CatBoost estimator.",
    )

    @model_validator(mode="after")
    def validate_local_uniqueness(self) -> Self:
        for label, values in (
            ("input_cols", self.input_cols),
            ("output_cols", self.output_cols),
            ("dependencies", self.dependencies),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"node {self.id}: duplicate {label}")
        if set(self.input_cols) & set(self.output_cols):
            raise ValueError(f"node {self.id}: inputs and outputs overlap")
        return self


class FeatureGraph(BaseModel):
    """Versioned feature graph whose node list is already topologically sorted."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    dataset: str = Field(min_length=1)
    raw_columns: list[str] = Field(min_length=1)
    nodes: list[FeatureNode] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if len(self.raw_columns) != len(set(self.raw_columns)):
            raise ValueError("raw_columns must be unique")

        node_ids: set[str] = set()
        produced: dict[str, str] = {}
        available = set(self.raw_columns)
        has_output = False

        for node in self.nodes:
            if node.id in node_ids:
                raise ValueError(f"duplicate node id {node.id!r}")
            missing_dependencies = set(node.dependencies) - node_ids
            if missing_dependencies:
                raise ValueError(
                    f"node {node.id}: dependencies must reference earlier nodes: "
                    f"{sorted(missing_dependencies)}"
                )
            dependency_outputs = {
                col
                for prior in self.nodes
                if prior.id in node.dependencies
                for col in prior.output_cols
            }
            missing_inputs = set(node.input_cols) - available
            if missing_inputs:
                raise ValueError(
                    f"node {node.id}: unavailable inputs {sorted(missing_inputs)}"
                )
            generated_inputs = set(node.input_cols) - set(self.raw_columns)
            if not generated_inputs <= dependency_outputs:
                raise ValueError(
                    f"node {node.id}: generated inputs must come from declared dependencies"
                )
            for col in node.output_cols:
                if col in available:
                    source = "raw input" if col in self.raw_columns else produced[col]
                    raise ValueError(
                        f"node {node.id}: output {col!r} collides with {source}"
                    )
                produced[col] = node.id
                available.add(col)
            node_ids.add(node.id)
            has_output = has_output or node.is_output

        if not has_output:
            raise ValueError("at least one node must have is_output=true")
        return self

    @property
    def output_columns(self) -> list[str]:
        return [
            col for node in self.nodes if node.is_output for col in node.output_cols
        ]

    @property
    def depth(self) -> int:
        depths: dict[str, int] = {}
        for node in self.nodes:
            depths[node.id] = 1 + max(
                (depths[parent] for parent in node.dependencies), default=0
            )
        return max(depths.values(), default=0)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> FeatureGraph:
        return cls.model_validate_json(text)
