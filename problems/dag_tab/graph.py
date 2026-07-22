"""Canonical JSON genome for tabular feature DAGs."""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

FeatureKind = Literal["rowwise", "aggregate"]
FeatureValueKind = Literal["numerical", "categorical", "binary", "ordinal"]

NODE_INPUT_DESCRIPTION = (
    "Your node receives exactly the columns listed in input_cols, in that order, and nothing "
    "else. df is that slice: df.to_numpy() is your declared inputs as a matrix, not the whole "
    "dataset, and a column you read but did not declare is not there to be read."
)
NODE_KIND_DESCRIPTION = (
    "rowwise receives only df and computes each row independently. aggregate additionally "
    "receives the frozen fitting frame df_fit and its target y_fit, so it can fit state on "
    "the training rows and query it for every row."
)
NODE_CODE_DESCRIPTION = (
    "Trusted Python: either a module defining transform (with optional setup and helper "
    "functions) or a bare transform body. Explicitly create every declared output and end "
    "by returning the frame. "
    + NODE_INPUT_DESCRIPTION
    + " numpy as np, pandas as pd and math are in scope; any "
    "installed third-party library may be imported, including scikit-learn and scipy. Every "
    "stochastic operation must be seeded: the graph is re-executed and must reproduce "
    "identical outputs. rowwise nodes see one row block of df and cannot reference df_fit or "
    "y_fit. aggregate nodes are called once per graph execution with the frozen fitting frame "
    "df_fit, its target y_fit, and the full frame df whose first len(df_fit) rows are df_fit in "
    "order; any deterministic state fitted from df_fit and, when supervised, from y_fit -- "
    "statistics, vocabularies, estimators, search structures -- may then be queried per row "
    "of df. y_fit is a one-dimensional numpy.ndarray, never a pandas object: do not use "
    "y_fit.groupby, .iloc, .loc, .name, or .values; when pandas alignment is needed, first "
    "construct target_fit = pd.Series(np.asarray(y_fit), index=df_fit.index, name='target'). "
    "A row's output must not depend on which other rows of df are present. A supervised "
    "output for a row at a position below len(df_fit) must not depend on that row's own "
    "y_fit: exclude its own contribution by position or by fold. The reserved sample_weight "
    "output sets finite non-negative CatBoost row weights, applied to both the training rows "
    "and the early-stopping eval set; query-row weights are discarded and the column is "
    "stripped from features."
)
NODE_RATIONALE_DESCRIPTION = (
    "Counterfactual feature hypothesis: what signal this field exposes, why the chosen "
    "operation matches that mechanism, and what information or robustness would be lost if "
    "the field were omitted."
)


class TargetTransform(BaseModel):
    """Regression target transform fitted under the same fold contract as features."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        max_length=3000,
        description=(
            "Trusted body of transform(y_fit, y); fit only on y_fit and return a numeric "
            "array aligned with y."
        ),
    )
    inverse_code: str = Field(
        min_length=1,
        max_length=3000,
        description=(
            "Trusted body of inverse(y_fit, predictions); use the same y_fit-fitted "
            "contract and return predictions on the original target scale."
        ),
    )


class FeatureNode(BaseModel):
    """One pandas transformation in topological graph order."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="Stable semantic name for the feature transformation.",
    )
    kind: FeatureKind = Field(
        default="rowwise",
        description=NODE_KIND_DESCRIPTION,
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
    output_types: dict[str, FeatureValueKind] = Field(
        default_factory=dict,
        description=(
            "Semantic type of each generated output. Existing genomes may omit this field; "
            "omitted outputs default to numerical."
        ),
    )
    code: str = Field(
        min_length=1,
        max_length=6000,
        description=NODE_CODE_DESCRIPTION,
    )
    rationale: str = Field(
        min_length=1,
        max_length=1000,
        description=NODE_RATIONALE_DESCRIPTION,
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
        unknown_types = set(self.output_types) - set(self.output_cols)
        if unknown_types:
            raise ValueError(
                f"node {self.id}: output_types contains undeclared outputs "
                f"{sorted(unknown_types)}"
            )
        self.output_types = {
            col: self.output_types.get(col, "numerical") for col in self.output_cols
        }
        return self

    def output_type(self, column: str) -> FeatureValueKind:
        return self.output_types[column]


class FeatureGraph(BaseModel):
    """Versioned feature graph whose node list is already topologically sorted."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    dataset: str = Field(min_length=1)
    raw_columns: list[str] = Field(min_length=1)
    dropped_raw_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Raw columns available to feature nodes but omitted from the estimator matrix."
        ),
    )
    target: TargetTransform | None = Field(
        default=None,
        description="Optional invertible regression target transform.",
    )
    nodes: list[FeatureNode] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if len(self.raw_columns) != len(set(self.raw_columns)):
            raise ValueError("raw_columns must be unique")
        if len(self.dropped_raw_columns) != len(set(self.dropped_raw_columns)):
            raise ValueError("dropped_raw_columns must be unique")
        unknown_dropped = set(self.dropped_raw_columns) - set(self.raw_columns)
        if unknown_dropped:
            raise ValueError(
                "dropped_raw_columns contains unknown raw columns "
                f"{sorted(unknown_dropped)}"
            )

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

        estimator_outputs = [
            column for column in self.output_columns if column != "sample_weight"
        ]
        if self.nodes and (not has_output or not estimator_outputs):
            raise ValueError(
                "non-empty graphs need at least one non-sample_weight node output "
                "with is_output=true"
            )
        return self

    @property
    def output_columns(self) -> list[str]:
        return [
            col for node in self.nodes if node.is_output for col in node.output_cols
        ]

    @property
    def feature_output_columns(self) -> list[str]:
        return [column for column in self.output_columns if column != "sample_weight"]

    @property
    def estimator_columns(self) -> list[str]:
        dropped = set(self.dropped_raw_columns)
        service_columns = [
            column
            for node in self.nodes
            for column in node.output_cols
            if column == "sample_weight"
        ]
        return list(
            dict.fromkeys(
                [
                    *[column for column in self.raw_columns if column not in dropped],
                    *self.feature_output_columns,
                    *service_columns,
                ]
            )
        )

    def _consumed_parents(self, node: FeatureNode) -> list[str]:
        """Declared parents whose outputs the node actually reads.

        A dependency can be declared without being used, so counting declared
        edges lets a functionally flat graph claim a deep archive cell and
        satisfy an extend_chain intent without composing anything.
        """

        generated_inputs = set(node.input_cols) - set(self.raw_columns)
        return [
            parent.id
            for parent in self.nodes
            if parent.id in node.dependencies
            and generated_inputs & set(parent.output_cols)
        ]

    @property
    def depth(self) -> int:
        depths: dict[str, int] = {}
        for node in self.nodes:
            depths[node.id] = 1 + max(
                (depths[parent] for parent in self._consumed_parents(node)), default=0
            )
        return max(depths.values(), default=0)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> FeatureGraph:
        return cls.model_validate_json(text)
