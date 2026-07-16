"""Typed causal, evolutionary, and decision records for memory v2."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.memory.cards import Card

CARD_DELIVERY_TEMPLATE = "[card 1] id={bank_card_id}\n{payload}"

SafetyGateMode = Literal[
    "exclude_confident_incremental_harm",
    "credible_joint_safe",
]


def canonical_digest(payload: Any) -> str:
    """Return a stable SHA-256 digest for JSON-native domain records."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=True,
    )


def qualified_class_name(symbol: type[Any]) -> str:
    """Return the importable identity used when a typed class enters JSON."""

    return f"{symbol.__module__}.{symbol.__qualname__}"


def import_qualified_class(path: str) -> type[Any]:
    """Resolve one canonical class path without accepting arbitrary callables."""

    parts = path.split(".")
    for boundary in range(len(parts) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:boundary]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[boundary:]:
            value = getattr(value, attribute)
        if not isinstance(value, type):
            raise TypeError(f"{path!r} does not resolve to a class")
        return value
    raise ImportError(f"cannot import class {path!r}")


class LLMFingerprint(StrictFrozenModel):
    """Model settings that change the mutation distribution."""

    model_name: str = Field(
        min_length=1,
        description="Configured mutation-model identity, including provider prefix.",
    )
    base_url: str = Field(
        min_length=1,
        description="Inference endpoint serving the configured mutation model.",
    )
    temperature: float = Field(
        ge=0.0,
        description="Sampling temperature used by the mutation model.",
    )


class EnvironmentFingerprint(StrictFrozenModel):
    """Compact run environment used to keep incompatible ledgers separate."""

    task_key: str = Field(
        min_length=1,
        description="Memory task identity used to scope cards and causal evidence.",
    )
    problem_name: str = Field(
        min_length=1,
        description="Resolved problem implementation/configuration identifier.",
    )
    llm: LLMFingerprint = Field(
        description="Mutation-model settings that determine the proposal distribution."
    )
    mutation_operator: type[MutationOperator] = Field(
        description=(
            "Concrete MutationOperator class governing mutation and failure behavior; "
            "this scopes evidence and is not a card feature."
        )
    )
    program_format: str = Field(
        min_length=1,
        description="Program representation consumed and emitted by mutation.",
    )
    pipeline: str = Field(
        min_length=1,
        description="Evaluation pipeline identifier, including memory injection semantics.",
    )
    algorithm: str = Field(
        min_length=1,
        description="Evolution/archive algorithm identifier governing parent contexts.",
    )

    @field_validator("mutation_operator", mode="before")
    @classmethod
    def _resolve_mutation_operator(cls, value: Any) -> type[MutationOperator]:
        try:
            symbol = import_qualified_class(value) if isinstance(value, str) else value
        except (ImportError, AttributeError, TypeError) as exc:
            raise ValueError(f"invalid mutation operator class: {value!r}") from exc
        if (
            not isinstance(symbol, type)
            or not issubclass(symbol, MutationOperator)
            or inspect.isabstract(symbol)
        ):
            raise ValueError(
                "mutation_operator must be a concrete MutationOperator class"
            )
        return symbol

    @field_serializer("mutation_operator", when_used="json")
    def _serialize_mutation_operator(self, value: type[MutationOperator]) -> str:
        return qualified_class_name(value)

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Stable digest of the frozen environment, used to detect accidental "
            "mixing of incompatible decision records."
        )
    )
    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json", exclude={"digest"}))


class LineageCreditConfig(StrictFrozenModel):
    """Fixed-opportunity reward credit for one randomized memory decision."""

    lineage_depth: int = Field(default=1, ge=1)
    opportunity_budget: int = Field(default=32, ge=1)


class RewardDefinition(StrictFrozenModel):
    """Scale and semantics for the modeled evolutionary endpoint."""

    primary_metric: str = Field(min_length=1)
    higher_is_better: bool
    metric_lower_bound: float
    metric_upper_bound: float
    endpoint: Literal["bounded_proximal_utility", "bounded_lineage_utility"] = (
        "bounded_proximal_utility"
    )
    lineage_depth: int = Field(default=1, ge=1)
    lineage_opportunity_budget: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> RewardDefinition:
        if self.metric_upper_bound <= self.metric_lower_bound:
            raise ValueError("primary metric bounds must be finite and increasing")
        expected_endpoint = (
            "bounded_proximal_utility"
            if self.lineage_depth == 1
            else "bounded_lineage_utility"
        )
        if self.endpoint != expected_endpoint:
            raise ValueError(
                f"lineage_depth={self.lineage_depth} requires "
                f"endpoint={expected_endpoint!r}"
            )
        if self.lineage_depth == 1 and self.lineage_opportunity_budget != 1:
            raise ValueError("depth-one reward canonicalizes opportunity budget to 1")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scale(self) -> float:
        return self.metric_upper_bound - self.metric_lower_bound


class CardSnapshot(StrictFrozenModel):
    """One immutable decision-time snapshot of a memory treatment.

    The stable bank lineage is the Bayesian arm. ``payload_sha256`` preserves the
    exact delivered content for audit and replay, but equivalent-card merges do
    not create a statistically cold arm.
    """

    bank_card_id: str = Field(min_length=1)
    absorbed_bank_card_ids: tuple[str, ...] = ()
    treatment_id: str = Field(min_length=1)
    payload_sha256: str = Field(min_length=64, max_length=64)
    render_template: str = Field(
        default="memory_context/single_numbered_card/v1", min_length=1
    )
    payload: str = Field(min_length=1)
    delivered_text: str = Field(min_length=1)
    source_task_key: str = ""
    kind: str = "insight"
    category: str = "general"
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_card(
        cls,
        card: Card,
        *,
        render_template: str = "memory_context/single_numbered_card/v1",
    ) -> CardSnapshot:
        payload = card.description.strip()
        if not payload:
            raise ValueError(f"card {card.id!r} has an empty delivered payload")
        identity = {
            "delivered_text": CARD_DELIVERY_TEMPLATE.format(
                bank_card_id=card.id, payload=payload
            ),
            "render_template": render_template,
        }
        payload_sha256 = canonical_digest(identity)
        return cls(
            bank_card_id=card.id,
            absorbed_bank_card_ids=tuple(sorted(set(card.absorbed_ids))),
            treatment_id=card.id,
            payload_sha256=payload_sha256,
            render_template=render_template,
            payload=payload,
            delivered_text=identity["delivered_text"],
            source_task_key=card.task_key,
            kind=str(card.kind),
            category=card.category,
            keywords=tuple(card.keywords),
        )

    @model_validator(mode="after")
    def _valid_snapshot(self) -> CardSnapshot:
        if any(not card_id for card_id in self.absorbed_bank_card_ids):
            raise ValueError("absorbed bank card ids cannot be empty")
        if self.bank_card_id in self.absorbed_bank_card_ids:
            raise ValueError("a card cannot absorb its own bank id")
        if len(set(self.absorbed_bank_card_ids)) != len(self.absorbed_bank_card_ids):
            raise ValueError("absorbed bank card ids must be unique")
        if self.absorbed_bank_card_ids != tuple(sorted(self.absorbed_bank_card_ids)):
            raise ValueError("absorbed bank card ids must be sorted")
        expected = canonical_digest(
            {
                "delivered_text": self.delivered_text,
                "render_template": self.render_template,
            }
        )
        expected_text = CARD_DELIVERY_TEMPLATE.format(
            bank_card_id=self.bank_card_id, payload=self.payload
        )
        if self.delivered_text != expected_text:
            raise ValueError("delivered_text does not match the singleton wrapper")
        if expected != self.payload_sha256:
            raise ValueError("payload_sha256 does not match the delivered treatment")
        if self.treatment_id != self.bank_card_id:
            raise ValueError("treatment_id must be the stable bank card id")
        return self

    @property
    def bank_lineage_ids(self) -> tuple[str, ...]:
        """All bank ids whose in-flight evidence belongs to this survivor."""

        return (self.bank_card_id, *self.absorbed_bank_card_ids)


class BehaviorCoordinate(StrictFrozenModel):
    """One smooth, replayable coordinate in the decision-time behavior space."""

    key: str = Field(min_length=1)
    raw_value: float
    semantic_normalized: float = Field(ge=0.0, le=1.0)
    dynamic_normalized: float = Field(ge=0.0, le=1.0)
    cell_index: int = Field(ge=0)
    num_bins: int = Field(gt=0)
    dynamic_lower_bound: float
    dynamic_upper_bound: float

    @model_validator(mode="after")
    def _valid_cell(self) -> BehaviorCoordinate:
        if self.cell_index >= self.num_bins:
            raise ValueError("cell_index must be smaller than num_bins")
        if self.dynamic_upper_bound < self.dynamic_lower_bound:
            raise ValueError("dynamic behavior bounds are reversed")
        return self


class MapElitesContext(StrictFrozenModel):
    """Frozen pre-treatment MAP-Elites state used by the contextual model."""

    island_id: str = Field(min_length=1)
    strategy_generation: int = Field(ge=0)
    archive_size: int = Field(ge=0)
    total_cells: int = Field(gt=0)
    coverage: float = Field(ge=0.0, le=1.0)
    parent_quality_quantile: float | None = Field(default=None, ge=0.0, le=1.0)
    parent_cell: tuple[int, ...]
    parent_cell_occupied: bool
    neighbor_occupancy: float = Field(ge=0.0, le=1.0)
    coordinates: tuple[BehaviorCoordinate, ...]
    semantic_schema_hash: str = Field(min_length=64, max_length=64)
    behavior_schema_hash: str = Field(min_length=64, max_length=64)
    archive_fingerprint: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _aligned_coordinates(self) -> MapElitesContext:
        if len(self.parent_cell) != len(self.coordinates):
            raise ValueError("parent cell and behavior coordinates are misaligned")
        if tuple(row.cell_index for row in self.coordinates) != self.parent_cell:
            raise ValueError("coordinate cell indices do not match parent_cell")
        if self.archive_size > self.total_cells:
            raise ValueError("archive_size cannot exceed total_cells")
        return self


class EvolutionContext(StrictFrozenModel):
    """Complete typed pre-treatment context consumed by inference and policy."""

    run_id: str = Field(min_length=1)
    environment: EnvironmentFingerprint
    parent_id: str = Field(min_length=1)
    parent_iteration: int = Field(ge=0)
    parent_generation: int = Field(ge=1)
    parent_metrics: dict[str, float]
    reward: RewardDefinition
    map_elites: MapElitesContext

    @model_validator(mode="after")
    def _finite_metrics(self) -> EvolutionContext:
        bad = [
            key
            for key, value in self.parent_metrics.items()
            if not math.isfinite(value)
        ]
        if bad:
            raise ValueError(f"parent_metrics contain non-finite values: {sorted(bad)}")
        primary = self.reward.primary_metric
        if primary not in self.parent_metrics:
            raise ValueError(f"parent_metrics omit primary metric {primary!r}")
        value = self.parent_metrics[primary]
        if not (
            self.reward.metric_lower_bound - 1e-9
            <= value
            <= self.reward.metric_upper_bound + 1e-9
        ):
            raise ValueError("parent primary metric is outside configured bounds")
        return self


class DecisionKey(StrictFrozenModel):
    """Every immutable input that determines one replayable policy action."""

    run_id: str = Field(min_length=1)
    run_seed: int
    task_key: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    parent_iteration: int = Field(ge=0)
    event_ordinal: int = Field(ge=0)
    environment_hash: str = Field(min_length=64, max_length=64)
    context_hash: str = Field(min_length=64, max_length=64)
    model_config_hash: str = Field(min_length=64, max_length=64)
    evidence_hash: str = Field(min_length=64, max_length=64)
    model_evidence_hash: str = Field(min_length=64, max_length=64)
    candidate_set_hash: str = Field(min_length=64, max_length=64)
    lineage_registry_hash: str = Field(min_length=64, max_length=64)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rng_key(self) -> str:
        return canonical_digest(
            self.model_dump(mode="json", exclude={"rng_key", "decision_id"})
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decision_id(self) -> str:
        return f"memv2-{self.rng_key[:24]}"


class PosteriorPrediction(StrictFrozenModel):
    """Decision-time posterior summary for one immutable card revision."""

    treatment_id: str
    usable_effect_mean: float
    usable_effect_sd: float = Field(ge=0.0)
    probability_helpful: float = Field(ge=0.0, le=1.0)
    usable_gain_control_mean: float
    usable_gain_treated_mean: float
    control_invalid_probability: float = Field(ge=0.0, le=1.0)
    treated_invalid_probability: float = Field(ge=0.0, le=1.0)
    incremental_invalid_probability: float = Field(ge=-1.0, le=1.0)
    treated_invalid_upper: float = Field(ge=0.0, le=1.0)
    incremental_invalid_upper: float = Field(ge=-1.0, le=1.0)
    probability_safe: float = Field(ge=0.0, le=1.0)
    safety_integration_error: float = Field(ge=0.0)
    probability_safe_and_helpful: float = Field(ge=0.0, le=1.0)
    usable_gain_predictive_sd: float = Field(ge=0.0)


class CandidateActionProbability(StrictFrozenModel):
    treatment_id: str
    bank_card_id: str
    proposal_probability: float = Field(ge=0.0, le=1.0)
    proposal_mc_se: float = Field(ge=0.0)
    offer_probability: float | None = Field(default=None, gt=0.0, lt=1.0)
    joint_treated_probability: float = Field(ge=0.0, le=1.0)
    joint_control_probability: float = Field(ge=0.0, le=1.0)
    safe: bool
    prediction: PosteriorPrediction

    @model_validator(mode="after")
    def _coherent_probability_row(self) -> CandidateActionProbability:
        if self.prediction.treatment_id != self.treatment_id:
            raise ValueError("prediction and action treatment ids differ")
        if not self.safe:
            if self.offer_probability is not None:
                raise ValueError("unsafe actions cannot have an offer probability")
            if self.joint_treated_probability or self.joint_control_probability:
                raise ValueError("unsafe actions cannot carry joint policy mass")
            if self.proposal_probability:
                raise ValueError("unsafe actions cannot carry proposal mass")
            return self
        if self.offer_probability is None:
            raise ValueError("safe actions require an offer probability")
        if not math.isclose(
            self.joint_treated_probability,
            self.proposal_probability * self.offer_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("treated joint probability is inconsistent")
        if not math.isclose(
            self.joint_control_probability,
            self.proposal_probability * (1.0 - self.offer_probability),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("control joint probability is inconsistent")
        return self


class PolicySpecification(StrictFrozenModel):
    """Complete durable behavior-policy configuration for replay and audit."""

    name: Literal["chance_constrained_probability_matching"] = (
        "chance_constrained_probability_matching"
    )
    safety_gate_mode: SafetyGateMode
    max_treated_invalid_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    max_incremental_invalid_probability: float = Field(ge=-1.0, le=1.0)
    safety_alpha: float = Field(gt=0.0, lt=0.5)
    offer_probability: float = Field(gt=0.0, lt=1.0)
    proposal_exploration_probability: float = Field(ge=0.0, lt=1.0)
    posterior_summary_samples: int = Field(ge=128)
    proposal_worlds: int = Field(ge=64)
    abstain_effect: float
    max_pending_per_card: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_safety_gate(self) -> PolicySpecification:
        if self.safety_gate_mode == "exclude_confident_incremental_harm":
            if self.max_treated_invalid_probability is not None:
                raise ValueError(
                    "incremental-harm mode requires the absolute invalidity limit "
                    "to be disabled"
                )
        elif self.max_treated_invalid_probability is None:
            raise ValueError(
                "credible-joint-safe mode requires an absolute invalidity limit"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json", exclude={"digest"}))


class PosteriorFitDiagnostics(StrictFrozenModel):
    """Decision-time numerical health needed for posterior monitoring."""

    evidence_count: int = Field(ge=0)
    reward_observations: int = Field(ge=0)
    safety_observations: int = Field(ge=0)
    reward_residual_sd: float = Field(gt=0.0)
    reward_card_effect_sd: float = Field(gt=0.0)
    reward_optimizer_method: str = Field(min_length=1)
    reward_optimizer_success: bool
    reward_optimizer_iterations: int = Field(ge=0)
    reward_hyperparameters_at_boundary: bool
    reward_residual_scale_ess: float = Field(gt=0.0)
    reward_quadrature_error: float = Field(ge=0.0)
    reward_residual_moment_error: float = Field(ge=0.0)
    reward_coefficient_mean_error: float = Field(ge=0.0)
    reward_residual_boundary_probability: float = Field(ge=0.0, le=1.0)
    reward_residual_upper_boundary_probability: float = Field(ge=0.0, le=1.0)
    safety_optimizer_method: str = Field(min_length=1)
    safety_optimizer_iterations: int = Field(ge=0)
    safety_objective: float
    safety_gradient_inf: float = Field(ge=0.0)
    safety_hessian_condition: float = Field(ge=1.0)
    offer_probability_hash: str = Field(min_length=64, max_length=64)


class DecisionRecord(StrictFrozenModel):
    """Durable write-ahead row for the complete two-stage memory action."""

    decision_id: str
    run_seed: int
    attempt_id: str = Field(min_length=1)
    event_ordinal: int = Field(ge=0)
    rng_key: str = Field(min_length=64, max_length=64)
    evidence_hash: str = Field(min_length=64, max_length=64)
    model_evidence_hash: str = Field(min_length=64, max_length=64)
    candidate_set_hash: str = Field(min_length=64, max_length=64)
    lineage_registry_hash: str = Field(min_length=64, max_length=64)
    context_hash: str = Field(min_length=64, max_length=64)
    model_config_hash: str = Field(min_length=64, max_length=64)
    posterior_config_hash: str = Field(min_length=64, max_length=64)
    policy: PolicySpecification
    fit_diagnostics: PosteriorFitDiagnostics
    context: EvolutionContext
    lineage_registry: tuple[CardSnapshot, ...]
    fitted_observation_ids: tuple[str, ...] = ()
    candidates: tuple[CardSnapshot, ...]
    action_probabilities: tuple[CandidateActionProbability, ...]
    pending_by_treatment: dict[str, int] = Field(default_factory=dict)
    pending_by_bank_card: dict[str, int] = Field(default_factory=dict)
    censored_count: int = Field(default=0, ge=0)
    ineligible_count: int = Field(default=0, ge=0)
    abstain_probability: float = Field(ge=0.0, le=1.0)
    proposed_treatment_id: str | None = None
    delivered: bool = False
    offer_probability: float | None = Field(default=None, gt=0.0, lt=1.0)
    proposal_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    joint_action_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    reward_q_hat_control: float | None = None
    reward_q_hat_treated: float | None = None
    risk_q_hat_control: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_q_hat_treated: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent_action(self) -> DecisionRecord:
        candidate_ids = {row.treatment_id for row in self.candidates}
        registry_ids = {row.treatment_id for row in self.lineage_registry}
        probability_ids = {row.treatment_id for row in self.action_probabilities}
        if len(candidate_ids) != len(self.candidates):
            raise ValueError("candidate treatment ids must be unique")
        if len(probability_ids) != len(self.action_probabilities):
            raise ValueError("action-probability treatment ids must be unique")
        if len(registry_ids) != len(self.lineage_registry):
            raise ValueError("lineage-registry treatment ids must be unique")
        if not candidate_ids <= registry_ids:
            raise ValueError("candidate set is outside the frozen lineage registry")
        if len(set(self.fitted_observation_ids)) != len(self.fitted_observation_ids):
            raise ValueError("fitted observation ids must be unique")
        if candidate_ids != probability_ids:
            raise ValueError("candidate and action-probability sets differ")
        if candidate_set_hash(self.candidates) != self.candidate_set_hash:
            raise ValueError("candidate_set_hash does not match candidates")
        if candidate_set_hash(self.lineage_registry) != self.lineage_registry_hash:
            raise ValueError("lineage_registry_hash does not match its revisions")
        expected_context_hash = canonical_digest(
            self.context.model_dump(mode="json", exclude_computed_fields=True)
        )
        if self.context_hash != expected_context_hash:
            raise ValueError("context_hash does not match the frozen decision context")
        expected_config_hash = canonical_digest(
            {
                "posterior": self.posterior_config_hash,
                "policy": self.policy.model_dump(mode="json", exclude={"digest"}),
            }
        )
        if expected_config_hash != self.model_config_hash:
            raise ValueError("model_config_hash omits posterior or policy settings")
        candidate_banks = {
            row.treatment_id: row.bank_card_id for row in self.candidates
        }
        if any(
            candidate_banks[row.treatment_id] != row.bank_card_id
            for row in self.action_probabilities
        ):
            raise ValueError("candidate and probability bank lineages differ")
        key = DecisionKey(
            run_id=self.context.run_id,
            run_seed=self.run_seed,
            task_key=self.context.environment.task_key,
            parent_id=self.context.parent_id,
            attempt_id=self.attempt_id,
            parent_iteration=self.context.parent_iteration,
            event_ordinal=self.event_ordinal,
            environment_hash=self.context.environment.digest,
            context_hash=self.context_hash,
            model_config_hash=self.model_config_hash,
            evidence_hash=self.evidence_hash,
            model_evidence_hash=self.model_evidence_hash,
            candidate_set_hash=self.candidate_set_hash,
            lineage_registry_hash=self.lineage_registry_hash,
        )
        if key.rng_key != self.rng_key or key.decision_id != self.decision_id:
            raise ValueError("decision identity does not match its immutable inputs")
        total = self.abstain_probability + sum(
            row.proposal_probability for row in self.action_probabilities
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("proposal probabilities and abstention must sum to one")
        if self.proposed_treatment_id is None:
            fields = (
                self.offer_probability,
                self.proposal_probability,
                self.joint_action_probability,
                self.reward_q_hat_control,
                self.reward_q_hat_treated,
                self.risk_q_hat_control,
                self.risk_q_hat_treated,
            )
            if self.delivered or any(value is not None for value in fields):
                raise ValueError("abstention carries treatment fields")
            return self
        if self.proposed_treatment_id not in candidate_ids:
            raise ValueError("proposed treatment is outside the candidate set")
        required = (
            self.offer_probability,
            self.proposal_probability,
            self.joint_action_probability,
            self.reward_q_hat_control,
            self.reward_q_hat_treated,
            self.risk_q_hat_control,
            self.risk_q_hat_treated,
        )
        if any(value is None for value in required):
            raise ValueError("a proposal requires propensities and frozen q-hats")
        assert self.offer_probability is not None
        assert self.proposal_probability is not None
        assert self.joint_action_probability is not None
        assert self.reward_q_hat_control is not None
        assert self.reward_q_hat_treated is not None
        assert self.risk_q_hat_control is not None
        assert self.risk_q_hat_treated is not None
        reward_scale = self.context.reward.scale
        if any(
            abs(float(value)) > reward_scale + 1e-12
            for value in (self.reward_q_hat_control, self.reward_q_hat_treated)
        ):
            raise ValueError("reward q-hats exceed the bounded proximal-gain scale")
        action = next(
            row
            for row in self.action_probabilities
            if row.treatment_id == self.proposed_treatment_id
        )
        if not action.safe:
            raise ValueError("the proposed treatment is unsafe")
        assert action.offer_probability is not None
        if not math.isclose(
            action.proposal_probability,
            self.proposal_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            action.offer_probability,
            self.offer_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selected action propensities differ from the policy row")
        expected_joint = (
            action.joint_treated_probability
            if self.delivered
            else action.joint_control_probability
        )
        if not math.isclose(
            expected_joint,
            self.joint_action_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selected joint propensity differs from the policy row")
        return self


class OutcomeMeasurement(StrictFrozenModel):
    """Terminal proximal gain with honest evaluation uncertainty."""

    value: float
    se: float | None = Field(default=None, ge=0.0)
    n_pairs: int | None = Field(default=None, ge=2)
    kind: Literal["scalar", "paired", "deterministic"]
    pairing_signature: str = ""

    @model_validator(mode="after")
    def _kind_contract(self) -> OutcomeMeasurement:
        if self.kind == "paired":
            if self.se is None or self.n_pairs is None or not self.pairing_signature:
                raise ValueError(
                    "paired measurements require se, n_pairs, and pairing_signature"
                )
        elif self.kind == "scalar":
            if (
                self.se is not None
                or self.n_pairs is not None
                or self.pairing_signature
            ):
                raise ValueError("scalar measurement uncertainty must remain unknown")
        elif self.se != 0.0 or self.n_pairs is not None or self.pairing_signature:
            raise ValueError("deterministic measurements require only se=0")
        return self


class TerminalOutcome(StrictFrozenModel):
    """Exactly one terminal state for a previously persisted decision."""

    decision_id: str
    child_id: str
    base_id: str
    primary_metric: str = Field(min_length=1)
    higher_is_better: bool
    ope_eligible: bool
    status: Literal["outcome", "invalid", "censored"]
    measurement: OutcomeMeasurement | None = None
    censor_reason: str = ""
    failure_stage: str = ""
    completion_ordinal: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _terminal_contract(self) -> TerminalOutcome:
        if self.status == "outcome" and self.measurement is None:
            raise ValueError("valid terminal outcome requires a measurement")
        if self.status != "outcome" and self.measurement is not None:
            raise ValueError(
                "invalid/censored terminal cannot carry a gain measurement"
            )
        if self.status == "censored" and not self.censor_reason:
            raise ValueError("censored terminal requires a reason")
        return self


class CausalObservation(StrictFrozenModel):
    """One reconciled randomized card/control terminal used for inference."""

    decision_id: str
    event_ordinal: int = Field(ge=0)
    card: CardSnapshot
    context: EvolutionContext
    treatment: bool
    offer_propensity: float = Field(gt=0.0, lt=1.0)
    proposal_propensity: float = Field(gt=0.0, le=1.0)
    joint_action_propensity: float = Field(gt=0.0, le=1.0)
    status: Literal["outcome", "invalid"]
    measurement: OutcomeMeasurement | None = None
    reward_q_hat_control: float
    reward_q_hat_treated: float
    risk_q_hat_control: float = Field(ge=0.0, le=1.0)
    risk_q_hat_treated: float = Field(ge=0.0, le=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def invalid(self) -> bool:
        return self.status == "invalid"

    @model_validator(mode="after")
    def _observed_action_contract(self) -> CausalObservation:
        if self.invalid and self.measurement is not None:
            raise ValueError("invalid observations cannot carry a gain measurement")
        if not self.invalid and self.measurement is None:
            raise ValueError("valid observations require a gain measurement")
        expected = self.proposal_propensity * (
            self.offer_propensity if self.treatment else 1.0 - self.offer_propensity
        )
        if not math.isclose(
            self.joint_action_propensity,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("joint action propensity is inconsistent")
        return self


def validate_reward_observations(
    observations: tuple[CausalObservation, ...],
    reward_observations: tuple[CausalObservation, ...],
) -> None:
    """Enforce one delayed measurement for an existing randomized decision."""

    immediate_by_id = {row.decision_id: row for row in observations}
    if len(immediate_by_id) != len(observations):
        raise ValueError("immediate evidence contains duplicate decision ids")
    reward_by_id = {row.decision_id: row for row in reward_observations}
    if len(reward_by_id) != len(reward_observations):
        raise ValueError("reward evidence contains duplicate decision ids")
    for decision_id, reward_row in reward_by_id.items():
        immediate = immediate_by_id.get(decision_id)
        if immediate is None:
            raise ValueError("reward evidence is not a subset of immediate evidence")
        reward_payload = reward_row.model_dump(mode="json", exclude={"measurement"})
        immediate_payload = immediate.model_dump(mode="json", exclude={"measurement"})
        if reward_payload != immediate_payload:
            raise ValueError(
                "delayed reward may change only the outcome measurement for "
                f"decision {decision_id!r}"
            )


class LineageOutcome(StrictFrozenModel):
    """Replayable fixed-horizon reward endpoint for one randomized decision."""

    decision_id: str = Field(min_length=1)
    root_child_id: str = Field(min_length=1)
    status: Literal["outcome", "invalid", "pending", "censored"]
    lineage_depth: int = Field(ge=1)
    opportunity_budget: int = Field(ge=1)
    opportunities_observed: int = Field(default=0, ge=0)
    maturity_ordinal: int | None = Field(default=None, ge=0)
    observed_through_ordinal: int = Field(ge=0)
    descendant_count: int = Field(default=0, ge=0)
    best_descendant_id: str = ""
    best_depth: int | None = Field(default=None, ge=1)
    measurement: OutcomeMeasurement | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _lineage_contract(self) -> LineageOutcome:
        if self.status == "outcome":
            if self.measurement is None:
                raise ValueError("lineage outcome requires a measurement")
            if self.maturity_ordinal is None:
                raise ValueError("lineage outcome requires a maturity ordinal")
            if not self.best_descendant_id or self.best_depth is None:
                raise ValueError("lineage outcome requires its best descendant")
            if self.descendant_count < 1:
                raise ValueError("lineage outcome requires at least one descendant")
        elif self.measurement is not None:
            raise ValueError("non-outcome lineage states cannot carry a measurement")
        if self.status != "outcome" and (
            self.descendant_count
            or self.best_descendant_id
            or self.best_depth is not None
        ):
            raise ValueError("non-outcome lineage states cannot name descendants")
        if self.best_depth is not None and self.best_depth > self.lineage_depth:
            raise ValueError("best descendant exceeds configured lineage depth")
        if self.opportunities_observed > self.opportunity_budget:
            raise ValueError("observed opportunities exceed the configured budget")
        if self.lineage_depth == 1 and self.status == "pending":
            raise ValueError("depth-one lineage outcomes cannot remain pending")
        if self.lineage_depth > 1 and self.status in ("outcome", "invalid"):
            if self.opportunities_observed != self.opportunity_budget:
                raise ValueError("mature lineage outcome lacks its opportunity budget")
        if self.status == "pending":
            if (
                self.opportunities_observed < self.opportunity_budget
                and self.maturity_ordinal is not None
            ):
                raise ValueError("an unenrolled opportunity window has no maturity")
            if (
                self.opportunities_observed == self.opportunity_budget
                and self.maturity_ordinal is None
            ):
                raise ValueError("a fully enrolled opportunity window needs maturity")
        elif self.maturity_ordinal is None:
            raise ValueError("terminal lineage states require a maturity ordinal")
        if (
            self.maturity_ordinal is not None
            and self.maturity_ordinal > self.observed_through_ordinal
        ):
            raise ValueError("lineage maturity exceeds the observed ledger")
        if self.status == "censored" and not self.reason:
            raise ValueError("censored lineage outcome requires a reason")
        return self


class EvidenceSnapshot(StrictFrozenModel):
    version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    observations: tuple[CausalObservation, ...] = ()
    reward_observations: tuple[CausalObservation, ...] = ()
    lineage_outcomes: tuple[LineageOutcome, ...] = ()
    pending_by_treatment: dict[str, int] = Field(default_factory=dict)
    pending_by_bank_card: dict[str, int] = Field(default_factory=dict)
    censored_count: int = Field(default=0, ge=0)
    ineligible_count: int = Field(default=0, ge=0)
    conflict_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _aligned_evidence(self) -> EvidenceSnapshot:
        validate_reward_observations(self.observations, self.reward_observations)
        lineage_by_id = {row.decision_id: row for row in self.lineage_outcomes}
        if len(lineage_by_id) != len(self.lineage_outcomes):
            raise ValueError("lineage evidence contains duplicate decision ids")
        if self.lineage_outcomes:
            mature_ids = {
                row.decision_id
                for row in self.lineage_outcomes
                if row.status in ("outcome", "invalid")
            }
            reward_ids = {row.decision_id for row in self.reward_observations}
            if mature_ids != reward_ids:
                raise ValueError("mature lineage and reward evidence ids differ")
            for reward_row in self.reward_observations:
                lineage = lineage_by_id[reward_row.decision_id]
                if lineage.status != reward_row.status:
                    raise ValueError("lineage and reward terminal states differ")
                if lineage.measurement != reward_row.measurement:
                    raise ValueError("lineage and reward measurements differ")
        return self

    @property
    def posterior_reward_observations(self) -> tuple[CausalObservation, ...]:
        if self.reward_observations:
            return self.reward_observations
        if self.observations and all(
            row.context.reward.lineage_depth == 1 for row in self.observations
        ):
            return self.observations
        return ()


class PolicyDecision(StrictFrozenModel):
    proposed_card: CardSnapshot | None = None
    delivered: bool = False
    offer_probability: float | None = Field(default=None, gt=0.0, lt=1.0)
    proposal_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    joint_action_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    action_probabilities: tuple[CandidateActionProbability, ...] = ()
    abstain_probability: float = Field(ge=0.0, le=1.0)
    sampled_effects: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _action_contract(self) -> PolicyDecision:
        total = self.abstain_probability + sum(
            row.proposal_probability for row in self.action_probabilities
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("policy proposal probabilities must sum to one")
        if self.proposed_card is None:
            if self.delivered or any(
                value is not None
                for value in (
                    self.offer_probability,
                    self.proposal_probability,
                    self.joint_action_probability,
                )
            ):
                raise ValueError("abstention contains treatment fields")
            return self
        action = next(
            (
                row
                for row in self.action_probabilities
                if row.treatment_id == self.proposed_card.treatment_id
            ),
            None,
        )
        if action is None or not action.safe:
            raise ValueError("proposed card is absent or unsafe")
        expected_joint = (
            action.joint_treated_probability
            if self.delivered
            else action.joint_control_probability
        )
        if (
            self.offer_probability != action.offer_probability
            or self.proposal_probability != action.proposal_probability
            or self.joint_action_probability != expected_joint
        ):
            raise ValueError("selected policy fields differ from the action row")
        return self


def candidate_set_hash(cards: tuple[CardSnapshot, ...]) -> str:
    return canonical_digest(
        [
            card.model_dump(mode="json")
            for card in sorted(cards, key=lambda row: row.treatment_id)
        ]
    )
