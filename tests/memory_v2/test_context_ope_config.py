from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pytest

from gigaevo.config.validation import validate_memory_v2_scope
from gigaevo.evolution.mutation.structured_diff import StructuredDiffMutationOperator
from gigaevo.evolution.strategies.models import (
    BehaviorModelTransform,
    BehaviorSpace,
    DynamicBehaviorSpace,
    LinearBinning,
)
from gigaevo.memory.write.eviction import NullEvictor
from gigaevo.memory_v2.candidates import (
    AgenticApplicabilityProvider,
    NullApplicabilityProvider,
    WholeBankCandidateSource,
)
from gigaevo.memory_v2.context import MapContextConfig, MapElitesContextSource
from gigaevo.memory_v2.eviction import CausalRetirementEvictor
from gigaevo.memory_v2.features import FeatureConfig, HierarchicalFeatureMap
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    DecisionKey,
    EnvironmentFingerprint,
    EvolutionContext,
    LineageCreditConfig,
    OutcomeMeasurement,
    canonical_digest,
)
from gigaevo.memory_v2.ope import ConditionalOfferDREvaluator, PreDecisionUnit
from gigaevo.memory_v2.transfer import CrossTaskUsefulnessConfig
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program


def test_full_audit_evidence_does_not_perturb_policy_rng(
    evolution_context: EvolutionContext,
) -> None:
    key = DecisionKey(
        run_id=evolution_context.run_id,
        run_seed=7,
        task_key=evolution_context.environment.task_key,
        parent_id=evolution_context.parent_id,
        attempt_id="attempt",
        parent_iteration=evolution_context.parent_iteration,
        event_ordinal=3,
        environment_hash=evolution_context.environment.digest,
        context_hash=canonical_digest(
            evolution_context.model_dump(
                mode="json",
                exclude_computed_fields=True,
            )
        ),
        model_config_hash="a" * 64,
        evidence_hash="b" * 64,
        model_evidence_hash="c" * 64,
        candidate_set_hash="d" * 64,
        lineage_registry_hash="e" * 64,
        pending_by_bank_card={},
        lineage_pending_by_bank_card={},
        assessed_bank_card_ids=(),
        applicable_bank_card_ids=(),
    )

    audit_only_change = key.model_copy(update={"evidence_hash": "f" * 64})
    model_change = key.model_copy(update={"model_evidence_hash": "f" * 64})
    pending_change = key.model_copy(update={"pending_by_bank_card": {"card": 1}})

    assert audit_only_change.rng_key == key.rng_key
    assert audit_only_change.decision_id == key.decision_id
    assert model_change.rng_key != key.rng_key
    assert model_change.decision_id != key.decision_id
    assert pending_change.rng_key != key.rng_key
    assert pending_change.decision_id != key.decision_id


@pytest.mark.asyncio
async def test_map_elites_context_is_typed_smooth_and_frozen(
    environment: EnvironmentFingerprint,
) -> None:
    space = BehaviorSpace(
        bins={
            "hop_depth": LinearBinning(min_val=0.0, max_val=5.0, num_bins=5),
            "passages_fetched": LinearBinning(
                min_val=0.0,
                max_val=45.0,
                num_bins=5,
                model_transform=BehaviorModelTransform(
                    lower_bound=0.0,
                    upper_bound=45.0,
                    transform="log1p",
                ),
            ),
            "instr_chars": LinearBinning(
                min_val=0.0,
                max_val=2500.0,
                num_bins=6,
                model_transform=BehaviorModelTransform(
                    lower_bound=0.0,
                    upper_bound=2500.0,
                    transform="log1p",
                ),
            ),
        }
    )
    parent = Program(code="def parent(): return 1", iteration=0)
    parent.metrics = {
        "is_valid": 1.0,
        "fitness": 0.6,
        "hop_depth": 2.0,
        "passages_fetched": 10.0,
        "instr_chars": 500.0,
    }
    parent.metadata["current_island"] = "main"
    peer = Program(code="def peer(): return 2", iteration=3)
    peer.metrics = {
        "is_valid": 1.0,
        "fitness": 0.4,
        "hop_depth": 4.0,
        "passages_fetched": 30.0,
        "instr_chars": 1500.0,
    }

    async def snapshot_archive_state():
        return [parent, peer], space.model_copy(deep=True)

    island = SimpleNamespace(
        config=SimpleNamespace(island_id="main", behavior_space=space),
        snapshot_archive_state=AsyncMock(side_effect=snapshot_archive_state),
    )
    strategy = SimpleNamespace(islands={"main": island}, generation=8)
    metrics = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            )
        }
    )
    source = MapElitesContextSource(
        strategy=strategy,
        metrics_context=metrics,
        environment=environment,
        trajectory_id_source=SimpleNamespace(trajectory_id="typed-run"),
        config=MapContextConfig(),
        credit=LineageCreditConfig(),
    )

    assert source.behavior_keys == (
        "hop_depth",
        "passages_fetched",
        "instr_chars",
    )
    context = await source.snapshot(parent)

    assert context.parent_iteration == 0
    assert context.parent_generation == 1
    assert context.map_elites.strategy_generation == 8
    assert context.map_elites.archive_size == 2
    assert context.map_elites.total_cells == 150
    assert context.map_elites.coverage == pytest.approx(2 / 150)
    instr = next(
        row for row in context.map_elites.coordinates if row.key == "instr_chars"
    )
    assert instr.raw_value == 500.0
    assert instr.semantic_normalized == pytest.approx(
        math.log1p(500.0) / math.log1p(2500.0)
    )
    assert context.map_elites.behavior_schema_hash != (
        context.map_elites.archive_fingerprint
    )

    parent.metrics["hop_depth"] = 4.5
    parent.metrics["passages_fetched"] = 35.0
    updated = await source.snapshot(parent)
    old_hop = next(
        row for row in context.map_elites.coordinates if row.key == "hop_depth"
    )
    new_hop = next(
        row for row in updated.map_elites.coordinates if row.key == "hop_depth"
    )
    assert new_hop.raw_value == 4.5
    assert new_hop.dynamic_normalized != old_hop.dynamic_normalized
    assert updated.map_elites.parent_cell != context.map_elites.parent_cell
    assert (
        updated.map_elites.semantic_schema_hash
        == context.map_elites.semantic_schema_hash
    )


@pytest.mark.asyncio
async def test_dynamic_cell_change_does_not_change_bayesian_features(
    environment: EnvironmentFingerprint,
) -> None:
    """A live rebin is audit state, never a change of Bayesian identity."""

    space = DynamicBehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=100.0, num_bins=10)}
    )
    parent = Program(code="def parent(): return 1", iteration=4)
    parent.metrics = {"fitness": 0.6, "x": 20.0}
    parent.metadata["current_island"] = "main"

    async def snapshot_archive_state():
        return [parent], space.model_copy(deep=True)

    island = SimpleNamespace(
        config=SimpleNamespace(island_id="main", behavior_space=space),
        snapshot_archive_state=AsyncMock(side_effect=snapshot_archive_state),
    )
    source = MapElitesContextSource(
        strategy=SimpleNamespace(islands={"main": island}, generation=8),
        metrics_context=MetricsContext(
            specs={
                "fitness": MetricSpec(
                    description="fitness",
                    is_primary=True,
                    higher_is_better=True,
                    lower_bound=0.0,
                    upper_bound=1.0,
                )
            }
        ),
        environment=environment,
        trajectory_id_source=SimpleNamespace(trajectory_id="dynamic-run"),
        config=MapContextConfig(),
        credit=LineageCreditConfig(),
    )
    feature_space = HierarchicalFeatureMap(
        config=FeatureConfig(behavior_keys=("x",))
    ).space(())

    before = await source.snapshot(parent)
    space.bins["x"].update_bounds(new_min=10.0, new_max=30.0)
    after = await source.snapshot(parent)

    assert before.map_elites.parent_cell == (2,)
    assert after.map_elites.parent_cell == (5,)
    assert before.map_elites.behavior_schema_hash != (
        after.map_elites.behavior_schema_hash
    )
    assert before.map_elites.semantic_schema_hash == (
        after.map_elites.semantic_schema_hash
    )
    assert before.map_elites.coordinates[0].semantic_normalized == (
        after.map_elites.coordinates[0].semantic_normalized
    )
    assert before.map_elites.coordinates[0].dynamic_normalized != (
        after.map_elites.coordinates[0].dynamic_normalized
    )
    replayed_after = EvolutionContext.model_validate_json(
        after.model_dump_json(exclude_computed_fields=True)
    )
    assert np.array_equal(
        feature_space.context_features(before),
        feature_space.context_features(replayed_after),
    )


def test_context_source_rejects_different_stable_transforms_across_islands(
    environment: EnvironmentFingerprint,
) -> None:
    first = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=10.0, num_bins=5)}
    )
    second = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=20.0, num_bins=5)}
    )
    source = MapElitesContextSource(
        strategy=SimpleNamespace(
            islands={
                "first": SimpleNamespace(config=SimpleNamespace(behavior_space=first)),
                "second": SimpleNamespace(
                    config=SimpleNamespace(behavior_space=second)
                ),
            }
        ),
        metrics_context=AsyncMock(),
        environment=environment,
        trajectory_id_source=SimpleNamespace(trajectory_id="schema-run"),
        config=MapContextConfig(),
        credit=LineageCreditConfig(),
    )

    with pytest.raises(ValueError, match="shared stable behavior transform"):
        _ = source.behavior_keys


def observation(
    context: EvolutionContext,
    card: CardSnapshot,
    *,
    decision_id: str,
    treatment: bool,
    value: float,
    run_id: str,
) -> CausalObservation:
    return CausalObservation(
        decision_id=decision_id,
        event_ordinal=0,
        card=card,
        context=context.model_copy(update={"run_id": run_id}),
        treatment=treatment,
        card_used=treatment,
        offer_propensity=0.5,
        proposal_propensity=1.0,
        joint_action_propensity=0.5,
        status="outcome",
        measurement=OutcomeMeasurement(value=value, se=None, kind="scalar"),
        reward_q_hat_control=0.0,
        reward_q_hat_treated=0.0,
        risk_q_hat_control=0.1,
        risk_q_hat_treated=0.2,
    )


def test_conditional_offer_ope_exposes_only_predecision_data(
    evolution_context: EvolutionContext,
    revisions: tuple[CardSnapshot, CardSnapshot],
) -> None:
    card = revisions[0]
    rows = (
        observation(
            evolution_context,
            card,
            decision_id="treated",
            treatment=True,
            value=1.0,
            run_id="run-a",
        ),
        observation(
            evolution_context,
            card,
            decision_id="control",
            treatment=False,
            value=0.2,
            run_id="run-b",
        ),
    )
    seen: list[PreDecisionUnit] = []

    def target(unit: PreDecisionUnit) -> float:
        seen.append(unit)
        return 0.75

    report = ConditionalOfferDREvaluator().evaluate_reward(
        rows, target_offer_probability=target
    )

    assert report.estimate == pytest.approx(0.8)
    assert report.clusters == 2
    assert report.cluster_robust_se is not None
    assert report.effective_sample_size == pytest.approx(1.6)
    assert set(PreDecisionUnit.model_fields) == {
        "decision_id",
        "card",
        "context",
        "behavior_offer_probability",
    }
    assert len(seen) == 2

    single_run = ConditionalOfferDREvaluator().evaluate_reward(
        rows[:1], target_offer_probability=lambda _: 0.75
    )
    assert single_run.cluster_robust_se is None


def test_memory_v2_rejects_crossover_before_instantiation() -> None:
    cfg = OmegaConf.create(
        {
            "num_parents": 2,
            "memory": {"capabilities": {"causal_v2": True}},
            "pipeline": {
                "id": "memory_guided",
                "routes_program_metadata": True,
            },
        }
    )

    with pytest.raises(ValueError, match="requires num_parents=1"):
        validate_memory_v2_scope(cfg)


def test_memory_v2_requires_fresh_decision_context() -> None:
    cfg = OmegaConf.create(
        {
            "num_parents": 1,
            "memory": {
                "capabilities": {"causal_v2": True},
                "coalesce_refresh": False,
            },
            "pipeline": {
                "id": "memory_guided",
                "routes_program_metadata": True,
            },
            "pipeline_builder": {"fresh_context_reorder": False},
        }
    )

    with pytest.raises(ValueError, match="fresh_context_reorder=true"):
        validate_memory_v2_scope(cfg)


def test_memory_v2_production_surface_composes_with_hydra() -> None:
    config_dir = Path(__file__).parents[2] / "config"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "storage=disk",
                    "problem.name=chains/hover/full7_vectorized",
                    "archive_selector=paired_bootstrap",
                    "program_format=json_document",
                    "mutation=carl_with_retrieval_tools",
                    "algorithm=chains_bd3d",
                    "enable_chain_structural_metrics=true",
                    "num_parents=1",
                    "pipeline=memory_guided",
                    "memory=v2",
                    "memory/write=live",
                    "memory/llm=qwen_instruct",
                    "checkpoint_dir=/tmp/memory-v2-compose",
                ],
            )
        validate_memory_v2_scope(cfg)
        assert cfg.memory.capabilities.causal_v2 is True
        assert cfg.memory.coalesce_refresh is False
        assert cfg.engine_config.terminal_drain_timeout_s == cfg.dag_timeout
        assert cfg.post_step_hook.refresh_every == 5
        assert cfg.memory.candidate_source._target_ == (
            f"{WholeBankCandidateSource.__module__}."
            f"{WholeBankCandidateSource.__qualname__}"
        )
        raw_candidate_source = OmegaConf.to_container(
            cfg.memory.candidate_source, resolve=False
        )
        assert isinstance(raw_candidate_source, dict)
        assert raw_candidate_source["applicability"] == "${ref:memory.applicability}"
        assert cfg.memory.applicability._target_ == (
            f"{AgenticApplicabilityProvider.__module__}."
            f"{AgenticApplicabilityProvider.__qualname__}"
        )
        assert not {
            "max_candidates",
            "exploration_candidates",
            "selection_logic",
        }.intersection(cfg.memory.candidate_source)
        assert cfg.memory.feature_config.card_kind_contrast is True
        assert cfg.memory.feature_config.retrieval_applicability_contrast is True
        assert cfg.memory.safety.gate_mode == "exclude_confident_incremental_harm"
        assert cfg.memory.safety.max_treated_invalid_probability is None
        assert cfg.memory.safety.max_incremental_invalid_probability == pytest.approx(
            0.10
        )
        assert cfg.memory.policy_config.offer_probability == pytest.approx(0.70)
        assert (
            cfg.memory.policy_config.proposal_exploration_probability
            == pytest.approx(0.05)
        )
        assert cfg.memory.policy_config.max_pending_per_card == 2
        assert cfg.memory.writer.dedup_policy.across_tasks is False
        assert cfg.memory.posterior_config.reference_offer_probability == pytest.approx(
            0.70
        )
        assert cfg.memory.posterior_config.reward_residual_sd_bounds == [0.01, 5.0]
        assert "max_task_cards" not in cfg.memory.writer
        assert cfg.memory.ledger._target_.endswith("SqliteCausalLedger")
        # ``compose`` does not populate HydraConfig's runtime choice table.
        # Production ``@hydra.main`` does, so substitute only the remaining
        # runtime-owned paths/provenance before exercising construction here.
        cfg.log_dir = "/tmp/memory-v2-compose/logs"
        cfg.disk_metrics_config.root_dir = "/tmp/memory-v2-compose/metrics"
        cfg.program_storage.config.root_dir = "/tmp/memory-v2-compose/storage"
        cfg.problem.dir = str(Path(__file__).parents[2] / "problems" / cfg.problem.name)
        cfg.memory.environment.algorithm = "chains_bd3d"
        instantiated_environment = instantiate(cfg.memory.environment)
        assert isinstance(instantiated_environment, EnvironmentFingerprint)
        assert (
            instantiated_environment.mutation_operator is StructuredDiffMutationOperator
        )
        raw_engine = OmegaConf.to_container(cfg.evolution_engine, resolve=False)
        assert isinstance(raw_engine, dict)
        assert raw_engine["memory_outcome_sink"] == "${ref:memory.outcome_sink}"

        cfg.memory.policy_config.proposal_exploration_probability = 0.0
        with pytest.raises(ValueError, match="proposal_exploration_probability"):
            validate_memory_v2_scope(cfg)
        cfg.memory.policy_config.proposal_exploration_probability = 0.05

        cfg.memory.policy_config.offer_probability = 0.01
        with pytest.raises(ValueError, match="offer_probability"):
            validate_memory_v2_scope(cfg)
        cfg.memory.policy_config.offer_probability = 0.7

        cfg.memory.candidate_source._target_ = "not.the.full.bank"
        with pytest.raises(ValueError, match="WholeBankCandidateSource"):
            validate_memory_v2_scope(cfg)
        cfg.memory.candidate_source._target_ = (
            f"{WholeBankCandidateSource.__module__}."
            f"{WholeBankCandidateSource.__qualname__}"
        )

        evictor_target = cfg.memory.evictor._target_
        assert evictor_target == (
            f"{CausalRetirementEvictor.__module__}."
            f"{CausalRetirementEvictor.__qualname__}"
        )
        cfg.memory.evictor._target_ = "gigaevo.memory.write.eviction.NullEvictor"
        validate_memory_v2_scope(cfg)
        cfg.memory.evictor._target_ = "not.a.causal.evictor"
        with pytest.raises(ValueError, match="causal-retirement"):
            validate_memory_v2_scope(cfg)
        cfg.memory.evictor._target_ = evictor_target
        cfg.memory.candidate_source.allow_cross_task = True
        with pytest.raises(ValueError, match="allow_cross_task=false"):
            validate_memory_v2_scope(cfg)
        cfg.memory.candidate_source.allow_cross_task = False

        cfg.memory.feature_config.retrieval_applicability_contrast = False
        with pytest.raises(ValueError, match="retrieval_applicability_contrast"):
            validate_memory_v2_scope(cfg)
        cfg.memory.feature_config.retrieval_applicability_contrast = True

        instantiated_evictor = instantiate(cfg.memory.evictor)
        assert isinstance(instantiated_evictor, CausalRetirementEvictor)
        assert instantiated_evictor.max_viability_probability == pytest.approx(
            instantiated_evictor.safety.alpha
        )
        assert instantiated_evictor.practical_effect_quantile == pytest.approx(0.10)
        instantiated_evictor.ledger.close()
    finally:
        GlobalHydra.instance().clear()


def test_memory_v2_null_retirement_ablation_composes() -> None:
    config_dir = Path(__file__).parents[2] / "config"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "problem.name=heilbron",
                    "memory=v2",
                    "memory/evictor=none",
                ],
            )
        validate_memory_v2_scope(cfg)
        assert isinstance(instantiate(cfg.memory.evictor), NullEvictor)
    finally:
        GlobalHydra.instance().clear()


def test_memory_v2_multitask_is_one_preset_plus_a_bank_path(tmp_path) -> None:
    config_dir = Path(__file__).parents[2] / "config"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "problem.name=heilbron",
                    "memory=v2_multitask",
                    "memory_task_key=heilbron/variant",
                    f"memory_bank_dir={tmp_path}",
                ],
            )
        validate_memory_v2_scope(cfg)
        assert cfg.memory.candidate_source.allow_cross_task is True
        assert isinstance(instantiate(cfg.memory.evictor), NullEvictor)
        assert cfg.memory.causal_writer_updater.record_use_trials is True
        assert cfg.memory.writer.dedup_policy.across_tasks is True
        assert cfg.memory.store.config.path == str(tmp_path)
        assert cfg.selection_leases.path == f"{tmp_path}/selection_leases.json"
        assert cfg.memory.environment.task_key == "heilbron/variant"
        assert cfg.memory.provider.task_key == "heilbron/variant"
        assert cfg.memory.context_model.task_key == "heilbron/variant"
        assert cfg.memory.writer.task_key == "heilbron/variant"
        assert isinstance(
            cfg.memory.provider.cross_task_usefulness,
            CrossTaskUsefulnessConfig,
        )
    finally:
        GlobalHydra.instance().clear()


def test_memory_v2_multitask_accepts_parameterized_task_identity() -> None:
    config_dir = Path(__file__).parents[2] / "config"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            configs = [
                compose(
                    config_name="config",
                    overrides=[
                        "problem.name=dag_tab",
                        "mutation=structured_diff_dag_tab",
                        f"problem.dataset={dataset}",
                        "memory=v2_multitask",
                        f"memory_task_key=dag_tab/{dataset}",
                    ],
                )
                for dataset in ("california", "adult")
            ]

        assert [cfg.memory.environment.task_key for cfg in configs] == [
            "dag_tab/california",
            "dag_tab/adult",
        ]
        for cfg in configs:
            expected = cfg.memory.environment.task_key
            assert cfg.memory.provider.task_key == expected
            assert cfg.memory.context_model.task_key == expected
            assert cfg.memory.writer.task_key == expected
    finally:
        GlobalHydra.instance().clear()


def test_memory_v2_multitask_composes_with_embedding_prior() -> None:
    config_dir = Path(__file__).parents[2] / "config"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "problem.name=heilbron",
                    "memory=v2_multitask",
                    "memory/embedding_prior=linear",
                ],
            )
        validate_memory_v2_scope(cfg)
        assert cfg.memory.evictor._target_.endswith("NullEvictor")
        assert "embedding_prior" in cfg.memory.evictor
        assert "card_embedder" in cfg.memory.evictor
        # ``compose`` lacks Hydra's runtime choice table; the evictor only needs
        # the embedding portion of this config, so pin the unrelated live axes.
        cfg.memory.feature_config.behavior_keys = []
        assert isinstance(instantiate(cfg.memory.evictor), NullEvictor)
    finally:
        GlobalHydra.instance().clear()


def test_memory_v2_whole_bank_control_composes_with_hydra() -> None:
    config_dir = Path(__file__).parents[2] / "config"
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "problem.name=heilbron",
                    "memory=v2",
                    "memory/applicability=none",
                ],
            )
        validate_memory_v2_scope(cfg)
        assert cfg.memory.candidate_source._target_.endswith("WholeBankCandidateSource")
        assert cfg.memory.applicability._target_ == (
            f"{NullApplicabilityProvider.__module__}."
            f"{NullApplicabilityProvider.__qualname__}"
        )
        assert cfg.memory.feature_config.retrieval_applicability_contrast is False
    finally:
        GlobalHydra.instance().clear()
